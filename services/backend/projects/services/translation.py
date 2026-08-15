"""Translation and voice-preserving dubbing workflow.

Handles:
- Language detection and translation
- Transcript alignment between source and target
- Voice profile extraction
- Dubbed video rendering with original speaker voice
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from django.conf import settings

from ..models import Project, Translation, VoiceProfile
from . import model_client

log = logging.getLogger(__name__)


def create_translation(project: Project, target_language: str) -> Translation:
    """Create a new translation for a project.
    
    Queues the translation task to the model server.
    """
    if project.status != Project.Status.READY:
        raise ValueError(f"Project must be ready, not {project.status}")
    
    # Check if translation already exists
    translation, created = Translation.objects.get_or_create(
        project=project,
        target_language=target_language,
        defaults={
            "source_language": project.language or "en",
            "status": Translation.Status.PENDING,
        }
    )
    
    return translation


def translate_transcript(project: Project, target_language: str) -> dict:
    """Translate a project's transcript to target language.
    
    Returns a dict with:
    - segments: list of {index, original_text, translated_text}
    - segments_text: dict {segment_index: translated_text}
    - metadata: {source_language, target_language, confidence}
    """
    source_language = project.language or "en"
    
    # Get original segments
    segments = list(project.segments.order_by("index").values(
        "index", "text"
    ))
    
    # Call model server for translation
    # This would be a new endpoint in the model service
    result = model_client.translate_segments(
        project.id,
        [seg["text"] for seg in segments],
        source_language=source_language,
        target_language=target_language,
    )
    
    segments_text = {}
    translated_segments = []
    
    translations = result.get("translations", [])
    for i, segment in enumerate(segments):
        translated = translations[i] if i < len(translations) else segment["text"]
        # JSONField round-trips through JSON, which only has string object
        # keys — an int key here would silently stop matching after the next
        # DB read, so every writer and reader of this dict must agree on str.
        segments_text[str(segment["index"])] = translated
        translated_segments.append({
            "index": segment["index"],
            "original_text": segment["text"],
            "translated_text": translated,
        })
    
    return {
        "segments": translated_segments,
        "segments_text": segments_text,
        "metadata": {
            "source_language": source_language,
            "target_language": target_language,
            "confidence": result.get("confidence", 0.8),
        }
    }


def extract_voice_profile(project: Project) -> VoiceProfile:
    """Extract and store voice profile from project's master audio.
    
    This calls the model server to analyze the original speaker's voice
    and create an embedding that can be used for voice-preserving synthesis.
    """
    if not project.master_path.exists():
        raise FileNotFoundError(f"Master audio not found for {project.id}")
    
    # Check if profile already exists
    profile, created = VoiceProfile.objects.get_or_create(
        project=project,
        defaults={
            "embedding": {},
            "pitch_info": {},
            "spectral_features": {},
            "supported_languages": [project.language or "en"],
        }
    )
    
    if not created and profile.embedding:
        # Profile already extracted
        return profile
    
    # Extract voice profile from model server
    result = model_client.extract_voice_profile(
        project.id,
        project.master_path,
    )
    
    profile.embedding = result.get("embedding", {})
    profile.pitch_info = result.get("pitch_info", {})
    profile.spectral_features = result.get("spectral_features", {})
    profile.supported_languages = result.get("supported_languages", [project.language or "en"])
    profile.save()
    
    return profile


def apply_translation_edits(translation: Translation, edits: list[dict]) -> dict:
    """Apply user edits to a translation.

    Edits can be:
    - {type: "segment", index: 0, text: "new text", keepOriginal: False}
    - {type: "segment", index: 0, deleted: True} — cut the sentence entirely,
      audio and video, rather than replacing its text
    - {type: "word", segment_index: 0, word_index: 2, text: "new word"}

    Returns the final translated text dict and edit list.
    """
    current_edits = translation.edits or []
    final_text = translation.translated_text or {}

    for edit in edits:
        edit_type = edit.get("type", "segment")

        if edit_type == "segment":
            seg_index = edit.get("index")
            if not edit.get("deleted"):
                final_text[str(seg_index)] = edit.get("text", "")
            current_edits.append(edit)

        elif edit_type == "word":
            # Word-level edits are tracked for finer render control
            seg_index = edit.get("segment_index")
            word_index = edit.get("word_index")
            text = edit.get("text", "")
            current_edits.append(edit)

    translation.edits = current_edits
    translation.translated_text = final_text
    translation.save()

    return {
        "translated_text": final_text,
        "edits": current_edits,
    }


def build_dub_edl(translation: Translation) -> dict:
    """Build EDL for dubbing, comparing original and translated transcripts.

    Returns:
    - segments: list of {type: "keep" | "replace" | "delete", index, ...}
    - keep_original_indices: indices where original audio+video are kept
    - replace_text: dict {segment_index: new_text} for segments to regenerate
    - deleted_indices: indices cut entirely — no audio, no video, a true jump
      cut straight to the next kept/replaced segment
    """
    project = translation.project

    # Get original segments
    original_segments = list(project.segments.order_by("index"))

    # Get user edits or full translation
    user_edits = {e["index"]: e for e in translation.edits if e.get("type") == "segment"}
    translated_text = translation.translated_text or {}

    keep_original_indices = []
    replace_indices = {}
    deleted_indices = []
    segments_edl = []

    for segment in original_segments:
        idx = segment.index
        orig_text = segment.text
        edit = user_edits.get(idx)

        if edit and edit.get("deleted"):
            deleted_indices.append(idx)
            segments_edl.append({"type": "delete", "index": idx, "text": orig_text})
        elif edit and edit.get("keepOriginal"):
            keep_original_indices.append(idx)
            segments_edl.append({
                "type": "keep",
                "index": idx,
                "text": orig_text,
                "start": segment.start,
                "end": segment.end,
            })
        else:
            # Use translation or user edit
            new_text = user_edits.get(idx, {}).get("text") or translated_text.get(str(idx), orig_text)
            if new_text != orig_text:
                replace_indices[idx] = new_text
                segments_edl.append({
                    "type": "replace",
                    "index": idx,
                    "original_text": orig_text,
                    "new_text": new_text,
                })
            else:
                keep_original_indices.append(idx)
                segments_edl.append({
                    "type": "keep",
                    "index": idx,
                    "text": orig_text,
                })

    return {
        "segments": segments_edl,
        "keep_original_indices": keep_original_indices,
        "replace_text": replace_indices,
        "deleted_indices": deleted_indices,
    }
