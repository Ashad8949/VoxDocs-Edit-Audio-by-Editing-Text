"""Dub rendering: combine original video with translated/edited audio.

Creates final MP4/WebM output with:
- Original video track
- New audio track (original kept where unchanged, synthesized where edited)
"""

from __future__ import annotations

import logging
from pathlib import Path

from django.conf import settings

from ..models import DubRender, Project, Translation
from . import ffmpeg, model_client
from .render import render_ops, RenderOp

log = logging.getLogger(__name__)


def perform_dub_render(dub_render: DubRender) -> DubRender:
    """Execute a queued dub render. Called from Celery task."""
    translation = dub_render.translation
    project = translation.project
    
    if project.status != Project.Status.READY:
        raise ValueError(f"Project must be ready, not {project.status}")
    
    if not project.has_video:
        raise ValueError("Project has no video to dub")
    
    # Build the dubbing EDL
    from .translation import build_dub_edl
    edl = build_dub_edl(translation)
    
    # Create work directory
    workdir = dub_render.directory
    workdir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Synthesize only the changed segments
        synthesis_result = _synthesize_dubbed_segments(
            project, translation, edl["replace_text"]
        )

        # Build audio render ops from the EDL. Every op is locked to its
        # original segment's exact duration (silence for deletions, a
        # time-stretch for replacements) so the audio track's total length
        # always matches the video's, which is never touched.
        ops = _build_dub_ops(project, translation, edl, synthesis_result)

        # Render audio
        audio_path = workdir / "dubbed_audio.wav"
        audio_info = render_ops(
            project.master_path,
            ops,
            audio_path,
            fade=settings.VOXDOCS.get("SEAM_FADE", 0.010),
            sample_rate=settings.VOXDOCS.get("RENDER_SAMPLE_RATE", 48000),
        )

        # Mux into the original, untouched video.
        output_format = (dub_render.format or "mp4").lower()
        output_path = workdir / f"dubbed.{output_format}"

        ffmpeg.mux_audio_to_video(
            project.source_path(), audio_path, output_path, output_format=output_format,
        )

        # Update render
        file_size = output_path.stat().st_size
        dub_render.file = output_path.name
        dub_render.bytes = file_size
        dub_render.duration = project.duration
        dub_render.status = DubRender.Status.READY
        dub_render.stats = audio_info
        dub_render.synthesis = synthesis_result
        dub_render.save()
        
        log.info("dub render %s completed successfully", dub_render.id)
        
    except Exception as exc:
        log.error("dub render %s failed: %s", dub_render.id, exc)
        dub_render.status = DubRender.Status.FAILED
        dub_render.error = str(exc)[:2000]
        dub_render.save()
        raise
    
    return dub_render


def _synthesize_dubbed_segments(
    project: Project,
    translation: Translation,
    replace_text: dict[int, str],
) -> dict:
    """Synthesize only the segments that changed.
    
    Uses the project's voice profile to generate speech in the target language
    that sounds like the original speaker.
    
    Returns dict of {segment_index: synthesis_result}
    """
    if not replace_text:
        return {}
    
    # Get the voice profile
    try:
        voice_profile = project.voice_profile
    except Exception:
        # If no profile exists, we'll use generic TTS
        voice_profile = None
    
    synthesis = {}
    target_language = translation.target_language
    
    # Get segments for context
    segments = {s.index: s for s in project.segments.all()}
    
    for seg_index, new_text in replace_text.items():
        segment = segments.get(seg_index)
        if not segment:
            continue
        
        # Get context from surrounding segments
        prev_seg = segments.get(seg_index - 1)
        next_seg = segments.get(seg_index + 1)
        
        context_before = prev_seg.text if prev_seg else ""
        context_after = next_seg.text if next_seg else ""
        
        # Synthesize using voice-preserving model
        result = model_client.synthesize_with_voice(
            project.id,
            text=new_text,
            target_language=target_language,
            context_before=context_before,
            context_after=context_after,
        )
        
        synthesis[seg_index] = result
    
    return synthesis


def _build_dub_ops(
    project: Project,
    translation: Translation,
    edl: dict,
    synthesis: dict,
) -> list[RenderOp]:
    """Build render ops for dubbing. The video is never touched, so every op
    must fill exactly its segment's original duration:

    - Kept segments: the original audio, an exact fit by construction.
    - Replaced segments: synthesized audio, time-stretched (see
      ``target_duration`` on ``RenderOp``) to exactly fill the original slot
      — this is how real dubbing keeps everything downstream in sync without
      re-cutting picture.
    - Deleted segments: silence for exactly that duration. The speaker's lips
      keep moving with nothing said, which is honest about what "delete the
      audio" without touching video actually means.
    """
    ops: list[RenderOp] = []
    segments = list(project.segments.order_by("index"))
    deleted = set(edl.get("deleted_indices", []))

    for segment in segments:
        idx = segment.index
        original_span = segment.end - segment.start

        if idx in deleted:
            ops.append(RenderOp(kind="silence", duration=original_span, label=f"deleted:{idx}"))
            continue

        if idx in edl["keep_original_indices"]:
            ops.append(RenderOp(
                kind="source",
                start=segment.start,
                end=segment.end,
                label=f"keep:{idx}",
            ))
            continue

        synth_result = synthesis.get(idx)
        if synth_result is None:
            continue

        if synth_result.get("type") == "source":
            # Model returned a reference to original audio at new timing —
            # e.g. a word/phrase reused verbatim from elsewhere in this same
            # recording.
            start = float(synth_result.get("start", segment.start))
            end = float(synth_result.get("end", segment.end))
            ops.append(RenderOp(
                kind="source", start=start, end=end,
                target_duration=original_span, label=f"synth_reuse:{idx}",
            ))
        elif synth_result.get("type") == "audio" and synth_result.get("data"):
            import base64

            inline_dir = project.directory / "dub_synthesis"
            inline_dir.mkdir(parents=True, exist_ok=True)

            audio_path = inline_dir / f"synth-{idx}.wav"
            audio_path.write_bytes(base64.b64decode(synth_result["data"]))

            ops.append(RenderOp(
                kind="file", file=audio_path,
                target_duration=original_span, label=f"synth:{idx}",
            ))

    return ops
