"""The two long-running operations: ingest and render.

Kept out of both the HTTP layer and the Celery layer so they can be called
directly from a test or a management command, and so the task functions stay
thin enough to be obviously correct.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from django.conf import settings
from django.db import transaction

from edl import CopySegment, SynthSegment, build_edl, diff_transcript, identity_tokens

from ..models import Project, Render, Segment, Word
from . import ffmpeg, model_client
from .render import encode_audio, expand_segments, op_durations, render_ops, render_video

log = logging.getLogger(__name__)

AUDIO_FORMATS = {"wav", "mp3", "m4a"}


class ValidationError(Exception):
    """The request cannot be satisfied as asked. Surfaces as a 400."""

    status = 400


# ------------------------------------------------------------------ ingest

def ingest(project: Project, language: str | None = None) -> Project:
    """Decode, measure, transcribe.

    The only place the media is inspected; everything downstream works from the
    master and the stored transcript.
    """
    source = project.source_path()
    if source is None:
        raise ValidationError("the uploaded file is missing")

    info = ffmpeg.probe(source)
    if not info["has_audio"]:
        raise ValidationError("this file has no audio track to transcribe")

    project.duration = info["duration"]
    project.has_video = info["has_video"]
    project.video_width = info["width"]
    project.video_height = info["height"]
    project.video_fps = round(info["fps"], 3)
    project.status = Project.Status.TRANSCRIBING
    project.save(update_fields=[
        "duration", "has_video", "video_width", "video_height", "video_fps",
        "status", "updated_at",
    ])

    project.directory.mkdir(parents=True, exist_ok=True)

    # One canonical master drives every later cut.
    ffmpeg.normalize_to_master(source, project.master_path)
    try:
        ffmpeg.make_preview_audio(source, project.preview_path)
        project.preview_failed = False
    except ffmpeg.FfmpegError:
        # A missing preview only costs the editor its scrub audio; not fatal.
        log.warning("preview encode failed for %s", project.id)
        project.preview_failed = True

    result = model_client.transcribe(project.master_path, project.id, language)

    words = result.get("words") or []
    segments = result.get("segments") or []
    segment_of = {}
    for index, segment in enumerate(segments):
        for i in range(segment.get("first_word", 0), segment.get("last_word", -1) + 1):
            segment_of[i] = index

    with transaction.atomic():
        project.words.all().delete()
        project.segments.all().delete()
        Word.objects.bulk_create([
            Word(
                project=project,
                index=i,
                token_id=str(word.get("id") or f"w{i}"),
                text=str(word.get("text", ""))[:255],
                start=float(word.get("start", 0.0)),
                end=float(word.get("end", 0.0)),
                confidence=float(word.get("confidence", 1.0)),
                segment=segment_of.get(i, 0),
            )
            for i, word in enumerate(words)
        ], batch_size=1000)
        Segment.objects.bulk_create([
            Segment(
                project=project,
                index=index,
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=segment.get("text", ""),
                first_word=int(segment.get("first_word", 0)),
                last_word=int(segment.get("last_word", 0)),
            )
            for index, segment in enumerate(segments)
        ], batch_size=500)

        project.language = result.get("language") or "en"
        project.asr_backend = result.get("backend") or "unknown"
        project.voice = result.get("voice")
        project.duration = result.get("duration") or project.duration
        project.status = Project.Status.READY
        project.error = ""
        project.save()

    # The envelope is large and only the editor wants it, so it stays a file
    # rather than bloating every read of the project row.
    project.envelope_path.write_text(
        json.dumps(result.get("envelope") or {"fps": 100, "rms": []}), encoding="utf-8"
    )
    return project


def load_envelope(project: Project) -> dict | None:
    try:
        return json.loads(project.envelope_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------- edits

def resolve_tokens(project: Project, payload: dict) -> list[dict]:
    """Turn a request body into an edit-token list.

    Two shapes are accepted: the structured list the editor maintains (exact, no
    guessing), and plain text (aligned against the original). The editor uses the
    first; pastes, scripts and API clients use the second.
    """
    raw_tokens = payload.get("tokens")
    if isinstance(raw_tokens, list):
        tokens = []
        for token in raw_tokens:
            if isinstance(token, dict) and isinstance(token.get("ref"), str):
                tokens.append({"ref": token["ref"]})
            elif isinstance(token, dict) and isinstance(token.get("insert"), str):
                tokens.append({"insert": token["insert"]})
            else:
                raise ValidationError("each token must be {ref:string} or {insert:string}")
        return tokens

    text = payload.get("text")
    if isinstance(text, str):
        return diff_transcript(project.edl_words(), text)

    return identity_tokens(project.edl_words())


def plan_edit(project: Project, tokens: list[dict]):
    """Build the EDL without rendering, for live duration feedback."""
    return build_edl(
        project.edl_words(), tokens,
        duration=project.duration,
        envelope=load_envelope(project),
    )


# ------------------------------------------------------------------ render

def perform_render(render: Render) -> Render:
    """Execute a queued render. Called from the Celery task."""
    project = render.project
    fmt = (render.format or "wav").lower()
    want_video = bool(render.with_video and project.has_video)
    if not want_video and fmt not in AUDIO_FORMATS:
        raise ValidationError(f'unsupported format "{fmt}" (use wav, mp3 or m4a)')

    edl = plan_edit(project, render.tokens)

    # Ask the model server for every insertion at once.
    synth_indices = []
    items = []
    for index, segment in enumerate(edl.segments):
        if not isinstance(segment, SynthSegment):
            continue
        synth_indices.append(index)
        items.append({
            "text": segment.text,
            "context_before": segment.context_before,
            "context_after": segment.context_after,
            "lead_gap": segment.lead_gap,
            "trail_gap": segment.trail_gap,
            "quality": render.quality,
        })

    synthesis: dict[int, dict] = {}
    if items:
        def reseed():
            model_client.put_voice_profile(
                project.id,
                [
                    {"text": w["text"], "start": w["start"], "end": w["end"],
                     "confidence": w.get("confidence", 1.0)}
                    for w in project.words.values("text", "start", "end", "confidence")
                ],
                project.duration,
                project.master_path,
            )

        response = model_client.synthesize_batch(project.id, items, reseed)
        for i, result in enumerate(response.get("results", [])):
            synthesis[synth_indices[i]] = result

    workdir = render.directory
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        ops, warnings = expand_segments(edl.segments, synthesis, workdir)

        master_out = workdir / "render.wav"
        rendered = render_ops(project.master_path, ops, master_out)

        # Video needs to know how long each insertion actually turned out.
        synth_durations: dict[int, float] = {}
        if want_video:
            durations = op_durations(ops)
            cursor = 0
            for index, segment in enumerate(edl.segments):
                if isinstance(segment, CopySegment):
                    cursor += 1
                    continue
                units = synthesis.get(index, {}).get("units", [])
                count = sum(
                    1 for u in units
                    if u.get("type") != "silence" or float(u.get("duration", 0)) > 0.001
                )
                total = 0.0
                for _ in range(count):
                    if cursor >= len(durations):
                        break
                    total += durations[cursor]
                    cursor += 1
                synth_durations[index] = total

        if want_video:
            output = workdir / "output.mp4"
            output_format = "mp4"
            source = project.source_path()
            if source is None:
                raise ValidationError("the original media file is no longer available")
            render_video(source, master_out, edl.segments, synth_durations, output)
        else:
            output_format = fmt
            output = workdir / f"output.{fmt}"
            encode_audio(master_out, output, fmt)

        # The intermediate master and inline TTS clips have served their purpose.
        master_out.unlink(missing_ok=True)
        for entry in workdir.glob("tts-*"):
            entry.unlink(missing_ok=True)

        render.status = Render.Status.READY
        render.format = output_format
        render.file = output.name
        render.bytes = output.stat().st_size
        render.duration = rendered["duration"]
        render.pieces = rendered["pieces"]
        render.stats = edl.stats.to_json()
        render.warnings = warnings
        render.synthesis = summarise_synthesis(synthesis)
        render.error = ""
        render.save()
        return render
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise


def summarise_synthesis(synthesis: dict[int, dict]) -> dict:
    covered, generated, missing = [], [], []
    backends: list[str] = []
    for result in synthesis.values():
        covered += result.get("covered", [])
        generated += result.get("generated", [])
        missing += result.get("missing", [])
        for backend in result.get("backends", []):
            if backend not in backends:
                backends.append(backend)

    total = len(covered) + len(generated) + len(missing)
    return {
        "words": total,
        "fromVoiceBank": len(covered),
        "fromTts": len(generated),
        "missing": missing,
        "backends": backends,
        "coverage": round(len(covered) / total, 4) if total else 1,
    }


def delete_project_files(project: Project) -> None:
    shutil.rmtree(project.directory, ignore_errors=True)
