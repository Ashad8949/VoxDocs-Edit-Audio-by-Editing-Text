"""EDL → audio (and video) rendering.

The EDL says *what* the output contains; this module makes it exist. The whole
render is expressed as one ffmpeg filter graph so the audio is decoded once and
never round-trips through an intermediate file, which matters when a heavily
edited hour of speech becomes several hundred pieces.

Two details carry most of the audible quality:

  - **Seam fades.** Butt-joining two unrelated pieces of a waveform steps the
    signal discontinuously and clicks. A few milliseconds of fade on each side
    of every seam removes it, below the threshold of audibility.
  - **One canonical format.** Everything is decoded to mono float32 at a single
    sample rate before it is cut, so no segment is silently resampled relative
    to its neighbour.
"""

from __future__ import annotations

import base64
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings

from edl import CopySegment, SynthSegment

from .ffmpeg import FfmpegError, duration_of, run_ffmpeg

log = logging.getLogger(__name__)


@dataclass
class RenderOp:
    """One piece of the output: source audio, an inline file, or silence."""

    kind: str                  # "source" | "file" | "silence"
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    gain: float = 1.0
    file: Path | None = None
    label: str = ""

    @property
    def audible(self) -> bool:
        """Zero-length pieces would make ffmpeg's atrim produce nothing."""
        if self.kind == "source":
            return self.end - self.start > 0.001
        if self.kind == "silence":
            return self.duration > 0.001
        return True


def _config(key: str):
    return settings.VOXDOCS[key]


def expand_segments(segments, synthesis: dict[int, dict],
                    inline_dir: Path) -> tuple[list[RenderOp], list[str]]:
    """Flatten EDL segments plus resolved synthesis units into render ops.

    Synthesis comes back from the model server as a *plan* rather than as audio
    whenever it can: units referencing the speaker's own recording become
    ordinary source ops here, so spliced words are lifted from the same
    full-quality master as every other segment instead of from a resampled copy
    shipped over HTTP.
    """
    ops: list[RenderOp] = []
    warnings: list[str] = []
    inline_dir.mkdir(parents=True, exist_ok=True)

    inline_index = 0
    for i, segment in enumerate(segments):
        if isinstance(segment, CopySegment):
            ops.append(RenderOp(
                kind="source", start=segment.start, end=segment.end,
                label=f"copy:{segment.word_ids[0] if segment.word_ids else ''}",
            ))
            continue

        resolved = synthesis.get(i)
        if resolved is None:
            warnings.append(
                f'no synthesis result for "{segment.text}"; the text was dropped'
            )
            continue
        if resolved.get("missing"):
            missing = ", ".join(f'"{w}"' for w in resolved["missing"])
            warnings.append(
                f"could not synthesise {missing} — no voice-bank match and no "
                "TTS backend available"
            )

        for unit in resolved.get("units", []):
            kind = unit.get("type")
            if kind == "source":
                ops.append(RenderOp(
                    kind="source",
                    start=float(unit["start"]), end=float(unit["end"]),
                    gain=float(unit.get("gain", 1.0)),
                    label=f"synth:{unit.get('word', '')}",
                ))
            elif kind == "silence":
                duration = float(unit.get("duration", 0.0))
                if duration > 0:
                    ops.append(RenderOp(kind="silence", duration=duration, label="gap"))
            elif kind == "audio" and unit.get("data"):
                inline_index += 1
                path = inline_dir / f"tts-{i}-{inline_index}.wav"
                path.write_bytes(base64.b64decode(unit["data"]))
                ops.append(RenderOp(kind="file", file=path,
                                    label=f"tts:{unit.get('word', '')}"))

    return [op for op in ops if op.audible], warnings


def op_durations(ops: list[RenderOp]) -> list[float]:
    """Exact duration of every op; only inline files need probing."""
    cache: dict[Path, float] = {}
    durations = []
    for op in ops:
        if op.kind == "source":
            durations.append(op.end - op.start)
        elif op.kind == "silence":
            durations.append(op.duration)
        else:
            if op.file not in cache:
                cache[op.file] = duration_of(op.file)
            durations.append(cache[op.file])
    return durations


def _fade_chain(fade_in: float, fade_out: float, duration: float) -> str:
    parts = []
    if fade_in > 0.0005:
        parts.append(f"afade=t=in:st=0:d={fade_in:.6f}:curve=tri")
    if fade_out > 0.0005:
        start = max(0.0, duration - fade_out)
        parts.append(f"afade=t=out:st={start:.6f}:d={fade_out:.6f}:curve=tri")
    return ("," + ",".join(parts)) if parts else ""


def render_ops(master: Path, ops: list[RenderOp], output: Path,
               fade: float | None = None, sample_rate: int | None = None) -> dict:
    """Render a list of ops to a single audio file."""
    fade = _config("SEAM_FADE") if fade is None else fade
    rate = _config("RENDER_SAMPLE_RATE") if sample_rate is None else sample_rate

    if not ops:
        # A transcript edited down to nothing is a legitimate outcome.
        run_ffmpeg([
            "-y", "-f", "lavfi", "-i", f"anullsrc=r={rate}:cl=mono",
            "-t", "0.05", "-c:a", "pcm_f32le", str(output),
        ])
        return {"duration": 0.0, "pieces": 0}

    # Very large edits render in batches, then join. Batch boundaries fall
    # between ops that already carry their own seam fades, so the split is
    # acoustically invisible.
    if len(ops) > _config("MAX_SEGMENTS_PER_PASS"):
        return _render_in_batches(master, ops, output, fade, rate)

    durations = op_durations(ops)
    inputs = ["-i", str(master)]
    input_index_by_file: dict[Path, int] = {}
    for op in ops:
        if op.kind == "file" and op.file not in input_index_by_file:
            input_index_by_file[op.file] = len(inputs) // 2
            inputs += ["-i", str(op.file)]

    lines = []
    labels = []
    for i, op in enumerate(ops):
        label = f"s{i}"
        labels.append(f"[{label}]")
        duration = durations[i]

        # Only the outer edges of the whole render keep their natural envelope.
        fade_in = min(fade, duration / 2) if i > 0 else 0.0
        fade_out = min(fade, duration / 2) if i < len(ops) - 1 else 0.0
        fades = _fade_chain(fade_in, fade_out, duration)

        if op.kind == "source":
            filters = [f"atrim=start={op.start:.6f}:end={op.end:.6f}", "asetpts=N/SR/TB"]
            if abs(op.gain - 1.0) > 0.001:
                filters.append(f"volume={op.gain:.4f}")
            lines.append(f"[0:a]{','.join(filters)}{fades}[{label}];")
        elif op.kind == "file":
            index = input_index_by_file[op.file]
            lines.append(
                f"[{index}:a]aresample={rate},"
                f"aformat=sample_fmts=fltp:channel_layouts=mono,"
                f"asetpts=N/SR/TB{fades}[{label}];"
            )
        else:
            lines.append(
                f"aevalsrc=0:d={op.duration:.6f}:s={rate}:c=mono,"
                f"aformat=sample_fmts=fltp:channel_layouts=mono[{label}];"
            )

    lines.append(f"{''.join(labels)}concat=n={len(ops)}:v=0:a=1[out]")

    run_ffmpeg(
        [
            "-y", *inputs,
            "-map", "[out]",
            "-ac", "1", "-ar", str(rate),
            "-c:a", "pcm_f32le",
            str(output),
        ],
        filter_script="\n".join(lines),
    )
    return {"duration": duration_of(output), "pieces": len(ops)}


def _render_in_batches(master: Path, ops: list[RenderOp], output: Path,
                       fade: float, rate: int) -> dict:
    batch_size = _config("MAX_SEGMENTS_PER_PASS")
    parts: list[Path] = []
    workdir = Path(tempfile.mkdtemp(prefix="voxdocs-batch-"))
    try:
        for offset in range(0, len(ops), batch_size):
            part = workdir / f"batch-{offset}.wav"
            render_ops(master, ops[offset:offset + batch_size], part,
                       fade=fade, sample_rate=rate)
            parts.append(part)

        listing = workdir / "concat.txt"
        listing.write_text(
            "\n".join(f"file '{str(p).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"
                      for p in parts),
            encoding="utf-8",
        )
        run_ffmpeg([
            "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c:a", "pcm_f32le", "-ac", "1", "-ar", str(rate),
            str(output),
        ])
        return {"duration": duration_of(output), "pieces": len(ops)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


AUDIO_CODECS = {
    "wav": ["-c:a", "pcm_s16le"],
    "mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
    "m4a": ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"],
}


def encode_audio(source: Path, output: Path, fmt: str) -> Path:
    """Encode the rendered master into a delivery format."""
    codec = AUDIO_CODECS.get(fmt)
    if not codec:
        raise FfmpegError(f"unsupported audio format: {fmt}")
    run_ffmpeg(["-y", "-i", str(source), *codec, str(output)])
    return output


def render_video(source_video: Path, rendered_audio: Path, segments,
                 synth_durations: dict[int, float], output: Path) -> Path:
    """Render video alongside the audio.

    Copy segments trim the picture exactly as they trim the sound. Inserted
    speech has no picture to go with it, so the preceding shot is frozen for the
    length of the insertion. Freezing is the honest choice: it keeps sound and
    picture in sync and makes the edit visible, rather than pretending the
    speaker's lips match words they never said.
    """
    shots: list[dict] = []
    # An insertion before any picture has no shot to extend yet, so the hold is
    # carried forward as a pre-roll on the first real shot. Emitting a
    # zero-length shot instead would trim to an empty frame range — at 25 fps a
    # sub-frame window contains no frame at all — leaving tpad nothing to clone.
    pending_lead_hold = 0.0

    for i, segment in enumerate(segments):
        if isinstance(segment, CopySegment):
            shots.append({
                "start": segment.start, "end": segment.end,
                "pad_start": pending_lead_hold, "pad_end": 0.0,
            })
            pending_lead_hold = 0.0
            continue

        hold = synth_durations.get(i)
        if hold is None:
            hold = getattr(segment, "estimated_duration", 0.0)
        if hold <= 0:
            continue
        if shots:
            shots[-1]["pad_end"] += hold      # freeze the shot just played
        else:
            pending_lead_hold += hold

    if not shots:
        raise FfmpegError("the edit removed every frame of video")
    # Trailing insertions after the last copy segment freeze the final frame.
    if pending_lead_hold > 0:
        shots[-1]["pad_end"] += pending_lead_hold

    lines = []
    labels = []
    for i, shot in enumerate(shots):
        label = f"v{i}"
        labels.append(f"[{label}]")
        end = max(shot["end"], shot["start"] + 0.001)
        filters = [f"trim=start={shot['start']:.6f}:end={end:.6f}", "setpts=PTS-STARTPTS"]
        if shot["pad_start"] > 0:
            filters.append(f"tpad=start_mode=clone:start_duration={shot['pad_start']:.6f}")
        if shot["pad_end"] > 0:
            filters.append(f"tpad=stop_mode=clone:stop_duration={shot['pad_end']:.6f}")
        filters.append("setpts=PTS-STARTPTS")
        lines.append(f"[0:v]{','.join(filters)}[{label}];")
    lines.append(f"{''.join(labels)}concat=n={len(shots)}:v=1:a=0[outv]")

    run_ffmpeg(
        [
            "-y", "-i", str(source_video), "-i", str(rendered_audio),
            "-map", "[outv]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            # Deliberately no -shortest: the video track quantises to whole
            # frames and can land a fraction of a frame short of the audio.
            # Truncating to the shorter stream would clip the tail off the last
            # word; letting the final frame hold a few extra milliseconds is
            # inaudible and invisible.
            "-movflags", "+faststart",
            str(output),
        ],
        filter_script="\n".join(lines),
    )
    return output
