"""Thin wrapper around the ffmpeg and ffprobe binaries.

Arguments are always passed as a list — never interpolated into a shell string —
so a filename containing a quote or a semicolon is data, not code.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from django.conf import settings

log = logging.getLogger(__name__)


class FfmpegError(RuntimeError):
    """ffmpeg could not do what was asked."""

    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


def _config(key: str):
    return settings.VOXDOCS[key]


def run_ffmpeg(args: list[str], filter_script: str | None = None,
               timeout: float | None = None) -> str:
    """Run ffmpeg.

    A ``filter_script`` is written to a file and passed with
    ``-filter_complex_script`` rather than on the command line: a heavily edited
    transcript produces a filter graph far longer than the OS argument limit.
    """
    script_path = None
    try:
        final_args = list(args)
        if filter_script:
            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False, encoding="utf-8"
            )
            handle.write(filter_script)
            handle.close()
            script_path = handle.name
            # Filter options must precede the output file, which callers put last.
            final_args = [*args[:-1], "-filter_complex_script", script_path, args[-1]]

        completed = subprocess.run(
            [_config("FFMPEG"), "-nostdin", "-v", "error", *final_args],
            capture_output=True,
            timeout=timeout,
        )
        stderr = completed.stderr.decode("utf-8", "replace")
        if completed.returncode != 0:
            detail = " ".join(stderr.strip().splitlines()[-3:])
            raise FfmpegError(f"ffmpeg failed: {detail or 'unknown error'}", stderr)
        return stderr
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError("ffmpeg timed out") from exc
    finally:
        if script_path:
            Path(script_path).unlink(missing_ok=True)


def probe(path: str | Path) -> dict:
    """Inspect a media file."""
    args = [
        _config("FFPROBE"), "-v", "error",
        "-show_entries", "format=duration,format_name",
        "-show_entries", "stream=codec_type,width,height,avg_frame_rate",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(args, capture_output=True)
    if completed.returncode != 0:
        raise FfmpegError(
            "could not read media file", completed.stderr.decode("utf-8", "replace")
        )

    parsed = json.loads(completed.stdout or b"{}")
    streams = parsed.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    fps = 0.0
    rate = (video or {}).get("avg_frame_rate", "0/0")
    if rate and rate != "0/0":
        numerator, _, denominator = rate.partition("/")
        try:
            if float(denominator):
                fps = float(numerator) / float(denominator)
        except ValueError:
            fps = 0.0

    try:
        duration = float(parsed.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0

    return {
        "duration": duration,
        "format": parsed.get("format", {}).get("format_name", ""),
        "has_video": video is not None,
        "has_audio": audio is not None,
        "width": int(video.get("width") or 0) if video else 0,
        "height": int(video.get("height") or 0) if video else 0,
        "fps": fps,
    }


def duration_of(path: str | Path) -> float:
    """Duration in seconds, or 0.0 when unknown."""
    try:
        return probe(path)["duration"]
    except (FfmpegError, ValueError):
        return 0.0


def normalize_to_master(source: str | Path, output: str | Path) -> Path:
    """Decode any input to the canonical render format.

    Working in one format throughout means concatenation is sample-exact and no
    hidden resampling creeps in between neighbouring segments.
    """
    run_ffmpeg([
        "-y", "-i", str(source),
        "-map", "a:0",
        "-ac", "1",
        "-ar", str(_config("RENDER_SAMPLE_RATE")),
        "-c:a", "pcm_f32le",
        str(output),
    ])
    return Path(output)


def make_preview_audio(source: str | Path, output: str | Path) -> Path:
    """A small AAC copy for the browser to stream while editing."""
    run_ffmpeg([
        "-y", "-i", str(source),
        "-map", "a:0", "-ac", "1", "-ar", "44100",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(output),
    ])
    return Path(output)
