"""Render tests that assert on samples, not just on return codes.

A render pipeline can produce a file of exactly the right length that contains
entirely the wrong audio, so these build a master of distinct tones and check by
frequency which slices actually survived.
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest

from edl import CopySegment, SynthSegment
from projects.services.ffmpeg import FfmpegError, duration_of, probe
from projects.services.render import (
    RenderOp,
    encode_audio,
    expand_segments,
    op_durations,
    render_ops,
    render_video,
)

#: Second 0 = 200 Hz, second 1 = 400 Hz, second 2 = 800 Hz, second 3 = 1600 Hz.
TONES = [200, 400, 800, 1600]


@pytest.fixture(scope="module")
def master(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("render") / "master.wav"
    filters = ";".join(
        f"sine=frequency={hz}:duration=1:sample_rate=48000[t{i}]"
        for i, hz in enumerate(TONES)
    )
    labels = "".join(f"[t{i}]" for i in range(len(TONES)))
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-filter_complex", f"{filters};{labels}concat=n={len(TONES)}:v=0:a=1[out]",
        "-map", "[out]", "-ac", "1", "-ar", "48000", "-c:a", "pcm_f32le", str(path),
    ], check=True)
    return path


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    """Four seconds of 25 fps video with sound."""
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4:sample_rate=48000",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True)
    return path


def read_samples(path: Path, start: float, duration: float) -> list[float]:
    """Decode a slice to raw float32 samples."""
    completed = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-af", f"atrim=start={start}:duration={duration}",
        "-f", "f32le", "-ac", "1", "-ar", "48000", "-",
    ], capture_output=True, check=True)
    raw = completed.stdout
    return list(struct.unpack(f"<{len(raw) // 4}f", raw[: len(raw) // 4 * 4]))


def dominant_frequency(path: Path, start: float, duration: float) -> float:
    """Estimate frequency by counting zero crossings — enough to tell tones apart."""
    samples = read_samples(path, start, duration)
    if len(samples) < 2:
        return 0.0
    crossings = sum(
        1 for i in range(1, len(samples))
        if (samples[i - 1] < 0) != (samples[i] < 0)
    )
    return crossings * 48000 / len(samples) / 2


def stream_duration(path: Path, kind: str) -> float:
    completed = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v" if kind == "video" else "a",
        "-show_entries", "stream=duration", "-of", "csv=p=0", str(path),
    ], capture_output=True, check=True)
    return float(completed.stdout.decode().strip().splitlines()[0])


# ------------------------------------------------------------------ audio

def test_a_single_copy_reproduces_that_slice(master, tmp_path):
    out = tmp_path / "copy.wav"
    result = render_ops(master, [RenderOp(kind="source", start=2.0, end=3.0)], out)

    assert result["duration"] == pytest.approx(1.0, abs=0.02)
    assert result["pieces"] == 1
    assert dominant_frequency(out, 0.1, 0.6) == pytest.approx(800, abs=60)


def test_deleting_the_middle_joins_the_outer_pieces(master, tmp_path):
    out = tmp_path / "cut.wav"
    result = render_ops(master, [
        RenderOp(kind="source", start=0.0, end=1.0),
        RenderOp(kind="source", start=3.0, end=4.0),
    ], out)

    assert result["duration"] == pytest.approx(2.0, abs=0.03)
    assert result["pieces"] == 2
    assert dominant_frequency(out, 0.2, 0.5) == pytest.approx(200, abs=40)
    assert dominant_frequency(out, 1.2, 0.5) == pytest.approx(1600, abs=120)


def test_silence_contributes_its_exact_duration(master, tmp_path):
    out = tmp_path / "silence.wav"
    result = render_ops(master, [
        RenderOp(kind="source", start=0.0, end=0.5),
        RenderOp(kind="silence", duration=0.25),
        RenderOp(kind="source", start=0.0, end=0.5),
    ], out)
    assert result["duration"] == pytest.approx(1.25, abs=0.03)


def test_an_empty_edit_renders_a_valid_file(master, tmp_path):
    out = tmp_path / "empty.wav"
    result = render_ops(master, [], out)
    assert result["pieces"] == 0
    info = probe(out)
    assert info["has_audio"] and info["duration"] < 0.2


def test_seam_fades_keep_the_join_continuous(master, tmp_path):
    # Concatenating the tail of one tone onto the head of another steps the
    # waveform; the fade must pull the samples through zero at the join.
    out = tmp_path / "seam.wav"
    render_ops(master, [
        RenderOp(kind="source", start=0.4, end=0.9),
        RenderOp(kind="source", start=3.1, end=3.6),
    ], out, fade=0.01)

    samples = read_samples(out, 0.495, 0.01)
    assert len(samples) > 100
    max_step = max(abs(samples[i] - samples[i - 1]) for i in range(1, len(samples)))
    # A 1600 Hz sine at 48 kHz steps ~0.21 per sample at its steepest; a hard
    # splice discontinuity would be far larger.
    assert max_step < 0.35, f"discontinuity at the seam: {max_step}"


def test_batched_rendering_matches_a_single_pass(master, tmp_path, settings):
    ops = [
        RenderOp(kind="source", start=(i % 4) * 1.0, end=(i % 4) * 1.0 + 0.25)
        for i in range(12)
    ]
    single = tmp_path / "single.wav"
    render_ops(master, ops, single)

    settings.VOXDOCS = {**settings.VOXDOCS, "MAX_SEGMENTS_PER_PASS": 5}
    batched = tmp_path / "batched.wav"
    result = render_ops(master, ops, batched)

    assert result["pieces"] == 12
    assert duration_of(batched) == pytest.approx(duration_of(single), abs=0.05)
    assert duration_of(batched) == pytest.approx(3.0, abs=0.1)


def test_op_durations_avoid_probing_known_lengths():
    assert op_durations([
        RenderOp(kind="source", start=1.0, end=2.5),
        RenderOp(kind="silence", duration=0.2),
    ]) == [1.5, 0.2]


@pytest.mark.parametrize("fmt,container", [("wav", "wav"), ("mp3", "mp3"), ("m4a", "mp4")])
def test_encode_audio_produces_each_format(master, tmp_path, fmt, container):
    source = tmp_path / "src.wav"
    render_ops(master, [RenderOp(kind="source", start=0.0, end=1.0)], source)

    out = tmp_path / f"enc.{fmt}"
    encode_audio(source, out, fmt)
    info = probe(out)
    assert info["has_audio"]
    assert container in info["format"]


def test_encode_audio_rejects_an_unknown_format(master, tmp_path):
    with pytest.raises(FfmpegError, match="unsupported audio format"):
        encode_audio(master, tmp_path / "x.ogg", "ogg")


# -------------------------------------------------------- segment expansion

def test_zero_length_ops_are_dropped_before_ffmpeg(tmp_path):
    ops, _ = expand_segments(
        [
            CopySegment(1.0, 1.0, ["a"], 0, 0),
            CopySegment(2.0, 3.0, ["b"], 1, 1),
        ],
        {}, tmp_path / "inline",
    )
    assert len(ops) == 1
    assert ops[0].start == 2.0


def test_voice_bank_units_become_ordinary_source_ops(tmp_path):
    segments = [
        CopySegment(0, 1, ["w0"], 0, 0),
        SynthSegment("hello", None, None, 0, 0, 0.3),
    ]
    synthesis = {1: {"units": [{"type": "source", "start": 2.0, "end": 2.4,
                                "gain": 1, "word": "hello"}], "missing": []}}
    ops, warnings = expand_segments(segments, synthesis, tmp_path / "inline")

    assert len(ops) == 2
    assert ops[1].kind == "source", "a bank hit is lifted from the master"
    assert ops[1].start == 2.0
    assert warnings == []


def test_inline_tts_audio_is_materialised(master, tmp_path):
    import base64

    wav = tmp_path / "tone.wav"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
        "-i", "sine=frequency=440:duration=0.3:sample_rate=48000",
        "-ac", "1", "-c:a", "pcm_s16le", str(wav),
    ], check=True)
    data = base64.b64encode(wav.read_bytes()).decode()

    segments = [SynthSegment("zebra", None, None, 0, 0, 0.3)]
    synthesis = {0: {"units": [{"type": "audio", "data": data,
                                "sample_rate": 48000, "word": "zebra"}], "missing": []}}
    ops, _ = expand_segments(segments, synthesis, tmp_path / "inline")

    assert len(ops) == 1 and ops[0].kind == "file"
    assert ops[0].file.stat().st_size > 0

    out = tmp_path / "inline.wav"
    assert render_ops(master, ops, out)["duration"] == pytest.approx(0.3, abs=0.05)


def test_unsynthesisable_words_are_warned_not_swallowed(tmp_path):
    segments = [SynthSegment("zebra", None, None, 0, 0, 0.3)]
    ops, warnings = expand_segments(segments, {0: {"units": [], "missing": ["zebra"]}},
                                    tmp_path / "inline")
    assert ops == []
    assert len(warnings) == 1 and "zebra" in warnings[0]


def test_a_missing_synthesis_result_warns(tmp_path):
    segments = [SynthSegment("ghost", None, None, 0, 0, 0.3)]
    ops, warnings = expand_segments(segments, {}, tmp_path / "inline")
    assert ops == []
    assert "ghost" in warnings[0]


# ------------------------------------------------------------------ video

def test_video_cuts_follow_the_same_edl(master, clip, tmp_path):
    audio = tmp_path / "cut.wav"
    render_ops(master, [
        RenderOp(kind="source", start=0.0, end=1.0),
        RenderOp(kind="source", start=3.0, end=4.0),
    ], audio)

    out = tmp_path / "cut.mp4"
    render_video(clip, audio, [CopySegment(0.0, 1.0, ["w0"], 0, 0),
                               CopySegment(3.0, 4.0, ["w3"], 3, 3)], {}, out)

    info = probe(out)
    assert info["has_video"] and info["has_audio"]
    assert info["duration"] == pytest.approx(2.0, abs=0.1)


def test_a_leading_insertion_pre_rolls_a_freeze_frame(master, clip, tmp_path):
    # Regression: a zero-length leading shot yielded no frames at all, so tpad
    # had nothing to clone and the picture came out shorter than the sound.
    audio = tmp_path / "lead.wav"
    render_ops(master, [
        RenderOp(kind="silence", duration=0.5),
        RenderOp(kind="source", start=2.0, end=4.0),
    ], audio)

    out = tmp_path / "lead.mp4"
    render_video(
        clip, audio,
        [SynthSegment("intro", None, None, 0, 0, 0.5), CopySegment(2.0, 4.0, ["w2"], 2, 3)],
        {0: 0.5}, out,
    )

    video_seconds = stream_duration(out, "video")
    audio_seconds = stream_duration(out, "audio")
    assert video_seconds > 2.3, f"the freeze frame is missing: video is {video_seconds}s"
    # Within one frame at 25 fps; no audio may be truncated to match the picture.
    assert abs(video_seconds - audio_seconds) <= 0.045


def test_a_mid_insertion_freezes_the_preceding_shot(master, clip, tmp_path):
    audio = tmp_path / "mid.wav"
    render_ops(master, [
        RenderOp(kind="source", start=0.0, end=1.0),
        RenderOp(kind="silence", duration=0.4),
        RenderOp(kind="source", start=3.0, end=4.0),
    ], audio)

    out = tmp_path / "mid.mp4"
    render_video(
        clip, audio,
        [
            CopySegment(0.0, 1.0, ["w0"], 0, 0),
            SynthSegment("middle", "a", "b", 0, 0, 0.4),
            CopySegment(3.0, 4.0, ["w3"], 3, 3),
        ],
        {1: 0.4}, out,
    )
    assert stream_duration(out, "video") == pytest.approx(2.4, abs=0.08)


def test_an_edit_that_removes_every_frame_is_refused(clip, tmp_path):
    with pytest.raises(FfmpegError, match="removed every frame"):
        render_video(clip, clip, [], {}, tmp_path / "none.mp4")
