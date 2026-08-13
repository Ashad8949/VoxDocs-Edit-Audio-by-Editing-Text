"""DSP tests. These exercise real ffmpeg round trips, not mocks."""

import numpy as np
import pytest

from voxdocs import audio as A


def tone(freq: float, seconds: float, rate: int = 16000, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_wav_round_trip_preserves_the_waveform(tmp_path):
    original = tone(220.0, 0.5)
    wav = A.encode_wav(original, 16000)
    assert wav[:4] == b"RIFF"

    path = tmp_path / "t.wav"
    path.write_bytes(wav)
    decoded = A.decode(str(path), 16000)

    assert abs(decoded.size - original.size) <= 2
    n = min(decoded.size, original.size)
    # 16-bit quantisation is the only loss we accept here.
    assert np.max(np.abs(decoded[:n] - original[:n])) < 1e-3


def test_decode_rejects_files_with_no_audio(tmp_path):
    path = tmp_path / "junk.wav"
    path.write_bytes(b"not actually audio at all")
    with pytest.raises(A.AudioError):
        A.decode(str(path), 16000)


def test_probe_duration_reads_the_container(tmp_path):
    path = tmp_path / "t.wav"
    path.write_bytes(A.encode_wav(tone(200.0, 1.25), 16000))
    assert A.probe_duration(str(path)) == pytest.approx(1.25, abs=0.02)


def test_probe_duration_is_zero_for_unreadable_files(tmp_path):
    path = tmp_path / "nope.wav"
    path.write_bytes(b"garbage")
    assert A.probe_duration(str(path)) == 0.0


def test_rms_envelope_has_one_frame_per_hop_and_tracks_level():
    rate = 16000
    loud = tone(300.0, 1.0, rate, amp=0.8)
    quiet = tone(300.0, 1.0, rate, amp=0.05)
    signal = np.concatenate([loud, quiet])

    envelope = A.rms_envelope(signal, rate, fps=100)
    assert envelope.size == pytest.approx(200, abs=1)
    assert envelope[:90].mean() > envelope[110:].mean() * 5


def test_rms_envelope_handles_empty_input():
    assert A.rms_envelope(np.zeros(0, dtype=np.float32), 16000).size == 0


def test_estimate_f0_recovers_a_known_pitch():
    # A sawtooth is periodic and richly harmonic, like voiced speech.
    rate = 16000
    t = np.arange(rate) / rate
    saw = (2 * (t * 150.0 % 1.0) - 1.0).astype(np.float32) * 0.4
    assert A.estimate_f0(saw, rate) == pytest.approx(150.0, rel=0.06)


def test_estimate_f0_returns_zero_for_silence_and_short_input():
    assert A.estimate_f0(np.zeros(16000, dtype=np.float32), 16000) == 0.0
    assert A.estimate_f0(np.zeros(100, dtype=np.float32), 16000) == 0.0


def test_match_loudness_moves_level_toward_the_target_without_clipping():
    quiet = tone(300.0, 1.0, amp=0.02)
    matched = A.match_loudness(quiet, target_rms=0.12)
    assert np.abs(matched).max() <= 0.99
    envelope = A.rms_envelope(matched, 16000)
    active = envelope[envelope > envelope.max() * 0.15]
    assert active.mean() == pytest.approx(0.12, rel=0.25)


def test_match_loudness_is_a_no_op_without_a_target():
    signal = tone(300.0, 0.2)
    assert np.array_equal(A.match_loudness(signal, 0.0), signal)


def test_pitch_shift_changes_pitch_and_keeps_duration():
    rate = 16000
    t = np.arange(rate) / rate
    saw = (2 * (t * 120.0 % 1.0) - 1.0).astype(np.float32) * 0.4

    shifted = A.pitch_shift(saw, rate, 1.5)
    assert shifted.size == pytest.approx(saw.size, rel=0.08)
    assert A.estimate_f0(shifted, rate) == pytest.approx(180.0, rel=0.12)


def test_pitch_shift_is_a_no_op_for_unity_ratio():
    signal = tone(200.0, 0.2)
    assert np.array_equal(A.pitch_shift(signal, 16000, 1.0), signal)


def test_resample_changes_rate_and_preserves_duration():
    signal = tone(300.0, 1.0, 16000)
    out = A.resample(signal, 16000, 48000)
    assert out.size == pytest.approx(48000, rel=0.02)
    assert A.estimate_f0(out, 48000) == pytest.approx(300.0, rel=0.08)


def test_voice_stats_summarises_a_speaker():
    rate = 16000
    t = np.arange(2 * rate) / rate
    saw = (2 * (t * 130.0 % 1.0) - 1.0).astype(np.float32) * 0.5
    stats = A.voice_stats(saw, rate)
    assert stats.median_f0 == pytest.approx(130.0, rel=0.08)
    assert stats.speech_rms > 0
    assert stats.peak <= 1.0
    assert stats.sample_rate == rate
    assert "median_f0" in stats.to_json()


def test_voice_stats_on_empty_input_is_all_zeros():
    stats = A.voice_stats(np.zeros(0, dtype=np.float32), 16000)
    assert stats.median_f0 == 0.0 and stats.peak == 0.0
