"""Audio I/O and lightweight DSP.

Everything here goes through ffmpeg rather than a Python decoding library, so
the model server accepts exactly the same container and codec zoo as the render
pipeline does and there is only one thing to install.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


class AudioError(RuntimeError):
    """Raised when ffmpeg cannot decode the supplied media."""


def decode(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Decode any media file to a mono float32 waveform at ``sample_rate``."""
    cmd = [
        FFMPEG, "-v", "error", "-nostdin",
        "-i", path,
        "-map", "a:0",          # first audio stream; video is ignored
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise AudioError(proc.stderr.decode("utf-8", "replace").strip()[:500])
    if not proc.stdout:
        raise AudioError("media contains no decodable audio stream")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 waveform as a 16-bit PCM WAV container."""
    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    cmd = [
        FFMPEG, "-v", "error", "-nostdin",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "-",
        "-f", "wav", "-acodec", "pcm_s16le", "-",
    ]
    proc = subprocess.run(cmd, input=samples.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise AudioError(proc.stderr.decode("utf-8", "replace").strip()[:500])
    return proc.stdout


def probe_duration(path: str) -> float:
    """Container duration in seconds, or 0.0 when ffprobe cannot tell."""
    cmd = [
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    try:
        return float(proc.stdout.decode().strip())
    except (ValueError, AttributeError):
        return 0.0


def rms_envelope(samples: np.ndarray, sample_rate: int, fps: int = 100) -> np.ndarray:
    """Short-time RMS at ``fps`` frames per second.

    Doubles as the waveform the editor draws and as the signal the EDL builder
    uses to nudge cut points into silence.
    """
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    hop = max(1, int(round(sample_rate / fps)))
    frames = int(np.ceil(samples.size / hop))
    padded = np.pad(samples, (0, frames * hop - samples.size))
    blocks = padded.reshape(frames, hop)
    return np.sqrt(np.mean(np.square(blocks, dtype=np.float64), axis=1)).astype(np.float32)


@dataclass
class VoiceStats:
    """Coarse description of a speaker, used to match synthesised speech."""

    median_f0: float          # Hz, 0.0 when no voiced frames were found
    speech_rms: float         # linear RMS of the voiced parts
    peak: float               # absolute peak, for headroom-safe mixing
    sample_rate: int

    def to_json(self) -> dict:
        return {
            "median_f0": round(self.median_f0, 2),
            "speech_rms": round(float(self.speech_rms), 6),
            "peak": round(float(self.peak), 6),
            "sample_rate": self.sample_rate,
        }


def estimate_f0(samples: np.ndarray, sample_rate: int,
                fmin: float = 70.0, fmax: float = 350.0) -> float:
    """Median fundamental frequency over voiced frames.

    Plain autocorrelation over 40 ms frames. This is not a pitch tracker good
    enough to resynthesise with, but it is entirely good enough to stop a
    fallback voice from coming out an octave away from the speaker.
    """
    if samples.size < sample_rate // 10:
        return 0.0

    frame = int(0.04 * sample_rate)
    hop = int(0.02 * sample_rate)
    min_lag = max(2, int(sample_rate / fmax))
    max_lag = min(frame - 1, int(sample_rate / fmin))
    if max_lag <= min_lag:
        return 0.0

    # Only bother with frames that carry real energy.
    threshold = max(1e-4, float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) * 0.5)

    pitches: list[float] = []
    for start in range(0, samples.size - frame, hop):
        block = samples[start:start + frame].astype(np.float64)
        if np.sqrt(np.mean(block ** 2)) < threshold:
            continue
        block = block - block.mean()
        norm = np.dot(block, block)
        if norm <= 1e-12:
            continue
        corr = np.correlate(block, block, mode="full")[frame - 1:]
        window = corr[min_lag:max_lag]
        if window.size == 0:
            continue
        lag = int(np.argmax(window)) + min_lag
        # Reject frames with no convincing periodicity.
        if corr[lag] / norm < 0.3:
            continue
        pitches.append(sample_rate / lag)
        if len(pitches) >= 400:  # plenty for a median; keep long files fast
            break

    if not pitches:
        return 0.0
    return float(np.median(pitches))


def voice_stats(samples: np.ndarray, sample_rate: int) -> VoiceStats:
    """Summarise a speaker from their recording."""
    if samples.size == 0:
        return VoiceStats(0.0, 0.0, 0.0, sample_rate)
    envelope = rms_envelope(samples, sample_rate, fps=100)
    # "Speech" is everything above a fraction of the loudest frame, which keeps
    # room tone out of the level measurement.
    if envelope.size:
        active = envelope[envelope > max(envelope.max() * 0.15, 1e-5)]
    else:
        active = envelope
    speech_rms = float(active.mean()) if active.size else 0.0
    return VoiceStats(
        median_f0=estimate_f0(samples, sample_rate),
        speech_rms=speech_rms,
        peak=float(np.abs(samples).max()),
        sample_rate=sample_rate,
    )


def match_loudness(samples: np.ndarray, target_rms: float, max_gain: float = 8.0) -> np.ndarray:
    """Scale ``samples`` so their speech level matches ``target_rms``."""
    if samples.size == 0 or target_rms <= 0:
        return samples
    envelope = rms_envelope(samples, 16000, fps=100)
    if envelope.size == 0:
        return samples
    active = envelope[envelope > max(envelope.max() * 0.15, 1e-5)]
    current = float(active.mean()) if active.size else 0.0
    if current <= 1e-6:
        return samples
    gain = min(target_rms / current, max_gain)
    out = samples * gain
    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > 0.99:  # never clip
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def pitch_shift(samples: np.ndarray, sample_rate: int, ratio: float) -> np.ndarray:
    """Shift pitch by ``ratio`` while preserving duration.

    Implemented as resample-then-time-stretch using only stock ffmpeg filters,
    so no extra native dependency is needed. ``atempo`` is chained in steps
    because it only accepts factors between 0.5 and 2.0.
    """
    if samples.size == 0 or abs(ratio - 1.0) < 0.01:
        return samples
    ratio = float(np.clip(ratio, 0.5, 2.0))

    tempo = 1.0 / ratio
    stages: list[str] = []
    remaining = tempo
    while remaining < 0.5:
        stages.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        stages.append("atempo=2.0")
        remaining /= 2.0
    stages.append(f"atempo={remaining:.6f}")

    chain = ",".join([f"asetrate={int(round(sample_rate * ratio))}", *stages,
                      f"aresample={sample_rate}"])
    cmd = [
        FFMPEG, "-v", "error", "-nostdin",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "-",
        "-af", chain,
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-",
    ]
    proc = subprocess.run(cmd, input=samples.astype(np.float32).tobytes(), capture_output=True)
    if proc.returncode != 0:
        return samples  # a failed cosmetic shift must not fail the render
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample through ffmpeg's soxr, which beats anything hand-rolled here."""
    if src_rate == dst_rate or samples.size == 0:
        return samples
    cmd = [
        FFMPEG, "-v", "error", "-nostdin",
        "-f", "f32le", "-ar", str(src_rate), "-ac", "1", "-i", "-",
        "-af", f"aresample={dst_rate}:resampler=soxr",
        "-f", "f32le", "-ar", str(dst_rate), "-ac", "1", "-",
    ]
    proc = subprocess.run(cmd, input=samples.astype(np.float32).tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise AudioError(proc.stderr.decode("utf-8", "replace").strip()[:500])
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)
