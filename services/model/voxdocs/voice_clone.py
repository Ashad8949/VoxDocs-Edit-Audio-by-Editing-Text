"""XTTS-v2-backed voice-preserving speech synthesis.

Given a short reference clip of the original speaker, synthesizes new text in
a chosen language while keeping that speaker's voice — this is what lets an
edited or translated segment sound like the same person who spoke the rest of
the video, even across languages (Hindi source, English-edited segment, etc).

XTTS-v2 ships under Coqui's CPML license (free for research/personal use, a
commercial license is required to monetize a product built on it) — hence the
explicit COQUI_TOS_AGREED opt-in rather than auto-accepting it in code.

Model load is expensive and not thread-safe, so it happens once, lazily, the
same way asr.py loads Whisper and translate.py loads IndicTrans2.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

log = logging.getLogger(__name__)

# XTTS-v2 has no separate Hinglish acoustic model — "hinglish" is spoken as
# Hindi; the code-mixed *wording* comes from translate.py's loanword pass,
# not from the voice.
_LANGUAGE_MAP = {"en": "en", "hi": "hi", "hinglish": "hi"}

SAMPLE_RATE = 24000
# XTTS needs a reference clip, not the whole source track: a few seconds is
# enough to condition on the speaker, and capping it keeps load fast even for
# a long video.
MAX_REFERENCE_SECONDS = 30.0


def _register_ffmpeg_dll_dir() -> None:
    """Windows dev-machine convenience only. torchcodec (XTTS's audio
    backend) needs its FFmpeg shared-DLL directory registered via
    os.add_dll_directory() because plain PATH isn't searched for a loaded
    DLL's own dependencies since Python 3.8 on Windows. Linux containers
    don't need this: apt's ffmpeg registers its .so files with ldconfig.
    """
    dll_dir = os.environ.get("VOXDOCS_FFMPEG_DLL_DIR")
    if dll_dir and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(dll_dir)


class XttsVoiceCloner:
    """Lazily-loaded cross-lingual voice cloning via Coqui XTTS-v2."""

    name = "xtts-v2"
    sample_rate = SAMPLE_RATE

    def __init__(self, device: str | None = None) -> None:
        self.device = device or os.environ.get("VOXDOCS_TTS_DEVICE", "cpu")
        self._tts = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        if os.environ.get("COQUI_TOS_AGREED") not in ("1", "true", "yes"):
            return False
        try:
            import TTS  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self):
        if self._tts is not None:
            return self._tts
        with self._lock:
            if self._tts is not None:
                return self._tts
            _register_ffmpeg_dll_dir()
            from TTS.api import TTS

            log.info("loading XTTS-v2 on %s", self.device)
            self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            return self._tts

    def synthesize(self, text: str, target_language: str,
                   speaker_wav_path: str | list[str]) -> np.ndarray:
        """Returns float32 samples at `self.sample_rate` for `text`, spoken
        in the reference speaker's voice, in `target_language`.

        `speaker_wav_path` may be a single clip or a list of clips — XTTS
        averages the speaker conditioning across all of them.
        """
        tts = self._load()
        lang = _LANGUAGE_MAP.get(target_language, "en")
        wav = tts.tts(text=text, speaker_wav=speaker_wav_path, language=lang)
        return np.asarray(wav, dtype=np.float32)


def trim_reference(samples: np.ndarray, sample_rate: int,
                    max_seconds: float = MAX_REFERENCE_SECONDS) -> np.ndarray:
    """Cap a project's cached audio to a short reference clip for XTTS."""
    max_samples = int(max_seconds * sample_rate)
    if samples.shape[0] <= max_samples:
        return samples
    return samples[:max_samples]


# A reference clip shorter than this is too little context to condition on
# well; longer than this and one clip crowds out variety from elsewhere.
MIN_CLIP_SECONDS = 2.0
MAX_CLIP_SECONDS = 12.0
# Words further apart than this start a new clip — a pause that long usually
# means a sentence boundary, and splicing across it into one clip sounds odd.
CLIP_GAP_SECONDS = 0.6
# Below this the transcriber wasn't sure what it heard; such spans tend to be
# noise, crosstalk or mumbling — poor material to clone a voice from.
MIN_CONFIDENCE = 0.5


def select_reference_clips(
    samples: np.ndarray,
    sample_rate: int,
    words: list[tuple[float, float, float]],
    max_seconds: float = MAX_REFERENCE_SECONDS,
    max_clips: int = 3,
) -> list[np.ndarray]:
    """Pick a few clean, confident, contiguous speech spans to clone from.

    XTTS conditions on a reference clip; the closer that clip is to clear,
    representative speech, the better the clone. The old "first 30 seconds"
    could easily land on silence, an intro sting or a throat-clear. Here we
    use the transcript's own per-word confidence and timings (``words`` is a
    list of ``(start, end, confidence)``) to find the best material, and
    return several spans since XTTS accepts multiple reference clips.

    Falls back to a plain head trim when there is no usable word timing.
    """
    if samples.size == 0:
        return []
    if not words:
        return [trim_reference(samples, sample_rate, max_seconds)]

    # Group consecutive words into runs, breaking on a long pause or a word
    # the transcriber flagged as low-confidence.
    runs: list[list[tuple[float, float, float]]] = []
    current: list[tuple[float, float, float]] = []
    prev_end = None
    for start, end, conf in words:
        if conf < MIN_CONFIDENCE or (prev_end is not None and start - prev_end > CLIP_GAP_SECONDS):
            if current:
                runs.append(current)
            current = []
        if conf >= MIN_CONFIDENCE:
            current.append((start, end, conf))
        prev_end = end
    if current:
        runs.append(current)

    # Score each run: prefer high confidence, then longer spans. Cap each run
    # to MAX_CLIP_SECONDS so one monologue can't monopolise the budget.
    spans: list[tuple[float, float, float]] = []  # (start, end, mean_conf)
    for run in runs:
        start = run[0][0]
        end = min(run[-1][1], start + MAX_CLIP_SECONDS)
        if end - start < MIN_CLIP_SECONDS:
            continue
        mean_conf = float(np.mean([c for _, _, c in run]))
        spans.append((start, end, mean_conf))

    if not spans:
        return [trim_reference(samples, sample_rate, max_seconds)]

    spans.sort(key=lambda s: (s[2], s[1] - s[0]), reverse=True)

    clips: list[np.ndarray] = []
    total = 0.0
    for start, end, _ in spans:
        if len(clips) >= max_clips or total >= max_seconds:
            break
        end = min(end, start + (max_seconds - total))
        if end - start < MIN_CLIP_SECONDS:
            continue
        lo = max(0, int(start * sample_rate))
        hi = min(samples.shape[0], int(end * sample_rate))
        if hi - lo > 0:
            clips.append(samples[lo:hi])
            total += (hi - lo) / sample_rate

    return clips or [trim_reference(samples, sample_rate, max_seconds)]
