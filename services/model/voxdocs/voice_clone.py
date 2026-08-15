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

    def synthesize(self, text: str, target_language: str, speaker_wav_path: str) -> np.ndarray:
        """Returns float32 samples at `self.sample_rate` for `text`, spoken
        in the reference speaker's voice, in `target_language`.
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
