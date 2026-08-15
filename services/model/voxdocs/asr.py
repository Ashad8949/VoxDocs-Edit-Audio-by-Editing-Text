"""Speech recognition with word-level timestamps.

Word timings are not a nice-to-have here, they are the entire product: every
cut point in the rendered audio comes from a word boundary. Backends that only
return segment-level timings cannot drive the editor, so the abstraction below
demands per-word `start`/`end`.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

ASR_SAMPLE_RATE = 16000


@dataclass
class Word:
    text: str
    start: float
    end: float
    confidence: float = 1.0

    def to_json(self, index: int) -> dict:
        return {
            "id": f"w{index}",
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class Segment:
    """A sentence-ish grouping, used purely for paragraph layout in the editor."""

    start: float
    end: float
    text: str
    first_word: int
    last_word: int


@dataclass
class Transcript:
    words: list[Word] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    language: str = "en"
    backend: str = "unknown"


class AsrBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def transcribe(self, samples: np.ndarray, language: str | None) -> Transcript: ...


def _sanitise(words: list[Word], duration: float) -> list[Word]:
    """Force timings to be monotonic, positive and inside the media.

    ASR models occasionally emit a word that ends before it starts, or that
    overlaps its neighbour. Left alone those produce zero-length or backwards
    ffmpeg trims, so they are clamped here rather than in the renderer.
    """
    cleaned: list[Word] = []
    previous_end = 0.0
    for w in words:
        text = w.text.strip()
        if not text:
            continue
        start = max(0.0, float(w.start))
        end = float(w.end)
        if duration > 0:
            start = min(start, duration)
            end = min(end, duration)
        start = max(start, previous_end)
        if end <= start:
            end = start + 0.02  # a plausible floor for a very short word
            if duration > 0:
                end = min(end, duration)
                start = min(start, max(0.0, end - 0.01))
        cleaned.append(Word(text=text, start=start, end=end, confidence=w.confidence))
        previous_end = end
    return cleaned


class FasterWhisperBackend:
    """CTranslate2 Whisper. Fast on CPU and gives well-calibrated word timings."""

    name = "faster-whisper"

    def __init__(self, model_size: str | None = None, device: str | None = None,
                 compute_type: str | None = None) -> None:
        self.model_size = model_size or os.environ.get("VOXDOCS_WHISPER_MODEL", "small")
        self.device = device or os.environ.get("VOXDOCS_ASR_DEVICE", "cpu")
        self.compute_type = compute_type or os.environ.get(
            "VOXDOCS_ASR_COMPUTE", "int8" if self.device == "cpu" else "float16"
        )
        self._model = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self):
        # Model load is expensive and not thread-safe; do it once, lazily, so an
        # unused backend never costs anything.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from faster_whisper import WhisperModel
                    log.info("loading faster-whisper %s on %s (%s)",
                             self.model_size, self.device, self.compute_type)
                    self._model = WhisperModel(
                        self.model_size, device=self.device, compute_type=self.compute_type
                    )
        return self._model

    def transcribe(self, samples: np.ndarray, language: str | None) -> Transcript:
        model = self._load()
        duration = samples.size / ASR_SAMPLE_RATE
        segments_iter, info = model.transcribe(
            samples,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,  # avoids runaway repetition loops
        )

        words: list[Word] = []
        segments: list[Segment] = []
        for seg in segments_iter:
            seg_words = list(seg.words or [])
            if not seg_words:
                continue
            first = len(words)
            for w in seg_words:
                words.append(Word(
                    text=w.word.strip(),
                    start=float(w.start),
                    end=float(w.end),
                    confidence=float(getattr(w, "probability", 1.0) or 1.0),
                ))
            if len(words) > first:
                segments.append(Segment(
                    start=float(seg.start), end=float(seg.end),
                    text=seg.text.strip(), first_word=first, last_word=len(words) - 1,
                ))

        words = _sanitise(words, duration)
        segments = _regroup(segments, words)
        return Transcript(
            words=words,
            segments=segments,
            language=getattr(info, "language", language or "en"),
            backend=f"{self.name}:{self.model_size}",
        )


class SileroBackend:
    """Silero STT, the engine used by the original VoxDocs prototype.

    Kept as an option because it is small and permissively licensed. Its decoder
    reports per-token alignments rather than true word boundaries, so timings
    are coarser than Whisper's; prefer it when model size matters more than
    cut precision.
    """

    name = "silero"

    def __init__(self, language: str = "en") -> None:
        self.language = language
        self._model = None
        self._decoder = None
        self._utils = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    log.info("loading silero stt (%s)", self.language)
                    model, decoder, utils = torch.hub.load(
                        repo_or_dir="snakers4/silero-models",
                        model="silero_stt",
                        language=self.language,
                        device=torch.device("cpu"),
                        trust_repo=True,
                    )
                    self._model, self._decoder, self._utils = model, decoder, utils
        return self._model, self._decoder, self._utils

    def transcribe(self, samples: np.ndarray, language: str | None) -> Transcript:
        import torch

        model, decoder, utils = self._load()
        duration = samples.size / ASR_SAMPLE_RATE

        # Silero is trained on ~15 s utterances; long input must be chunked or
        # the alignment drifts badly toward the end.
        chunk_seconds = 15.0
        chunk = int(chunk_seconds * ASR_SAMPLE_RATE)
        words: list[Word] = []
        for offset in range(0, max(1, samples.size), chunk):
            block = samples[offset:offset + chunk]
            if block.size < ASR_SAMPLE_RATE // 10:
                continue
            tensor = torch.from_numpy(block.copy()).unsqueeze(0)
            with torch.no_grad():
                output = model(tensor)
            base = offset / ASR_SAMPLE_RATE
            for item in decoder(output[0].cpu(), wordlevel=True):
                # Silero returns (word, start, end) triples in seconds.
                if isinstance(item, dict):
                    text, start, end = item.get("word", ""), item.get("start_ts", 0), item.get("end_ts", 0)
                else:
                    text, start, end = item[0], item[1], item[2]
                words.append(Word(text=str(text).strip(),
                                  start=base + float(start), end=base + float(end)))

        words = _sanitise(words, duration)
        return Transcript(
            words=words,
            segments=_regroup([], words),
            language=language or self.language,
            backend=self.name,
        )


def _regroup(segments: list[Segment], words: list[Word]) -> list[Segment]:
    """Rebuild segment spans after sanitising, or invent them if absent.

    Sanitising can drop words, which invalidates the index ranges the ASR
    backend reported. Rather than trying to patch them up, segments are derived
    from the final word list: break on sentence-final punctuation or a long
    pause, which is what a reader expects a paragraph to be.
    """
    if not words:
        return []

    result: list[Segment] = []
    start_index = 0
    for i, w in enumerate(words):
        is_last = i == len(words) - 1
        ends_sentence = w.text.rstrip().endswith((".", "?", "!", "…"))
        long_pause = not is_last and (words[i + 1].start - w.end) > 0.7
        too_long = i - start_index >= 60
        if is_last or ends_sentence or long_pause or too_long:
            chunk = words[start_index:i + 1]
            result.append(Segment(
                start=chunk[0].start,
                end=chunk[-1].end,
                text=" ".join(c.text for c in chunk),
                first_word=start_index,
                last_word=i,
            ))
            start_index = i + 1
    return result


def build_backend(name: str) -> AsrBackend:
    """Instantiate an ASR backend by name."""
    name = (name or "").lower()
    if name in ("faster-whisper", "whisper", "auto", ""):
        return FasterWhisperBackend()
    if name == "silero":
        return SileroBackend()
    raise ValueError(f"unknown ASR backend: {name!r}")


def select_backend(preferred: str) -> AsrBackend:
    """Pick the preferred backend, falling back to whatever is installed."""
    candidates = [preferred] if preferred and preferred != "auto" else []
    candidates += ["faster-whisper", "silero"]
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            backend = build_backend(name)
        except ValueError:
            continue
        if backend.available():
            return backend
    raise RuntimeError(
        "no ASR backend available: install faster-whisper (pip install faster-whisper) "
        "or torch for the Silero backend"
    )
