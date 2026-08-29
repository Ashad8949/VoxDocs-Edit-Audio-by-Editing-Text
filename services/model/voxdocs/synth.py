"""Speech synthesis for inserted text.

Four strategies, tried in order, because they trade off very differently:

1. **Unit selection from the speaker's own recording.** If the words the user
   typed were already said somewhere in the file, the best possible synthesis
   is the speaker actually saying them. This costs no model, no GPU and no
   download, and the result is indistinguishable from the surrounding audio
   because it *is* the surrounding audio. Longest-match n-gram selection keeps
   naturally coarticulated runs intact.

2. **Neural voice cloning (XTTS-v2).** For words that were never said, cloning
   the speaker's voice from a clip of their own audio is the best available
   synthesis — natural prosody, and it actually sounds like them, unlike the
   formant fallback below. The same model voice_clone.py uses for dubbing.

3. **Neural voice cloning (PaddleSpeech).** The stack from the original
   VoxDocs prototype: GE2E speaker embedding conditioning a FastSpeech2
   acoustic model, vocoded by Parallel WaveGAN. Disabled unless
   VOXDOCS_ENABLE_PADDLE is set; XTTS supersedes it when both are available.

4. **Formant-matched fallback.** When no neural model is installed, eSpeak NG
   synthesises the words and the result is pitch- and level-matched to the
   speaker. It does not sound like them, and it is not meant to: it makes the
   edit audible and reviewable rather than silently dropping words.

The result is not audio but a *plan*: a list of units the render pipeline
expands. Units that come from the source are expressed as time ranges, so the
renderer lifts them from the original full-quality master rather than from a
resampled copy shipped over HTTP.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field

import numpy as np

from . import audio as audio_util
from .voice_clone import XttsVoiceCloner, select_reference_clips
from .voice_rvc import RvcConverter

log = logging.getLogger(__name__)

# Longest run of words we will try to lift from the source in one piece.
MAX_NGRAM = 6
# Padding kept around a lifted word, in seconds.
UNIT_PAD = 0.02


def normalize_token(raw: str) -> str:
    """Mirror of the JavaScript ``normalizeToken`` so both sides agree."""
    import unicodedata

    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", raw).lower()
    text = text.replace("‘", "").replace("’", "").replace("ʼ", "").replace("'", "")
    return "".join(ch for ch in text if ch.isalnum())


def tokenize(text: str) -> list[str]:
    """Split text into pronounceable surface words."""
    return [t for t in (text or "").split() if normalize_token(t)]


@dataclass
class ProfileWord:
    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class VoiceProfile:
    """Everything the synthesiser knows about one project's speaker."""

    project_id: str
    words: list[ProfileWord]
    stats: audio_util.VoiceStats
    duration: float
    sample_rate: int
    median_gap: float = 0.08
    sec_per_word: float = 0.34
    # Retained only when a neural backend needs a speaker embedding.
    samples: np.ndarray | None = None
    _index: dict[str, list[int]] = field(default_factory=dict, repr=False)

    def build_index(self) -> None:
        index: dict[str, list[int]] = {}
        for i, w in enumerate(self.words):
            key = normalize_token(w.text)
            if key:
                index.setdefault(key, []).append(i)
        self._index = index

    @property
    def index(self) -> dict[str, list[int]]:
        if not self._index and self.words:
            self.build_index()
        return self._index

    def vocabulary_size(self) -> int:
        return len(self.index)


@dataclass
class Unit:
    """One piece of the synthesised output."""

    type: str                       # "source" | "audio" | "silence"
    word: str = ""
    start: float = 0.0              # source units only
    end: float = 0.0                # source units only
    duration: float = 0.0           # silence units only
    gain: float = 1.0
    sample_rate: int = 0            # audio units only
    data: bytes | None = None       # audio units only, WAV container
    origin: str = ""                # which strategy produced this

    def to_json(self) -> dict:
        out: dict = {"type": self.type, "origin": self.origin}
        if self.word:
            out["word"] = self.word
        if self.type == "source":
            out["start"] = round(self.start, 4)
            out["end"] = round(self.end, 4)
            out["gain"] = round(self.gain, 4)
        elif self.type == "silence":
            out["duration"] = round(self.duration, 4)
        elif self.type == "audio":
            out["sample_rate"] = self.sample_rate
            out["format"] = "wav"
            out["data"] = base64.b64encode(self.data or b"").decode("ascii")
        return out


@dataclass
class SynthesisResult:
    units: list[Unit]
    backends: list[str]
    covered: list[str]              # words taken from the speaker's own voice
    generated: list[str]            # words produced by a TTS model
    missing: list[str]              # words nothing could produce

    @property
    def coverage(self) -> float:
        total = len(self.covered) + len(self.generated) + len(self.missing)
        return len(self.covered) / total if total else 1.0

    def to_json(self) -> dict:
        return {
            "units": [u.to_json() for u in self.units],
            "backends": self.backends,
            "covered": self.covered,
            "generated": self.generated,
            "missing": self.missing,
            "coverage": round(self.coverage, 4),
        }


# --------------------------------------------------------------------------
# Strategy 1: unit selection from the speaker's own audio
# --------------------------------------------------------------------------

def _find_runs(profile: VoiceProfile, ngram: list[str]) -> list[int]:
    """Start indices where ``ngram`` occurs contiguously in the source."""
    if not ngram:
        return []
    candidates = profile.index.get(ngram[0], [])
    if len(ngram) == 1:
        return list(candidates)
    hits = []
    for start in candidates:
        if start + len(ngram) > len(profile.words):
            continue
        if all(normalize_token(profile.words[start + k].text) == ngram[k]
               for k in range(1, len(ngram))):
            hits.append(start)
    return hits


def _score_run(profile: VoiceProfile, start: int, length: int,
               left_context: str | None, right_context: str | None) -> float:
    """Rank candidate source runs; higher is better.

    Preferring a run whose neighbours match the target context means the lifted
    audio carries the right coarticulation into and out of the splice, which is
    most of what makes concatenation sound seamless.
    """
    words = profile.words
    end = start + length - 1
    score = 0.0

    if left_context:
        left = normalize_token(words[start - 1].text) if start > 0 else ""
        if left and left == normalize_token(left_context):
            score += 3.0
    if right_context:
        right = normalize_token(words[end + 1].text) if end + 1 < len(words) else ""
        if right and right == normalize_token(right_context):
            score += 3.0

    # Trust well-recognised audio more.
    score += 2.0 * float(np.mean([words[i].confidence for i in range(start, end + 1)]))

    # A pause on either side gives a cleaner boundary to cut at.
    if start > 0 and words[start].start - words[start - 1].end > 0.06:
        score += 0.6
    if end + 1 < len(words) and words[end + 1].start - words[end].end > 0.06:
        score += 0.6

    # Avoid pathologically clipped or drawn-out realisations.
    span = words[end].end - words[start].start
    expected = length * profile.sec_per_word
    if expected > 0:
        score -= min(2.0, abs(span - expected) / expected)
    return score


def _extract_bounds(profile: VoiceProfile, start: int, length: int) -> tuple[float, float]:
    """Time range to lift, padded into the surrounding gaps but not beyond."""
    words = profile.words
    end = start + length - 1
    lo = words[start].start
    hi = words[end].end

    if start > 0:
        gap = words[start].start - words[start - 1].end
        lo -= min(UNIT_PAD, max(0.0, gap) / 2)
    else:
        lo = max(0.0, lo - UNIT_PAD)

    if end + 1 < len(words):
        gap = words[end + 1].start - words[end].end
        hi += min(UNIT_PAD, max(0.0, gap) / 2)
    else:
        hi = min(profile.duration, hi + UNIT_PAD) if profile.duration > 0 else hi + UNIT_PAD
    return max(0.0, lo), max(lo + 0.01, hi)


def select_units(profile: VoiceProfile, phrase: list[str],
                 context_before: str | None, context_after: str | None
                 ) -> tuple[list[Unit | str], list[str]]:
    """Greedy longest-match unit selection.

    Returns a mixed list where a ``Unit`` is resolved audio and a bare ``str``
    is a word that still needs a TTS model, plus the list of covered words.
    """
    out: list[Unit | str] = []
    covered: list[str] = []
    keys = [normalize_token(w) for w in phrase]
    i = 0
    while i < len(phrase):
        matched = 0
        best_start = -1
        best_score = -1e9
        for n in range(min(MAX_NGRAM, len(phrase) - i), 0, -1):
            runs = _find_runs(profile, keys[i:i + n])
            if not runs:
                continue
            left = phrase[i - 1] if i > 0 else context_before
            right = phrase[i + n] if i + n < len(phrase) else context_after
            for start in runs:
                score = _score_run(profile, start, n, left, right)
                if score > best_score:
                    best_score = score
                    best_start = start
            matched = n
            break  # longest match wins outright

        if matched and best_start >= 0:
            lo, hi = _extract_bounds(profile, best_start, matched)
            text = " ".join(phrase[i:i + matched])
            out.append(Unit(type="source", word=text, start=lo, end=hi, origin="voice-bank"))
            covered.extend(phrase[i:i + matched])
            i += matched
        else:
            out.append(phrase[i])
            i += 1
    return out, covered


# --------------------------------------------------------------------------
# Strategy 2 and 3: text-to-speech backends
# --------------------------------------------------------------------------

class TtsBackend:
    name = "none"

    def available(self) -> bool:
        return False

    def synthesize(self, text: str, profile: VoiceProfile,
                   target_language: str = "en") -> tuple[np.ndarray, int]:
        raise NotImplementedError


class PaddleSpeechBackend(TtsBackend):
    """GE2E speaker embedding + FastSpeech2 + Parallel WaveGAN voice cloning.

    This mirrors the stack described in the original VoxDocs work. PaddleSpeech
    ships pretrained Chinese voice-cloning weights; for English the acoustic
    model must be trained on an English corpus (LibriTTS or AISHELL-3's English
    counterpart) and the checkpoint path supplied via the environment.
    """

    name = "paddlespeech"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = None
        self._vector = None
        self.am = os.environ.get("VOXDOCS_PADDLE_AM", "fastspeech2_mix")
        self.voc = os.environ.get("VOXDOCS_PADDLE_VOC", "pwgan_aishell3")
        self.lang = os.environ.get("VOXDOCS_PADDLE_LANG", "en")

    def available(self) -> bool:
        if os.environ.get("VOXDOCS_ENABLE_PADDLE", "0") not in ("1", "true", "yes"):
            return False
        try:
            import paddlespeech  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self):
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    from paddlespeech.cli.tts.infer import TTSExecutor
                    from paddlespeech.cli.vector.infer import VectorExecutor

                    log.info("loading paddlespeech tts am=%s voc=%s", self.am, self.voc)
                    self._executor = TTSExecutor()
                    self._vector = VectorExecutor()
        return self._executor, self._vector

    def synthesize(self, text: str, profile: VoiceProfile,
                   target_language: str = "en") -> tuple[np.ndarray, int]:
        executor, vector = self._load()
        if profile.samples is None:
            raise RuntimeError("voice cloning needs the source audio; none was cached")

        # PaddleSpeech's mixed acoustic model covers en/zh; anything else it
        # can't speak natively falls back to the configured default lang.
        lang = {"en": "en", "hi": "mix", "hinglish": "mix"}.get(target_language, self.lang)
        with tempfile.TemporaryDirectory() as tmp:
            reference = os.path.join(tmp, "reference.wav")
            output = os.path.join(tmp, "out.wav")
            with open(reference, "wb") as fh:
                fh.write(audio_util.encode_wav(profile.samples, profile.sample_rate))

            # GE2E embedding of the target speaker conditions the acoustic model.
            embedding = vector(audio_file=reference)
            executor(
                text=text,
                output=output,
                am=self.am,
                voc=self.voc,
                lang=lang,
                spk_emb=embedding,
            )
            samples = audio_util.decode(output, profile.sample_rate)
        return samples, profile.sample_rate


class XttsBackend(TtsBackend):
    """Cross-lingual neural voice cloning via Coqui XTTS-v2.

    Conditions directly on a clip of the project's own source audio, so
    inserted words come out sounding like the actual speaker rather than a
    pitch-matched formant voice — the same model voice_clone.py uses for
    dubbing, reused here for same-language insertions too. Tried before
    PaddleSpeech/eSpeak since it is the highest-quality option available.
    """

    name = "xtts"

    def __init__(self) -> None:
        self._cloner = XttsVoiceCloner()

    def available(self) -> bool:
        return self._cloner.available()

    def synthesize(self, text: str, profile: VoiceProfile,
                   target_language: str = "en") -> tuple[np.ndarray, int]:
        if profile.samples is None:
            raise RuntimeError("voice cloning needs the source audio; none was cached")

        # Condition on the cleanest, most confident speech in the recording —
        # several short clips beat one arbitrary head slice (see
        # select_reference_clips).
        clips = select_reference_clips(
            profile.samples,
            profile.sample_rate,
            [(w.start, w.end, w.confidence) for w in profile.words],
        )
        with tempfile.TemporaryDirectory() as tmp:
            ref_paths = []
            for i, clip in enumerate(clips):
                path = os.path.join(tmp, f"reference-{i}.wav")
                with open(path, "wb") as fh:
                    fh.write(audio_util.encode_wav(clip, profile.sample_rate))
                ref_paths.append(path)
            refs = ref_paths[0] if len(ref_paths) == 1 else ref_paths
            samples = self._cloner.synthesize(text, target_language, refs)

        rate = self._cloner.sample_rate
        samples = self._match_speaker(samples, rate, profile.stats)
        return samples, rate

    @staticmethod
    def _match_speaker(samples: np.ndarray, rate: int,
                       stats: "audio_util.VoiceStats") -> np.ndarray:
        """Gently pull the clone toward the speaker's own pitch and loudness.

        XTTS is already close, so this is a light correction, not a heavy
        transform: pitch is only nudged when it has drifted more than ~8% (and
        the shift is clamped tight to avoid artefacts), and loudness is matched
        to the speaker's measured level. Same idea eSpeak uses, applied softly.
        """
        if samples.size == 0:
            return samples
        target_f0 = stats.median_f0
        if target_f0 > 0:
            produced = audio_util.estimate_f0(samples, rate)
            if produced > 0:
                ratio = target_f0 / produced
                if abs(ratio - 1.0) > 0.08:
                    samples = audio_util.pitch_shift(
                        samples, rate, float(np.clip(ratio, 0.85, 1.18))
                    )
        if stats.speech_rms > 0:
            samples = audio_util.match_loudness(samples, stats.speech_rms)
        return samples


class RvcBackend(TtsBackend):
    """Pro tier: exact-speaker match via Retrieval-based Voice Conversion.

    XTTS produces the words in an approximate clone; RVC then converts that
    audio into the project speaker's *exact* timbre using a small model trained
    on that speaker (see the backend's Kaggle training pipeline). Listed before
    XTTS in the Pro chain, so it wins when a trained model exists — and cleanly
    yields to XTTS when it doesn't: `convert` raises on a missing per-project
    model, and `_run_tier` treats that as "try the next engine".
    """

    name = "rvc"

    def __init__(self, xtts: "XttsBackend") -> None:
        self._xtts = xtts
        self._converter = RvcConverter()

    def available(self) -> bool:
        # Global availability only (the RVC lib is importable). Whether *this*
        # project has a trained model is decided in synthesize(), which raises
        # to fall through to XTTS when it doesn't.
        return self._converter.available() and self._xtts.available()

    def synthesize(self, text: str, profile: VoiceProfile,
                   target_language: str = "en") -> tuple[np.ndarray, int]:
        base, base_rate = self._xtts.synthesize(text, profile, target_language)
        converted, rate = self._converter.convert(base, base_rate, profile.project_id)
        return XttsBackend._match_speaker(converted, rate, profile.stats), rate


class EspeakBackend(TtsBackend):
    """Formant synthesis, pitch- and level-matched to the speaker.

    Deliberately the last resort. It keeps an edit audible rather than silently
    dropping words, and it never fails for lack of a downloaded model.
    """

    name = "espeak-ng"

    def available(self) -> bool:
        return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None

    def synthesize(self, text: str, profile: VoiceProfile,
                   target_language: str = "en") -> tuple[np.ndarray, int]:
        binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if not binary:
            raise RuntimeError("espeak-ng is not installed")

        target_f0 = profile.stats.median_f0
        if target_language in ("hi", "hinglish"):
            # eSpeak has a Hindi voice but no gendered +fN variants for it.
            voice = "hi"
        else:
            # A higher-pitched speaker gets a voice variant closer to their
            # register, which leaves less work for the pitch shifter and sounds
            # less artefacty.
            voice = "en-us+f3" if target_f0 >= 165 else "en-us"
        words_per_minute = int(np.clip(60.0 / max(profile.sec_per_word, 0.12), 110, 260))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tts.wav")
            cmd = [binary, "-v", voice, "-s", str(words_per_minute), "-w", path, text]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0 or not os.path.exists(path):
                raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:300] or "espeak failed")
            samples = audio_util.decode(path, profile.sample_rate)

        if samples.size == 0:
            raise RuntimeError("espeak produced no audio")

        # Measure what came out and bend it toward the speaker.
        if target_f0 > 0:
            produced = audio_util.estimate_f0(samples, profile.sample_rate)
            if produced > 0:
                ratio = float(np.clip(target_f0 / produced, 0.7, 1.45))
                samples = audio_util.pitch_shift(samples, profile.sample_rate, ratio)
        if profile.stats.speech_rms > 0:
            samples = audio_util.match_loudness(samples, profile.stats.speech_rms)
        return samples, profile.sample_rate


# --------------------------------------------------------------------------
# Quality tiers
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TierConfig:
    """One subscription quality level.

    ``engines`` is the ordered generative-engine chain tried for words the
    source never spoke; the first *available* one that produces audio wins,
    so higher tiers simply list better engines first and fall through to the
    same safety net. ``postprocess`` pitch/loudness-matches the generated
    audio toward the speaker (wired in Phase 1; eSpeak already self-matches).
    """

    engines: tuple[str, ...]
    postprocess: bool = False


# Engine names refer to TtsBackend.name. "rvc" and "xtts-ft" are future
# (Pro/Studio) engines that are not registered yet — listing them here is
# deliberate: until they exist those tiers transparently fall through to
# XTTS, so selecting Pro today behaves like Standard rather than erroring.
TIERS: dict[str, TierConfig] = {
    "free":     TierConfig(("espeak-ng",)),
    "standard": TierConfig(("xtts", "paddlespeech", "espeak-ng"), postprocess=True),
    "pro":      TierConfig(("rvc", "xtts", "paddlespeech", "espeak-ng"), postprocess=True),
    "studio":   TierConfig(("xtts-ft", "xtts", "paddlespeech", "espeak-ng"), postprocess=True),
}
DEFAULT_TIER = "standard"


def resolve_tier(tier: str | None) -> str:
    """Normalise an untrusted tier string to a known one."""
    return tier if tier in TIERS else DEFAULT_TIER


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

class Synthesizer:
    """Applies the three strategies in order and assembles the unit plan."""

    def __init__(self, enable_voice_bank: bool = True,
                 tts_backends: list[TtsBackend] | None = None) -> None:
        self.enable_voice_bank = enable_voice_bank
        if tts_backends is not None:
            self.tts_backends = tts_backends
        else:
            xtts = XttsBackend()
            # RvcBackend composes XTTS (XTTS generates, RVC re-timbres), so it
            # shares the one XTTS instance rather than loading the model twice.
            self.tts_backends = [
                RvcBackend(xtts),
                xtts,
                PaddleSpeechBackend(),
                EspeakBackend(),
            ]
        self._by_name = {b.name: b for b in self.tts_backends}

    def describe(self) -> dict:
        return {
            "voice_bank": self.enable_voice_bank,
            "tts": [
                {"name": b.name, "available": b.available()}
                for b in self.tts_backends
            ],
            "tiers": {
                name: {
                    # The engine each tier actually resolves to right now,
                    # after skipping unbuilt (rvc/xtts-ft) and unavailable ones.
                    "engine": next(
                        (e for e in cfg.engines
                         if e in self._by_name and self._by_name[e].available()),
                        None,
                    ),
                    "postprocess": cfg.postprocess,
                }
                for name, cfg in TIERS.items()
            },
        }

    def _run_tier(self, text: str, profile: VoiceProfile, tier: str,
                  target_language: str) -> tuple[np.ndarray, int, str] | None:
        """Walk a tier's engine chain, returning the first engine's
        (samples, sample_rate, engine_name) or None if none produced audio.
        """
        config = TIERS[resolve_tier(tier)]
        for name in config.engines:
            backend = self._by_name.get(name)
            if backend is None or not backend.available():
                continue
            try:
                samples, rate = backend.synthesize(text, profile, target_language)
            except Exception as exc:  # a broken backend must not sink the request
                log.warning("tts backend %s failed: %s", name, exc)
                continue
            if samples.size == 0:
                continue
            return samples, rate, backend.name
        return None

    def clone_phrase(self, profile: VoiceProfile, text: str,
                     tier: str = DEFAULT_TIER,
                     target_language: str = "en") -> tuple[np.ndarray, int, str] | None:
        """Generative-only synthesis of a whole phrase in the speaker's voice
        for the given tier — no voice-bank splicing. Used by dubbing, where a
        segment is regenerated as one continuous cross-lingual utterance.
        """
        return self._run_tier(text, profile, tier, target_language)

    def _tts(self, text: str, profile: VoiceProfile, tier: str,
             target_language: str = "en") -> tuple[Unit | None, str]:
        produced = self._run_tier(text, profile, tier, target_language)
        if produced is None:
            return None, ""
        samples, rate, name = produced
        return Unit(
            type="audio",
            word=text,
            sample_rate=rate,
            data=audio_util.encode_wav(samples, rate),
            origin=name,
        ), name

    def synthesize(self, profile: VoiceProfile, text: str,
                   context_before: str | None = None,
                   context_after: str | None = None,
                   lead_gap: float = 0.0,
                   trail_gap: float = 0.0,
                   tier: str = DEFAULT_TIER) -> SynthesisResult:
        phrase = tokenize(text)
        if not phrase:
            return SynthesisResult([], [], [], [], [])

        if self.enable_voice_bank and profile.words:
            resolved, covered = select_units(profile, phrase, context_before, context_after)
        else:
            resolved, covered = list(phrase), []

        fully_covered = bool(covered) and not any(isinstance(item, str) for item in resolved)

        units: list[Unit] = []
        backends: list[str] = []
        generated: list[str] = []
        missing: list[str] = []
        gap = profile.median_gap if profile.median_gap > 0 else 0.08

        if fully_covered:
            # Every word already exists in the speaker's own recording: splice
            # those clips directly. Real audio always beats even a good clone.
            backends.append("voice-bank")
            for item in resolved:
                if units:
                    units.append(Unit(type="silence", duration=gap, origin="spacing"))
                units.append(item)
        else:
            # Splicing a handful of voice-bank words into a mostly-synthesised
            # phrase sounds worse than it should: each spliced word carries the
            # pitch and pacing of a *different* sentence, so the result stutters
            # — stops and starts — exactly where the source changes. A neural
            # voice carrying the *whole* phrase in one continuous pass keeps
            # prosody smooth; that only costs something when nothing needs
            # synthesis at all, which is the branch above.
            unit, backend = self._tts(text, profile, tier)
            if unit is not None:
                units.append(unit)
                generated = list(phrase)
                backends.append(backend)
            else:
                missing = list(phrase)

        if units:
            if lead_gap > 0:
                units.insert(0, Unit(type="silence", duration=lead_gap, origin="spacing"))
            if trail_gap > 0:
                units.append(Unit(type="silence", duration=trail_gap, origin="spacing"))

        return SynthesisResult(
            units=units,
            backends=backends,
            covered=covered if fully_covered else [],
            generated=generated,
            missing=missing,
        )
