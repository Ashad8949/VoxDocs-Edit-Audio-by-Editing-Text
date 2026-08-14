"""Edit Decision List construction.

An EDL bridges "what the transcript now says" and "what the renderer must do
with samples". Every surviving stretch of original speech becomes a ``copy``
segment carrying an exact source time range; every stretch the user typed
becomes a ``synth`` segment carrying the text plus its neighbouring words, which
the synthesiser uses to match prosody.

The interesting decisions are about *where to cut*. ASR word timings mark
roughly where the vowel energy is, not where the silence is. Cutting exactly on
``word.end`` reliably clips the release of a final stop consonant and produces
the tell-tale "chopped" sound of naive text-based editing. Instead cuts land in
the middle of the gap between words, bounded so removing a word never drags a
long silence with it, and optionally snapped to the quietest point nearby using
an energy envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .tokens import tokenize_text

#: Most silence (seconds) pulled in on either side of a cut.
DEFAULT_MAX_PAD = 0.12
#: How far (seconds) to search for a quieter place to cut.
DEFAULT_SNAP_WINDOW = 0.06
#: Silence around synthesised speech when the source gives no clue.
DEFAULT_SYNTH_GAP = 0.08
#: Fallback speech rate when the source is too short to measure.
FALLBACK_SEC_PER_WORD = 0.34
#: A candidate cut must be this much quieter before it is worth moving to.
SNAP_IMPROVEMENT = 0.7


@dataclass
class Word:
    """A transcript word. Mirrors the model, but usable without the ORM."""

    id: str
    text: str
    start: float
    end: float

    @classmethod
    def from_any(cls, item) -> "Word":
        get = item.get if isinstance(item, dict) else lambda n: getattr(item, n)
        return cls(str(get("id")), str(get("text")), float(get("start")), float(get("end")))


@dataclass
class Envelope:
    """Short-time RMS of the source, used to nudge cuts into silence."""

    fps: float
    rms: Sequence[float]

    @classmethod
    def from_any(cls, item) -> "Envelope | None":
        if not item:
            return None
        get = item.get if isinstance(item, dict) else lambda n: getattr(item, n)
        rms = get("rms")
        fps = get("fps")
        if not rms or not fps:
            return None
        return cls(float(fps), rms)


@dataclass
class CopySegment:
    """A stretch of the original recording, kept verbatim."""

    start: float
    end: float
    word_ids: list[str]
    first_word_index: int
    last_word_index: int
    kind: str = "copy"


@dataclass
class SynthSegment:
    """Text the user typed, to be spoken by the synthesiser."""

    text: str
    context_before: str | None
    context_after: str | None
    lead_gap: float
    trail_gap: float
    estimated_duration: float
    kind: str = "synth"


Segment = CopySegment | SynthSegment


@dataclass
class EdlStats:
    source_words: int
    kept_words: int
    deleted_words: int
    inserted_words: int
    source_duration: float
    estimated_duration: float
    cuts: int

    def to_json(self) -> dict:
        return {
            "sourceWords": self.source_words,
            "keptWords": self.kept_words,
            "deletedWords": self.deleted_words,
            "insertedWords": self.inserted_words,
            "sourceDuration": round(self.source_duration, 4),
            "estimatedDuration": round(self.estimated_duration, 4),
            "cuts": self.cuts,
        }


@dataclass
class Edl:
    segments: list[Segment] = field(default_factory=list)
    stats: EdlStats | None = None


def speech_rate(words: Sequence[Word]) -> float:
    """The speaker's articulation rate, in seconds per word.

    Used to predict how long inserted text will take in their voice, so the
    editor can show a duration before anything is rendered.
    """
    if not words or len(words) < 4:
        return FALLBACK_SEC_PER_WORD
    spoken = sum(max(0.0, w.end - w.start) for w in words)
    if spoken <= 0:
        return FALLBACK_SEC_PER_WORD
    span = words[-1].end - words[0].start
    gap_share = max(0.0, span - spoken) / len(words)
    return spoken / len(words) + min(gap_share, 0.12)


def median_gap(words: Sequence[Word]) -> float:
    """Median inter-word gap, used to space synthesised insertions naturally."""
    if not words or len(words) < 2:
        return DEFAULT_SYNTH_GAP
    gaps = sorted(
        words[i].start - words[i - 1].end
        for i in range(1, len(words))
        if 0 <= words[i].start - words[i - 1].end < 1.0
    )
    if not gaps:
        return DEFAULT_SYNTH_GAP
    return gaps[len(gaps) // 2]


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def snap_to_quiet(target: float, low: float, high: float,
                  envelope: Envelope | None,
                  window: float = DEFAULT_SNAP_WINDOW) -> float:
    """Find the quietest instant near ``target``, within ``[low, high]``.

    Deliberately conservative. The gap-midpoint target is already a good cut;
    moving it is only justified by a clearly quieter neighbour, and among equally
    quiet candidates the one nearest the target wins. Without this, a uniformly
    quiet envelope would drag every cut to the edge of its search window for no
    acoustic benefit.
    """
    fallback = _clamp(target, low, high)
    if envelope is None or not envelope.rms or not envelope.fps:
        return fallback

    search_low = _clamp(target - window, low, high)
    search_high = _clamp(target + window, low, high)
    if search_high <= search_low:
        return fallback

    last = min(len(envelope.rms) - 1, int(search_high * envelope.fps) + 1)
    first = max(0, min(last, int(search_low * envelope.fps)))
    if last <= first:
        return fallback

    target_index = int(_clamp(round(fallback * envelope.fps), first, last))
    target_value = envelope.rms[target_index]

    best_value = min(envelope.rms[first:last + 1])
    if not best_value < target_value * SNAP_IMPROVEMENT:
        return fallback

    # Among near-optimal candidates prefer the least movement.
    tolerance = best_value * 1.05 + 1e-9
    best_index = target_index
    best_distance = float("inf")
    for i in range(first, last + 1):
        if envelope.rms[i] > tolerance:
            continue
        distance = abs(i - target_index)
        if distance < best_distance:
            best_distance = distance
            best_index = i
    return _clamp(best_index / envelope.fps, low, high)


def _boundary_before(words: Sequence[Word], index: int, *, max_pad: float,
                     envelope: Envelope | None, snap_window: float) -> float:
    """Time at which playback of ``words[index]`` should begin."""
    word = words[index]
    if index == 0:
        pad = min(max_pad, word.start)
        return snap_to_quiet(word.start - pad, max(0.0, word.start - pad), word.start,
                             envelope, snap_window)
    previous = words[index - 1]
    gap = word.start - previous.end
    if gap <= 0:
        return word.start
    pad = min(max_pad, gap / 2)
    return snap_to_quiet(word.start - pad, word.start - gap / 2, word.start,
                         envelope, snap_window)


def _boundary_after(words: Sequence[Word], index: int, *, max_pad: float,
                    envelope: Envelope | None, snap_window: float,
                    duration: float) -> float:
    """Time at which playback of ``words[index]`` should end."""
    word = words[index]
    if index == len(words) - 1:
        room = max(0.0, duration - word.end)
        pad = min(max_pad, room)
        return snap_to_quiet(word.end + pad, word.end, word.end + pad, envelope, snap_window)
    following = words[index + 1]
    gap = following.start - word.end
    if gap <= 0:
        return word.end
    pad = min(max_pad, gap / 2)
    return snap_to_quiet(word.end + pad, word.end, word.end + gap / 2, envelope, snap_window)


def is_contiguous(a: Segment, b: Segment) -> bool:
    """True when playing ``a`` then ``b`` reproduces the original waveform.

    Such a join has no seam, so it needs no crossfade.
    """
    if not isinstance(a, CopySegment) or not isinstance(b, CopySegment):
        return False
    if b.first_word_index != a.last_word_index + 1:
        return False
    return abs(b.start - a.end) < 1e-6


def identity_tokens(words: Iterable) -> list[dict]:
    """The identity edit, keeping every word."""
    return [{"ref": str(Word.from_any(w).id)} for w in words]


def build_edl(words: Iterable, tokens: Iterable[dict], *,
              duration: float | None = None,
              envelope=None,
              max_pad: float = DEFAULT_MAX_PAD,
              snap_window: float = DEFAULT_SNAP_WINDOW) -> Edl:
    """Build an EDL from the original words and the user's edit tokens."""
    word_list = [Word.from_any(w) for w in words]
    token_list = list(tokens)
    envelope_obj = Envelope.from_any(envelope)

    if duration is None:
        duration = word_list[-1].end if word_list else 0.0

    bounds = {
        "max_pad": max_pad,
        "envelope": envelope_obj,
        "snap_window": snap_window,
    }

    index_by_id = {w.id: i for i, w in enumerate(word_list)}
    gap = median_gap(word_list)
    sec_per_word = speech_rate(word_list)

    segments: list[Segment] = []
    run: list[int] = []
    inserted_words = 0
    kept_indices: set[int] = set()

    def flush_run() -> None:
        nonlocal run
        if not run:
            return
        first, last = run[0], run[-1]
        segments.append(CopySegment(
            start=_boundary_before(word_list, first, **bounds),
            end=_boundary_after(word_list, last, duration=duration, **bounds),
            word_ids=[word_list[i].id for i in run],
            first_word_index=first,
            last_word_index=last,
        ))
        run = []

    for token in token_list:
        if "ref" in token:
            index = index_by_id.get(str(token["ref"]))
            if index is None:
                continue  # a stale id: ignore rather than fail the whole edit
            kept_indices.add(index)
            if run and index != run[-1] + 1:
                flush_run()  # a cut: these kept words are not neighbours
            run.append(index)
        elif "insert" in token:
            text = str(token.get("insert") or "").strip()
            word_count = len(tokenize_text(text))
            if word_count == 0:
                continue
            context_before = (
                word_list[run[-1]].text if run else _preceding_word_text(segments, word_list)
            )
            flush_run()
            inserted_words += word_count
            segments.append(SynthSegment(
                text=text,
                context_before=context_before,
                context_after=None,   # filled in below, once we know what follows
                lead_gap=gap,
                trail_gap=gap,
                estimated_duration=word_count * sec_per_word + gap,
            ))
    flush_run()

    # Second pass: give each synth segment the word that follows it, so the
    # synthesiser can pick a coarticulation-friendly candidate from the bank.
    for i, segment in enumerate(segments):
        if not isinstance(segment, SynthSegment):
            continue
        following = segments[i + 1] if i + 1 < len(segments) else None
        if isinstance(following, CopySegment):
            segment.context_after = word_list[following.first_word_index].text
        # Nothing to bridge to at the very start or end of the timeline.
        if i == 0:
            segment.lead_gap = 0.0
        if i == len(segments) - 1:
            segment.trail_gap = 0.0

    estimated = 0.0
    cuts = 0
    for i, segment in enumerate(segments):
        estimated += (
            segment.end - segment.start
            if isinstance(segment, CopySegment)
            else segment.estimated_duration
        )
        if i > 0 and not is_contiguous(segments[i - 1], segment):
            cuts += 1

    return Edl(
        segments=segments,
        stats=EdlStats(
            source_words=len(word_list),
            kept_words=len(kept_indices),
            deleted_words=len(word_list) - len(kept_indices),
            inserted_words=inserted_words,
            source_duration=duration,
            estimated_duration=estimated,
            cuts=cuts,
        ),
    )


def _preceding_word_text(segments: Sequence[Segment], words: Sequence[Word]) -> str | None:
    """Surface text of the last original word emitted so far."""
    for segment in reversed(segments):
        if isinstance(segment, CopySegment):
            return words[segment.last_word_index].text
    return None
