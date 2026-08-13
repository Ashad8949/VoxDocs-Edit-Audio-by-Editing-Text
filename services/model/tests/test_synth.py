"""Unit-selection tests. These are pure logic: no models, no audio, no network."""

import numpy as np
import pytest

from voxdocs.audio import VoiceStats
from voxdocs.synth import (
    MAX_NGRAM,
    Synthesizer,
    TtsBackend,
    Unit,
    VoiceProfile,
    ProfileWord,
    normalize_token,
    select_units,
    tokenize,
)

SENTENCE = ("four score and seven years ago our fathers brought forth on this "
            "continent a new nation conceived in liberty")


def make_profile(sentence: str = SENTENCE, gap: float = 0.1) -> VoiceProfile:
    """One word every 0.5 s: 0.4 s of speech then a 0.1 s gap."""
    words = [
        ProfileWord(text=t, start=i * 0.5, end=i * 0.5 + 0.4, confidence=0.9)
        for i, t in enumerate(sentence.split())
    ]
    profile = VoiceProfile(
        project_id="p",
        words=words,
        stats=VoiceStats(120.0, 0.1, 0.8, 16000),
        duration=len(words) * 0.5,
        sample_rate=16000,
        median_gap=gap,
        sec_per_word=0.5,
    )
    profile.build_index()
    return profile


class RecordingTts(TtsBackend):
    """A TTS stand-in that records what it was asked to say."""

    name = "recording"

    def __init__(self, works: bool = True):
        self.calls: list[str] = []
        self.works = works

    def available(self) -> bool:
        return True

    def synthesize(self, text: str, profile: VoiceProfile):
        self.calls.append(text)
        if not self.works:
            raise RuntimeError("backend is down")
        return np.zeros(1600, dtype=np.float32), 16000


def test_normalize_token_matches_the_javascript_rules():
    assert normalize_token(" The") == "the"
    assert normalize_token("space.") == "space"
    assert normalize_token("don't") == "dont"
    assert normalize_token("don’t") == "dont"
    assert normalize_token("—") == ""
    assert normalize_token("246") == "246"


def test_tokenize_drops_punctuation_only_words():
    assert tokenize("hello, world — ok") == ["hello,", "world", "ok"]
    assert tokenize("   ") == []


def test_single_word_present_in_the_source_is_lifted_not_synthesised():
    profile = make_profile()
    units, covered = select_units(profile, ["liberty"], None, None)
    assert covered == ["liberty"]
    assert len(units) == 1
    assert isinstance(units[0], Unit)
    assert units[0].type == "source"
    assert units[0].origin == "voice-bank"


def test_longest_contiguous_run_is_lifted_in_one_piece():
    profile = make_profile()
    units, covered = select_units(profile, ["seven", "years", "ago"], None, None)
    assert covered == ["seven", "years", "ago"]
    assert len(units) == 1, "a contiguous run must not be chopped into three splices"
    assert units[0].word == "seven years ago"
    # "seven" is index 3 -> starts at 1.5, "ago" is index 5 -> ends at 2.9.
    assert units[0].start == pytest.approx(1.5, abs=0.03)
    assert units[0].end == pytest.approx(2.9, abs=0.03)


def test_ngram_matching_is_capped_so_long_phrases_still_terminate():
    profile = make_profile()
    phrase = SENTENCE.split()[:MAX_NGRAM + 3]
    units, covered = select_units(profile, phrase, None, None)
    assert covered == phrase
    assert all(isinstance(u, Unit) for u in units)
    assert max(len(u.word.split()) for u in units) <= MAX_NGRAM


def test_words_absent_from_the_source_are_left_for_tts():
    profile = make_profile()
    units, covered = select_units(profile, ["246", "years"], None, None)
    assert covered == ["years"]
    assert units[0] == "246", "an unresolved word is passed through as plain text"
    assert isinstance(units[1], Unit)


def test_context_steers_selection_toward_the_right_occurrence():
    # "the" appears twice: once before "cat", once before "dog".
    words = "the cat sat and then the dog ran".split()
    profile = make_profile(" ".join(words))
    units, _ = select_units(profile, ["the"], context_before=None, context_after="dog")
    # "the" before "dog" is index 5 -> starts at 2.5.
    assert units[0].start == pytest.approx(2.5, abs=0.03)

    units, _ = select_units(profile, ["the"], context_before=None, context_after="cat")
    assert units[0].start == pytest.approx(0.0, abs=0.03)


def test_lifted_bounds_stay_inside_the_neighbouring_gaps():
    profile = make_profile()
    units, _ = select_units(profile, ["score"], None, None)
    unit = units[0]
    # "score" is index 1: [0.5, 0.9], neighbours end at 0.4 and start at 1.0.
    assert unit.start >= 0.4 and unit.start <= 0.5
    assert unit.end >= 0.9 and unit.end <= 1.0


def test_lifted_bounds_are_clamped_at_the_start_of_the_file():
    profile = make_profile()
    units, _ = select_units(profile, ["four"], None, None)
    assert units[0].start >= 0.0


def test_case_and_punctuation_do_not_prevent_a_match():
    profile = make_profile()
    units, covered = select_units(profile, ["Liberty!"], None, None)
    assert covered == ["Liberty!"]
    assert isinstance(units[0], Unit)


def test_synthesizer_prefers_the_voice_bank_over_tts():
    profile = make_profile()
    tts = RecordingTts()
    result = Synthesizer(tts_backends=[tts]).synthesize(profile, "seven years ago")
    assert tts.calls == [], "no TTS call when the speaker already said the words"
    assert result.coverage == 1.0
    assert result.backends == ["voice-bank"]
    assert result.missing == []


def test_consecutive_unknown_words_are_synthesised_as_one_phrase():
    profile = make_profile()
    tts = RecordingTts()
    result = Synthesizer(tts_backends=[tts]).synthesize(profile, "zebra xylophone years")
    # One call for both unknown words, not two: phrase-level prosody matters.
    assert tts.calls == ["zebra xylophone"]
    assert result.generated == ["zebra", "xylophone"]
    assert result.covered == ["years"]
    assert 0.0 < result.coverage < 1.0


def test_a_failing_tts_backend_falls_through_to_the_next():
    profile = make_profile()
    broken = RecordingTts(works=False)
    working = RecordingTts()
    result = Synthesizer(tts_backends=[broken, working]).synthesize(profile, "zebra")
    assert broken.calls == ["zebra"] and working.calls == ["zebra"]
    assert result.generated == ["zebra"]
    assert result.missing == []


def test_words_nothing_can_produce_are_reported_not_silently_dropped():
    profile = make_profile()
    result = Synthesizer(tts_backends=[]).synthesize(profile, "zebra years")
    assert result.missing == ["zebra"]
    assert result.covered == ["years"]


def test_silence_is_inserted_between_separate_splices():
    profile = make_profile()
    result = Synthesizer(tts_backends=[]).synthesize(profile, "liberty four")
    kinds = [u.type for u in result.units]
    assert kinds == ["source", "silence", "source"]
    assert result.units[1].duration == pytest.approx(profile.median_gap)


def test_lead_and_trail_gaps_bracket_the_insertion():
    profile = make_profile()
    result = Synthesizer(tts_backends=[]).synthesize(
        profile, "liberty", lead_gap=0.2, trail_gap=0.3
    )
    assert result.units[0].type == "silence" and result.units[0].duration == pytest.approx(0.2)
    assert result.units[-1].type == "silence" and result.units[-1].duration == pytest.approx(0.3)


def test_empty_text_produces_no_units():
    profile = make_profile()
    result = Synthesizer(tts_backends=[]).synthesize(profile, "   ")
    assert result.units == []
    assert result.coverage == 1.0


def test_disabling_the_voice_bank_routes_everything_to_tts():
    profile = make_profile()
    tts = RecordingTts()
    synth = Synthesizer(enable_voice_bank=False, tts_backends=[tts])
    result = synth.synthesize(profile, "seven years ago")
    assert tts.calls == ["seven years ago"]
    assert result.covered == []


def test_unit_json_is_serialisable_and_hides_raw_bytes():
    unit = Unit(type="audio", word="hi", sample_rate=16000, data=b"RIFFdata", origin="x")
    payload = unit.to_json()
    assert payload["format"] == "wav"
    assert isinstance(payload["data"], str)  # base64, not bytes
    source = Unit(type="source", word="hi", start=1.0, end=2.0, origin="voice-bank").to_json()
    assert "data" not in source and source["start"] == 1.0
