"""EDL tests, ported one-for-one from the original JavaScript suite.

Keeping the assertions identical is the point: this is the evidence that moving
the compiler from Node to Django preserved its behaviour exactly, including the
cut-point arithmetic that decides how the edit sounds.
"""

import pytest

from edl import (
    CopySegment,
    SynthSegment,
    align_tokens,
    build_edl,
    diff_transcript,
    identity_tokens,
    is_contiguous,
    normalize_token,
    snap_to_quiet,
    speech_rate,
    tokenize_text,
)

SENTENCE = "four score and seven years ago our fathers brought forth"


def make_words(sentence: str) -> list[dict]:
    """One word every 0.5 s: 0.4 s of speech then a 0.1 s gap."""
    return [
        {"id": f"w{i}", "text": text, "start": i * 0.5, "end": i * 0.5 + 0.4}
        for i, text in enumerate(sentence.split(" "))
    ]


def describe(tokens, words) -> str:
    by_id = {w["id"]: w["text"] for w in words}
    return " ".join(
        by_id[t["ref"]] if "ref" in t else f"+[{t['insert']}]" for t in tokens
    )


# ---------------------------------------------------------------- tokens

def test_normalize_token_strips_case_punctuation_and_apostrophes():
    assert normalize_token(" The") == "the"
    assert normalize_token("space.") == "space"
    assert normalize_token("don't") == "dont"
    assert normalize_token("don’t") == "dont"
    assert normalize_token("—") == ""
    assert normalize_token("2022") == "2022"


def test_tokenize_text_drops_punctuation_only_tokens():
    assert tokenize_text("hello,  world —  ok") == ["hello,", "world", "ok"]
    assert tokenize_text("   ") == []


# ------------------------------------------------------------------ diff

def test_an_unchanged_transcript_keeps_every_word():
    words = make_words("the goal of contrastive representation learning")
    tokens = diff_transcript(words, "the goal of contrastive representation learning")
    assert len(tokens) == len(words)
    assert all("ref" in t for t in tokens)
    assert [t["ref"] for t in tokens] == [w["id"] for w in words]


def test_deleting_a_middle_phrase_keeps_the_surrounding_words():
    words = make_words("the goal of contrastive representation learning is to learn")
    tokens = diff_transcript(words, "the goal is to learn")
    assert describe(tokens, words) == "the goal is to learn"
    assert all("ref" in t for t in tokens), "a pure deletion needs no synthesis"


def test_a_head_deletion_is_not_misread_as_a_replacement():
    words = make_words("four score and seven years ago our fathers")
    tokens = diff_transcript(words, "our fathers")
    assert [t["ref"] for t in tokens] == ["w6", "w7"]


def test_inserted_words_become_synthesis_tokens():
    words = make_words("four score and seven years ago our fathers")
    tokens = diff_transcript(words, "246 years ago our fathers")
    assert describe(tokens, words) == "+[246] years ago our fathers"
    inserts = [t for t in tokens if "insert" in t]
    assert len(inserts) == 1 and inserts[0]["insert"] == "246"


def test_consecutive_inserted_words_merge_into_one_token():
    words = make_words("hello world")
    tokens = diff_transcript(words, "hello brave new world")
    assert describe(tokens, words) == "hello +[brave new] world"


def test_punctuation_and_capitalisation_do_not_force_resynthesis():
    words = make_words("all men are created equal")
    tokens = diff_transcript(words, "All men, are created equal!")
    assert all("ref" in t for t in tokens)
    assert len(tokens) == 5


def test_a_word_replaced_in_place_yields_one_insertion():
    words = make_words("the quick brown fox jumps")
    tokens = diff_transcript(words, "the quick red fox jumps")
    assert describe(tokens, words) == "the quick +[red] fox jumps"


def test_repeated_words_align_to_the_nearest_surviving_occurrence():
    words = make_words("the cat sat on the mat and the cat left")
    tokens = diff_transcript(words, "the cat sat on the mat and the cat left")
    assert [t["ref"] for t in tokens] == [w["id"] for w in words]


def test_emptying_the_transcript_synthesises_nothing():
    words = make_words("one two three")
    assert diff_transcript(words, "") == []
    assert diff_transcript(words, "   ") == []


def test_typing_into_an_empty_transcript_produces_one_insertion():
    assert diff_transcript([], "brand new sentence") == [{"insert": "brand new sentence"}]


def test_align_tokens_is_exact_on_a_large_document_with_a_small_edit():
    # 4000 tokens exercises the patience-anchoring path, not plain Myers.
    a = [f"tok{i}" for i in range(4000)]
    b = a[:2000] + a[2003:]
    ops = align_tokens(a, b)
    assert sum(1 for o in ops if o.type == "equal") == 3997
    assert sum(1 for o in ops if o.type == "delete") == 3
    assert sum(1 for o in ops if o.type == "insert") == 0


def test_align_tokens_handles_a_large_document_with_heavy_repetition():
    # Almost no unique anchors: the fallback must still produce a valid script.
    a = ["a" if i % 3 == 0 else "b" for i in range(1500)]
    b = a[:1400]
    ops = align_tokens(a, b)
    equals = sum(1 for o in ops if o.type == "equal")
    assert equals + sum(1 for o in ops if o.type == "delete") == len(a)
    assert equals + sum(1 for o in ops if o.type == "insert") == len(b)


@pytest.mark.parametrize("left,right", [
    ("a b c d e", "a c e"),
    ("a b c", "x y z"),
    ("", "a b"),
    ("a b", ""),
    ("same words here", "same words here"),
    ("one two two three", "one two three"),
])
def test_edit_scripts_always_reconstruct_the_target(left, right):
    a = left.split(" ") if left else []
    b = right.split(" ") if right else []
    ops = align_tokens(a, b)

    rebuilt = [
        a[o.a_index] if o.type == "equal" else b[o.b_index]
        for o in ops if o.type != "delete"
    ]
    assert rebuilt == b

    for op in ops:
        if op.type == "equal":
            assert a[op.a_index] == b[op.b_index]

    # Indices must advance strictly on both sides.
    ai = bi = -1
    for op in ops:
        if op.type == "equal":
            assert op.a_index > ai and op.b_index > bi
            ai, bi = op.a_index, op.b_index
        elif op.type == "delete":
            assert op.a_index > ai
            ai = op.a_index
        else:
            assert op.b_index > bi
            bi = op.b_index


# ------------------------------------------------------------------- edl

def test_an_unedited_document_yields_one_uninterrupted_copy():
    words = make_words(SENTENCE)
    edl = build_edl(words, identity_tokens(words), duration=5.0)
    assert len(edl.segments) == 1
    assert isinstance(edl.segments[0], CopySegment)
    assert edl.stats.cuts == 0
    assert edl.stats.deleted_words == 0
    assert edl.stats.inserted_words == 0


def test_cuts_land_in_the_gap_between_words():
    words = make_words(SENTENCE)
    tokens = diff_transcript(words, SENTENCE)
    segment = build_edl(words, tokens, duration=5.0).segments[0]
    assert segment.start >= 0
    assert segment.start <= words[0]["start"]
    assert segment.end >= words[-1]["end"]


def test_deleting_a_phrase_removes_that_span_and_creates_one_seam():
    words = make_words(SENTENCE)
    # Drop "score and seven" (w1..w3).
    tokens = diff_transcript(words, "four years ago our fathers brought forth")
    edl = build_edl(words, tokens, duration=5.0)

    assert len(edl.segments) == 2
    assert all(isinstance(s, CopySegment) for s in edl.segments)
    assert edl.stats.deleted_words == 3
    assert edl.stats.kept_words == 7
    assert edl.stats.cuts == 1

    # Ends after "four" (0.4) and before "score" (0.5).
    assert 0.4 < edl.segments[0].end <= 0.5
    # Resumes in the gap before "years" (2.0).
    assert 1.9 < edl.segments[1].start <= 2.0

    kept = sum(s.end - s.start for s in edl.segments)
    assert kept < 5.0 - 1.4


def test_padding_never_drags_a_long_silence_in_with_a_deleted_word():
    words = [
        {"id": "a", "text": "hello", "start": 0.0, "end": 0.5},
        {"id": "b", "text": "um", "start": 3.0, "end": 3.3},   # 2.5 s of silence before
        {"id": "c", "text": "world", "start": 6.0, "end": 6.5},
    ]
    edl = build_edl(words, [{"ref": "a"}, {"ref": "c"}], duration=7.0, max_pad=0.12)
    assert len(edl.segments) == 2
    assert edl.segments[0].end <= 0.5 + 0.12 + 1e-9
    assert edl.segments[1].start >= 6.0 - 0.12 - 1e-9


def test_adjacent_kept_words_never_produce_a_seam():
    words = make_words(SENTENCE)
    edl = build_edl(words, [{"ref": w["id"]} for w in words], duration=5.0)
    assert len(edl.segments) == 1
    assert edl.stats.cuts == 0


def test_an_insertion_carries_both_neighbours():
    words = make_words(SENTENCE)
    tokens = diff_transcript(words, "246 years ago our fathers brought forth")
    edl = build_edl(words, tokens, duration=5.0)

    assert edl.stats.inserted_words == 1
    synth = next(s for s in edl.segments if isinstance(s, SynthSegment))
    assert synth.text == "246"
    assert synth.context_after == "years"
    assert edl.segments[0] is synth
    assert synth.context_before is None
    assert synth.lead_gap == 0, "no leading silence at the very start"


def test_a_mid_sentence_insertion_sees_the_words_on_both_sides():
    words = make_words(SENTENCE)
    tokens = diff_transcript(words, "four score and seven long years ago our fathers brought forth")
    synth = next(s for s in build_edl(words, tokens, duration=5.0).segments
                 if isinstance(s, SynthSegment))
    assert synth.text == "long"
    assert synth.context_before == "seven"
    assert synth.context_after == "years"
    assert synth.estimated_duration > 0


def test_estimated_duration_tracks_the_speakers_own_rate():
    words = make_words(SENTENCE)
    rate = speech_rate([__import__("edl").Word.from_any(w) for w in words])
    assert 0.3 < rate < 0.7

    tokens = diff_transcript(words, f"{SENTENCE} and a few extra words here")
    edl = build_edl(words, tokens, duration=5.0)
    synth = next(s for s in edl.segments if isinstance(s, SynthSegment))
    assert edl.stats.inserted_words == 6
    assert abs(synth.estimated_duration - 6 * rate) < 0.5
    assert edl.stats.estimated_duration > edl.stats.source_duration


def test_deleting_words_shortens_the_estimate():
    words = make_words(SENTENCE)
    edl = build_edl(words, diff_transcript(words, "four years ago"), duration=5.0)
    assert edl.stats.estimated_duration < edl.stats.source_duration


def test_stale_word_ids_are_ignored_rather_than_raising():
    words = make_words("one two three")
    edl = build_edl(words, [{"ref": "w0"}, {"ref": "gone"}, {"ref": "w2"}], duration=2.0)
    assert edl.stats.kept_words == 2
    assert len(edl.segments) == 2


def test_blank_insertions_are_dropped():
    words = make_words("one two")
    edl = build_edl(words, [{"ref": "w0"}, {"insert": "   "}, {"ref": "w1"}], duration=1.5)
    assert edl.stats.inserted_words == 0
    assert len(edl.segments) == 1, "the two words rejoin as one contiguous copy"


def test_an_all_synthetic_document_produces_no_copy_segments():
    words = make_words("one two")
    edl = build_edl(words, [{"insert": "completely new line"}], duration=1.5)
    assert len(edl.segments) == 1
    assert isinstance(edl.segments[0], SynthSegment)
    assert edl.stats.kept_words == 0
    assert edl.stats.deleted_words == 2
    assert edl.stats.inserted_words == 3


def test_segments_are_ordered_and_never_overlap():
    words = make_words(SENTENCE)
    tokens = diff_transcript(words, "four seven ago fathers forth")
    previous_end = float("-inf")
    for segment in build_edl(words, tokens, duration=5.0).segments:
        if not isinstance(segment, CopySegment):
            continue
        assert segment.start < segment.end
        assert segment.start >= previous_end - 1e-9
        previous_end = segment.end


def test_reordering_is_expressed_as_copies_not_resynthesis():
    words = make_words("alpha bravo charlie")
    tokens = diff_transcript(words, "charlie alpha bravo")
    segments = build_edl(words, tokens, duration=1.5).segments
    copied = [wid for s in segments if isinstance(s, CopySegment) for wid in s.word_ids]
    assert len(copied) >= 2, "the alpha/bravo run is reused verbatim"


def test_is_contiguous_only_accepts_genuinely_adjacent_copies():
    a = CopySegment(0, 1, ["w0"], 0, 0)
    b = CopySegment(1, 2, ["w1"], 1, 1)
    c = CopySegment(5, 6, ["w9"], 9, 9)
    assert is_contiguous(a, b) is True
    assert is_contiguous(a, c) is False
    assert is_contiguous(a, SynthSegment("x", None, None, 0, 0, 0.1)) is False


# -------------------------------------------------------------- snapping

def test_snap_to_quiet_moves_a_cut_to_the_local_minimum():
    rms = [0.5] * 200
    rms[100] = 0.01
    snapped = snap_to_quiet(1.03, 0.9, 1.1, __import__("edl").Envelope(100, rms), 0.06)
    assert abs(snapped - 1.0) < 0.011


def test_snap_to_quiet_respects_bounds_and_missing_envelopes():
    rms = [0.5] * 200
    rms[0] = 0.0  # a quiet point far outside the allowed range
    envelope = __import__("edl").Envelope(100, rms)
    snapped = snap_to_quiet(1.0, 0.98, 1.02, envelope, 0.5)
    assert 0.98 <= snapped <= 1.02
    assert snap_to_quiet(1.0, 0.9, 1.1, None) == 1.0


def test_a_flat_envelope_does_not_move_cuts_pointlessly():
    words = make_words(SENTENCE)
    envelope = {"fps": 100, "rms": [0.4] * 600}
    tokens = diff_transcript(words, "four years ago our fathers brought forth")
    segments = build_edl(words, tokens, duration=5.0, envelope=envelope).segments
    assert words[0]["end"] < segments[0].end <= words[1]["start"]
