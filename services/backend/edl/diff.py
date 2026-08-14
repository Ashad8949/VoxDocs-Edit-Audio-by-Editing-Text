"""Sequence alignment between the original transcript and the edited text.

The editor normally knows exactly which original words survived an edit, because
it renders one span per word and tracks them by id. But text can also arrive as
an opaque string — a paste, an external API caller, an undo that replaced a whole
paragraph. Then the mapping has to be recovered, and the quality of that recovery
decides whether an edit re-uses the speaker's real audio or needlessly
re-synthesises it.

Strategy:
  - Myers O(ND) diff for ordinary edits, where D (the number of differences) is
    small. Exact and fast when a user tweaks a few words.
  - Patience-style anchoring for large inputs, where a full Myers trace would
    cost too much memory. Words occurring exactly once on both sides are
    reliable anchors; the longest increasing run of them is kept and the gaps
    between are diffed recursively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from .tokens import normalize_token, tokenize_text

#: Beyond this many total tokens, anchor before diffing.
ANCHOR_THRESHOLD = 2000
#: Hard ceiling on the Myers edit distance worth tracing.
MAX_EDIT_DISTANCE = 3000


@dataclass(frozen=True)
class EditOp:
    """One step of an edit script.

    ``a_index`` is set for ``equal`` and ``delete``; ``b_index`` for ``equal``
    and ``insert``.
    """

    type: Literal["equal", "delete", "insert"]
    a_index: int = -1
    b_index: int = -1


def _myers_diff(a: Sequence[str], b: Sequence[str], max_d: int) -> list[EditOp] | None:
    """Myers greedy diff with a recorded trace, capped at ``max_d`` differences.

    Returns ``None`` when the edit distance exceeds the cap, so the caller can
    fall back rather than spend unbounded memory.
    """
    n, m = len(a), len(b)
    maximum = min(max_d, n + m)
    size = 2 * maximum + 1
    offset = maximum

    v = [0] * size
    trace: list[list[int]] = []

    for d in range(maximum + 1):
        trace.append(v[:])
        for k in range(-d, d + 1, 2):
            if k == -d or (k != d and v[offset + k - 1] < v[offset + k + 1]):
                x = v[offset + k + 1]      # move down: an insertion from b
            else:
                x = v[offset + k - 1] + 1  # move right: a deletion from a
            y = x - k
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1
            v[offset + k] = x
            if x >= n and y >= m:
                return _backtrack(trace, n, m, d, offset, size)
    return None


def _backtrack(trace: list[list[int]], n: int, m: int, final_d: int,
               offset: int, size: int) -> list[EditOp]:
    """Walk the recorded trace backwards to recover the edit script."""
    ops: list[EditOp] = []
    x, y = n, m

    for d in range(final_d, 0, -1):
        v = trace[d]
        k = x - y
        if k == -d or (k != d and v[offset + k - 1] < v[offset + k + 1]):
            prev_k = k + 1
        else:
            prev_k = k - 1
        prev_x = v[offset + prev_k] if 0 <= offset + prev_k < size else 0
        prev_y = prev_x - prev_k

        # The diagonal run preceding this move is a block of equal tokens.
        while x > prev_x and y > prev_y:
            x -= 1
            y -= 1
            ops.append(EditOp("equal", x, y))

        if x == prev_x:
            y -= 1
            ops.append(EditOp("insert", b_index=y))
        else:
            x -= 1
            ops.append(EditOp("delete", a_index=x))

    # The leading diagonal, before any edit was made.
    while x > 0 and y > 0:
        x -= 1
        y -= 1
        ops.append(EditOp("equal", x, y))
    while y > 0:
        y -= 1
        ops.append(EditOp("insert", b_index=y))
    while x > 0:
        x -= 1
        ops.append(EditOp("delete", a_index=x))

    ops.reverse()
    return ops


def _unique_common_tokens(a: Sequence[str], b: Sequence[str]) -> list[tuple[int, int]]:
    """Tokens occurring exactly once in both sequences make unambiguous anchors."""
    a_count: dict[str, int] = {}
    a_pos: dict[str, int] = {}
    for i, token in enumerate(a):
        a_count[token] = a_count.get(token, 0) + 1
        a_pos[token] = i

    b_count: dict[str, int] = {}
    b_pos: dict[str, int] = {}
    for j, token in enumerate(b):
        b_count[token] = b_count.get(token, 0) + 1
        b_pos[token] = j

    anchors = [
        (a_pos[token], b_pos[token])
        for token, count in a_count.items()
        if count == 1 and b_count.get(token) == 1
    ]
    anchors.sort()
    return anchors


def _longest_increasing_anchors(anchors: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Longest increasing subsequence over anchor b-indices.

    Keeps the anchors mutually consistent — no crossings — and as numerous as
    possible, which is what makes the recursion into the gaps well-founded.
    """
    if not anchors:
        return []

    import bisect

    tails: list[int] = []          # indices into `anchors`
    tail_values: list[int] = []    # their b-indices, kept sorted
    previous = [-1] * len(anchors)

    for i, (_, b_index) in enumerate(anchors):
        position = bisect.bisect_left(tail_values, b_index)
        if position > 0:
            previous[i] = tails[position - 1]
        if position == len(tails):
            tails.append(i)
            tail_values.append(b_index)
        else:
            tails[position] = i
            tail_values[position] = b_index

    result = []
    cursor = tails[-1]
    while cursor != -1:
        result.append(anchors[cursor])
        cursor = previous[cursor]
    result.reverse()
    return result


def _diff_range(a: Sequence[str], b: Sequence[str], a_start: int, a_end: int,
                b_start: int, b_end: int) -> list[EditOp]:
    """Diff a slice of both sequences, offsetting the emitted indices."""
    a_slice = a[a_start:a_end]
    b_slice = b[b_start:b_end]
    if not a_slice and not b_slice:
        return []

    budget = min(MAX_EDIT_DISTANCE, len(a_slice) + len(b_slice))
    ops = _myers_diff(a_slice, b_slice, budget)
    if ops is not None:
        return [
            EditOp(
                op.type,
                op.a_index + a_start if op.a_index >= 0 else -1,
                op.b_index + b_start if op.b_index >= 0 else -1,
            )
            for op in ops
        ]

    # The two sides share nothing tractable: replace the range wholesale. Still
    # correct, it just re-synthesises more than strictly necessary.
    fallback = [EditOp("delete", a_index=i) for i in range(a_start, a_end)]
    fallback += [EditOp("insert", b_index=j) for j in range(b_start, b_end)]
    return fallback


def align_tokens(a: Sequence[str], b: Sequence[str]) -> list[EditOp]:
    """Align two normalised token sequences."""
    if len(a) + len(b) <= ANCHOR_THRESHOLD:
        return _diff_range(a, b, 0, len(a), 0, len(b))

    anchors = _longest_increasing_anchors(_unique_common_tokens(a, b))
    if not anchors:
        return _diff_range(a, b, 0, len(a), 0, len(b))

    ops: list[EditOp] = []
    a_cursor = b_cursor = 0
    for a_index, b_index in anchors:
        ops.extend(_diff_range(a, b, a_cursor, a_index, b_cursor, b_index))
        ops.append(EditOp("equal", a_index, b_index))
        a_cursor = a_index + 1
        b_cursor = b_index + 1
    ops.extend(_diff_range(a, b, a_cursor, len(a), b_cursor, len(b)))
    return ops


def diff_transcript(words: Iterable, edited_text: str) -> list[dict]:
    """Recover an edit-token list from original words and an edited transcript.

    ``words`` may be model instances or plain dicts; only ``id`` and ``text`` are
    read. Returns ``[{"ref": word_id} | {"insert": text}, ...]``.
    """
    word_list = list(words)
    surface_b = tokenize_text(edited_text)

    a = [normalize_token(_field(w, "text")) for w in word_list]
    b = [normalize_token(token) for token in surface_b]

    tokens: list[dict] = []
    pending: list[str] = []

    def flush() -> None:
        nonlocal pending
        if pending:
            tokens.append({"insert": " ".join(pending)})
            pending = []

    for op in align_tokens(a, b):
        if op.type == "equal":
            flush()
            tokens.append({"ref": str(_field(word_list[op.a_index], "id"))})
        elif op.type == "insert":
            pending.append(surface_b[op.b_index])
        # Deletions contribute nothing to the output.
    flush()
    return tokens


def _field(item, name):
    """Read an attribute from a model instance or a key from a dict."""
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name)
