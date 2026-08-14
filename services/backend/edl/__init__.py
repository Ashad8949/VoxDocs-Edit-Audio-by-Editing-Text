"""Transcript alignment and edit-decision-list construction.

Deliberately free of Django imports so it can be unit-tested, reused from a
management command, or lifted into another service without dragging a web
framework along.
"""

from .builder import (
    CopySegment,
    Edl,
    EdlStats,
    Envelope,
    SynthSegment,
    Word,
    build_edl,
    identity_tokens,
    is_contiguous,
    median_gap,
    snap_to_quiet,
    speech_rate,
)
from .diff import EditOp, align_tokens, diff_transcript
from .tokens import normalize_token, tokenize_text

__all__ = [
    "CopySegment", "Edl", "EdlStats", "Envelope", "SynthSegment", "Word",
    "build_edl", "identity_tokens", "is_contiguous", "median_gap",
    "snap_to_quiet", "speech_rate",
    "EditOp", "align_tokens", "diff_transcript",
    "normalize_token", "tokenize_text",
]
