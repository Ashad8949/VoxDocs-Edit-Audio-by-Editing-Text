"""Token normalisation.

Alignment between the original transcript and the user's edited text has to
ignore differences that carry no audio consequence: case, punctuation, smart
quotes, and the leading space that ASR engines attach to each word. Two tokens
that normalise to the same key are considered the same spoken word, so the
original audio for it can be kept verbatim.

This must agree exactly with ``voxdocs.synth.normalize_token`` in the model
server, or a word the API believes is in the voice bank will not be found there.
"""

from __future__ import annotations

import unicodedata

#: Curly and straight apostrophes, which live inside words rather than around them.
_APOSTROPHES = "‘’ʼ'"


def normalize_token(raw: str) -> str:
    """Reduce a surface word to its comparison key.

    Returns ``''`` for punctuation-only input, which callers treat as "not a
    word" rather than as an empty word.
    """
    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", str(raw)).lower()
    for mark in _APOSTROPHES:
        text = text.replace(mark, "")
    return "".join(ch for ch in text if ch.isalnum())


def tokenize_text(text: str) -> list[str]:
    """Split free text into surface words.

    The original spelling is preserved so inserted text reaches the synthesiser
    exactly as the user typed it; only the comparison key is normalised.
    """
    if not text:
        return []
    return [word for word in str(text).split() if normalize_token(word)]
