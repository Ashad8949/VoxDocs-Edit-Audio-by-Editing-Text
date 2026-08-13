/**
 * The editable document.
 *
 * A transcript editor cannot be a plain text box. The moment text becomes an
 * opaque string, the link between a word and the samples it was spoken in is
 * lost, and it has to be guessed back by alignment. So the document is a list
 * of blocks that each keep their identity:
 *
 *   - `word`  — an original transcript word, holding the id the EDL needs.
 *               Deleting it sets a flag rather than removing it, which makes
 *               deletion reversible and lets the UI show what was cut.
 *   - `insert` — text the user typed, which has no audio yet and will be sent
 *               to the synthesiser.
 *
 * Every function here is pure: given a document it returns a new one. That
 * keeps undo trivial (keep the old reference) and makes the whole editing model
 * testable without a browser.
 */

let insertCounter = 0;

/** @typedef {{ kind: 'word', id: string, text: string, deleted: boolean, start: number, end: number, segment: number }} WordBlock */
/** @typedef {{ kind: 'insert', id: string, text: string }} InsertBlock */
/** @typedef {WordBlock | InsertBlock} Block */

/**
 * Build the initial document from a transcript.
 * @param {Array<{ id: string, text: string, start: number, end: number }>} words
 * @param {Array<{ first_word: number, last_word: number }>} [segments]
 * @returns {Block[]}
 */
export function docFromWords(words, segments = []) {
  const segmentOf = new Map();
  segments.forEach((segment, index) => {
    for (let i = segment.first_word; i <= segment.last_word; i++) segmentOf.set(i, index);
  });

  return words.map((word, i) => ({
    kind: 'word',
    id: word.id,
    text: word.text,
    deleted: false,
    start: word.start,
    end: word.end,
    segment: segmentOf.get(i) ?? 0,
  }));
}

/**
 * The edit-token list the API expects: surviving words by reference, typed
 * text as insertions.
 * @param {Block[]} doc
 */
export function toTokens(doc) {
  const tokens = [];
  for (const block of doc) {
    if (block.kind === 'word') {
      if (!block.deleted) tokens.push({ ref: block.id });
    } else if (block.text.trim()) {
      tokens.push({ insert: block.text.trim() });
    }
  }
  return tokens;
}

/** The text as it would be spoken after the edit. */
export function docText(doc) {
  return doc
    .filter((b) => (b.kind === 'word' ? !b.deleted : b.text.trim()))
    .map((b) => b.text.trim())
    .join(' ');
}

/**
 * Mark a range of blocks deleted. Insert blocks in the range are removed
 * outright, since there is nothing to restore them to.
 * @param {Block[]} doc
 * @param {number} from inclusive block index
 * @param {number} to inclusive block index
 */
export function deleteRange(doc, from, to) {
  const lo = Math.max(0, Math.min(from, to));
  const hi = Math.min(doc.length - 1, Math.max(from, to));
  if (hi < lo) return doc;

  const next = [];
  for (let i = 0; i < doc.length; i++) {
    const block = doc[i];
    if (i < lo || i > hi) {
      next.push(block);
      continue;
    }
    if (block.kind === 'word') next.push({ ...block, deleted: true });
    // insert blocks inside the range simply disappear
  }
  return next;
}

/**
 * Restore previously deleted words.
 * @param {Block[]} doc
 * @param {number} from
 * @param {number} to
 */
export function restoreRange(doc, from, to) {
  const lo = Math.max(0, Math.min(from, to));
  const hi = Math.min(doc.length - 1, Math.max(from, to));
  return doc.map((block, i) =>
    i >= lo && i <= hi && block.kind === 'word' && block.deleted
      ? { ...block, deleted: false }
      : block
  );
}

/**
 * Insert typed text at a gap. `position` is the number of blocks before the
 * insertion point, so 0 puts text at the very start and `doc.length` at the end.
 * Adjacent insertions merge, which keeps the synthesiser working on whole
 * phrases instead of isolated words.
 * @param {Block[]} doc
 * @param {number} position
 * @param {string} text
 */
export function insertAt(doc, position, text) {
  const trimmed = text.trim();
  if (!trimmed) return doc;
  const at = Math.max(0, Math.min(doc.length, position));

  const before = doc[at - 1];
  if (before && before.kind === 'insert') {
    const merged = [...doc];
    merged[at - 1] = { ...before, text: `${before.text} ${trimmed}`.trim() };
    return merged;
  }
  const after = doc[at];
  if (after && after.kind === 'insert') {
    const merged = [...doc];
    merged[at] = { ...after, text: `${trimmed} ${after.text}`.trim() };
    return merged;
  }

  insertCounter += 1;
  const block = { kind: 'insert', id: `i${insertCounter}`, text: trimmed };
  return [...doc.slice(0, at), block, ...doc.slice(at)];
}

/**
 * Replace the text of an existing insert block, dropping it when emptied.
 * @param {Block[]} doc
 * @param {string} id
 * @param {string} text
 */
export function updateInsert(doc, id, text) {
  const trimmed = text.trim();
  if (!trimmed) return doc.filter((b) => b.id !== id);
  return doc.map((b) => (b.id === id ? { ...b, text: trimmed } : b));
}

/**
 * Replace a run of words with typed text: the words are struck out and the
 * text takes their place. This is what typing over a selection does.
 * @param {Block[]} doc
 * @param {number} from
 * @param {number} to
 * @param {string} text
 */
export function replaceRange(doc, from, to, text) {
  const lo = Math.max(0, Math.min(from, to));
  const hi = Math.min(doc.length - 1, Math.max(from, to));
  const deleted = deleteRange(doc, lo, hi);

  // Land the insertion immediately after the struck-out run. Insert blocks in
  // the range were removed, so recompute where that is.
  let removed = 0;
  for (let i = lo; i <= hi; i++) if (doc[i].kind === 'insert') removed += 1;
  return insertAt(deleted, hi + 1 - removed, text);
}

/** Summary the UI shows without asking the server. */
export function localStats(doc) {
  let kept = 0;
  let deleted = 0;
  let inserted = 0;
  let keptSeconds = 0;
  for (const block of doc) {
    if (block.kind === 'word') {
      if (block.deleted) deleted += 1;
      else {
        kept += 1;
        keptSeconds += Math.max(0, block.end - block.start);
      }
    } else {
      inserted += block.text.trim().split(/\s+/).filter(Boolean).length;
    }
  }
  return { kept, deleted, inserted, keptSeconds };
}

/**
 * Index of the word block covering a playback time, or -1.
 * @param {Block[]} doc
 * @param {number} time
 */
export function wordIndexAtTime(doc, time) {
  for (let i = 0; i < doc.length; i++) {
    const block = doc[i];
    if (block.kind !== 'word') continue;
    if (time >= block.start && time < block.end) return i;
  }
  return -1;
}

/** Contiguous deleted spans, for painting cuts onto the waveform. */
export function deletedSpans(doc) {
  const spans = [];
  let current = null;
  for (const block of doc) {
    if (block.kind !== 'word') continue;
    if (block.deleted) {
      if (current && Math.abs(block.start - current.end) < 0.35) current.end = block.end;
      else {
        current = { start: block.start, end: block.end };
        spans.push(current);
      }
    } else {
      current = null;
    }
  }
  return spans;
}

/** Reset the id counter; used by tests to keep ids predictable. */
export function __resetInsertIds() {
  insertCounter = 0;
}
