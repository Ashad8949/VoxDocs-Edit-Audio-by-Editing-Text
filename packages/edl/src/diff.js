/**
 * Sequence alignment between the original transcript and the edited text.
 *
 * The editor UI normally knows exactly which original words survived an edit,
 * because it renders one span per word and tracks them by id. But text can also
 * arrive as an opaque string — a paste, an external API caller, an undo that
 * replaced a whole paragraph. In that case we have to recover the mapping, and
 * the quality of that recovery decides whether an edit re-uses the speaker's
 * real audio or needlessly re-synthesises it.
 *
 * Strategy:
 *   - Myers O(ND) diff for ordinary edits, where D (the number of differences)
 *     is small. This is exact and fast when a user tweaks a few words.
 *   - Patience-style anchoring for large inputs, where a full Myers trace would
 *     cost too much memory. Words that occur exactly once on both sides are
 *     reliable anchors; we take the longest increasing run of them and recurse
 *     into the gaps.
 */

import { normalizeToken, tokenizeText } from './tokens.js';

/** Beyond this many total tokens we anchor before diffing. */
const ANCHOR_THRESHOLD = 2000;
/** Hard ceiling on the Myers edit distance we are willing to trace. */
const MAX_EDIT_DISTANCE = 3000;

/**
 * @typedef {{ type: 'equal', aIndex: number, bIndex: number }
 *         | { type: 'delete', aIndex: number }
 *         | { type: 'insert', bIndex: number }} EditOp
 */

/**
 * Myers greedy diff with a recorded trace, restricted to `maxD` differences.
 * @param {string[]} a normalised original tokens
 * @param {string[]} b normalised edited tokens
 * @param {number} maxD
 * @returns {EditOp[] | null} null when the edit distance exceeds `maxD`
 */
function myersDiff(a, b, maxD) {
  const n = a.length;
  const m = b.length;
  const max = Math.min(maxD, n + m);
  const size = 2 * max + 1;
  const offset = max;
  let v = new Int32Array(size);
  const trace = [];

  for (let d = 0; d <= max; d++) {
    trace.push(v.slice());
    for (let k = -d; k <= d; k += 2) {
      let x;
      if (k === -d || (k !== d && v[offset + k - 1] < v[offset + k + 1])) {
        x = v[offset + k + 1]; // move down: an insertion from b
      } else {
        x = v[offset + k - 1] + 1; // move right: a deletion from a
      }
      let y = x - k;
      while (x < n && y < m && a[x] === b[y]) {
        x++;
        y++;
      }
      if (offset + k >= 0 && offset + k < size) v[offset + k] = x;
      if (x >= n && y >= m) {
        return backtrack(trace, a, b, d, offset, size);
      }
    }
  }
  return null;
}

/**
 * Walk the recorded Myers trace backwards to recover the edit script.
 * @param {Int32Array[]} trace
 * @param {string[]} a
 * @param {string[]} b
 * @param {number} finalD
 * @param {number} offset
 * @param {number} size
 * @returns {EditOp[]}
 */
function backtrack(trace, a, b, finalD, offset, size) {
  /** @type {EditOp[]} */
  const ops = [];
  let x = a.length;
  let y = b.length;

  for (let d = finalD; d > 0; d--) {
    const v = trace[d];
    const k = x - y;
    let prevK;
    if (k === -d || (k !== d && v[offset + k - 1] < v[offset + k + 1])) {
      prevK = k + 1;
    } else {
      prevK = k - 1;
    }
    const prevX = offset + prevK >= 0 && offset + prevK < size ? v[offset + prevK] : 0;
    const prevY = prevX - prevK;

    // The diagonal run that preceded this move is a block of equal tokens.
    while (x > prevX && y > prevY) {
      x--;
      y--;
      ops.push({ type: 'equal', aIndex: x, bIndex: y });
    }
    if (d > 0) {
      if (x === prevX) {
        y--;
        ops.push({ type: 'insert', bIndex: y });
      } else {
        x--;
        ops.push({ type: 'delete', aIndex: x });
      }
    }
  }
  // Leading diagonal before any edit was made.
  while (x > 0 && y > 0) {
    x--;
    y--;
    ops.push({ type: 'equal', aIndex: x, bIndex: y });
  }
  while (y > 0) {
    y--;
    ops.push({ type: 'insert', bIndex: y });
  }
  while (x > 0) {
    x--;
    ops.push({ type: 'delete', aIndex: x });
  }

  ops.reverse();
  return ops;
}

/**
 * Tokens occurring exactly once in both sequences make unambiguous anchors.
 * @param {string[]} a
 * @param {string[]} b
 * @returns {Array<{ aIndex: number, bIndex: number }>} anchors in a-order
 */
function uniqueCommonTokens(a, b) {
  /** @type {Map<string, number>} */
  const aCount = new Map();
  /** @type {Map<string, number>} */
  const aPos = new Map();
  for (let i = 0; i < a.length; i++) {
    aCount.set(a[i], (aCount.get(a[i]) ?? 0) + 1);
    aPos.set(a[i], i);
  }
  /** @type {Map<string, number>} */
  const bCount = new Map();
  /** @type {Map<string, number>} */
  const bPos = new Map();
  for (let j = 0; j < b.length; j++) {
    bCount.set(b[j], (bCount.get(b[j]) ?? 0) + 1);
    bPos.set(b[j], j);
  }
  const anchors = [];
  for (const [tok, count] of aCount) {
    if (count !== 1) continue;
    if (bCount.get(tok) !== 1) continue;
    anchors.push({ aIndex: /** @type {number} */ (aPos.get(tok)), bIndex: /** @type {number} */ (bPos.get(tok)) });
  }
  anchors.sort((x, y) => x.aIndex - y.aIndex);
  return anchors;
}

/**
 * Longest increasing subsequence over anchor b-indices, so the anchors we keep
 * are mutually consistent (no crossing) and as numerous as possible.
 * @param {Array<{ aIndex: number, bIndex: number }>} anchors sorted by aIndex
 */
function longestIncreasingAnchors(anchors) {
  if (anchors.length === 0) return [];
  /** @type {number[]} tails[len] = index into anchors of the smallest tail */
  const tails = [];
  const prev = new Int32Array(anchors.length).fill(-1);

  for (let i = 0; i < anchors.length; i++) {
    let lo = 0;
    let hi = tails.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (anchors[tails[mid]].bIndex < anchors[i].bIndex) lo = mid + 1;
      else hi = mid;
    }
    if (lo > 0) prev[i] = tails[lo - 1];
    if (lo === tails.length) tails.push(i);
    else tails[lo] = i;
  }

  const result = [];
  let cursor = tails[tails.length - 1];
  while (cursor !== -1) {
    result.push(anchors[cursor]);
    cursor = prev[cursor];
  }
  result.reverse();
  return result;
}

/**
 * Diff a slice of both sequences, offsetting the emitted indices.
 * @param {string[]} a
 * @param {string[]} b
 * @param {number} aStart
 * @param {number} aEnd
 * @param {number} bStart
 * @param {number} bEnd
 * @returns {EditOp[]}
 */
function diffRange(a, b, aStart, aEnd, bStart, bEnd) {
  const aSlice = a.slice(aStart, aEnd);
  const bSlice = b.slice(bStart, bEnd);
  if (aSlice.length === 0 && bSlice.length === 0) return [];

  const budget = Math.min(MAX_EDIT_DISTANCE, aSlice.length + bSlice.length);
  const ops = myersDiff(aSlice, bSlice, budget);
  if (ops) {
    return ops.map((op) => {
      if (op.type === 'equal') {
        return { type: 'equal', aIndex: op.aIndex + aStart, bIndex: op.bIndex + bStart };
      }
      if (op.type === 'delete') return { type: 'delete', aIndex: op.aIndex + aStart };
      return { type: 'insert', bIndex: op.bIndex + bStart };
    });
  }

  // The two sides share nothing tractable: replace the range wholesale. The
  // result is still correct, it just re-synthesises more than strictly needed.
  /** @type {EditOp[]} */
  const fallback = [];
  for (let i = aStart; i < aEnd; i++) fallback.push({ type: 'delete', aIndex: i });
  for (let j = bStart; j < bEnd; j++) fallback.push({ type: 'insert', bIndex: j });
  return fallback;
}

/**
 * Align two normalised token sequences.
 * @param {string[]} a
 * @param {string[]} b
 * @returns {EditOp[]}
 */
export function alignTokens(a, b) {
  if (a.length + b.length <= ANCHOR_THRESHOLD) {
    return diffRange(a, b, 0, a.length, 0, b.length);
  }

  const anchors = longestIncreasingAnchors(uniqueCommonTokens(a, b));
  if (anchors.length === 0) {
    return diffRange(a, b, 0, a.length, 0, b.length);
  }

  /** @type {EditOp[]} */
  const ops = [];
  let aCursor = 0;
  let bCursor = 0;
  for (const anchor of anchors) {
    ops.push(...diffRange(a, b, aCursor, anchor.aIndex, bCursor, anchor.bIndex));
    ops.push({ type: 'equal', aIndex: anchor.aIndex, bIndex: anchor.bIndex });
    aCursor = anchor.aIndex + 1;
    bCursor = anchor.bIndex + 1;
  }
  ops.push(...diffRange(a, b, aCursor, a.length, bCursor, b.length));
  return ops;
}

/**
 * Recover an edit-token list from the original word timings and a plain-text
 * edit of the transcript.
 *
 * @param {Array<{ id: string, text: string }>} words original transcript words
 * @param {string} editedText
 * @returns {Array<{ ref: string } | { insert: string }>}
 */
export function diffTranscript(words, editedText) {
  const surfaceB = tokenizeText(editedText);
  const a = words.map((w) => normalizeToken(w.text));
  const b = surfaceB.map((t) => normalizeToken(t));

  const ops = alignTokens(a, b);

  /** @type {Array<{ ref: string } | { insert: string }>} */
  const tokens = [];
  /** @type {string[]} */
  let pendingInsert = [];

  const flush = () => {
    if (pendingInsert.length > 0) {
      tokens.push({ insert: pendingInsert.join(' ') });
      pendingInsert = [];
    }
  };

  for (const op of ops) {
    if (op.type === 'equal') {
      flush();
      tokens.push({ ref: words[op.aIndex].id });
    } else if (op.type === 'insert') {
      pendingInsert.push(surfaceB[op.bIndex]);
    }
    // Deletions simply contribute nothing to the output.
  }
  flush();
  return tokens;
}
