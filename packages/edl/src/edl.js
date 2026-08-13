/**
 * Edit Decision List construction.
 *
 * An EDL is the bridge between "what the transcript now says" and "what the
 * renderer has to do with samples". Every surviving stretch of original speech
 * becomes a `copy` segment carrying the exact source time range; every stretch
 * the user typed becomes a `synth` segment carrying the text plus the
 * neighbouring words, which the synthesiser uses to match prosody.
 *
 * The interesting decisions here are about *where to cut*. Word timings from an
 * ASR model mark roughly where the vowel energy is, not where the silence is.
 * Cutting exactly on `word.end` reliably clips the release of a final stop
 * consonant and produces the tell-tale "chopped" sound of naive text-based
 * editing. Instead we cut in the middle of the gap between words, bounded so
 * that removing a word never drags a long silence in with it, and optionally
 * snap to the quietest point nearby using an energy envelope.
 */

import { tokenizeText } from './tokens.js';

/** Most silence (seconds) we will pull in on either side of a cut. */
const DEFAULT_MAX_PAD = 0.12;
/** How far (seconds) we search for a quieter place to cut. */
const DEFAULT_SNAP_WINDOW = 0.06;
/** Silence inserted around synthesised speech when the source gives no clue. */
const DEFAULT_SYNTH_GAP = 0.08;
/** Fallback speech rate (seconds per word) if the source is too short to measure. */
const FALLBACK_SEC_PER_WORD = 0.34;

/**
 * @typedef {{ id: string, text: string, start: number, end: number }} Word
 * @typedef {{ ref: string } | { insert: string }} EditToken
 * @typedef {{ fps: number, rms: number[] }} Envelope
 */

/**
 * @typedef {{
 *   kind: 'copy',
 *   start: number,
 *   end: number,
 *   wordIds: string[],
 *   firstWordIndex: number,
 *   lastWordIndex: number
 * }} CopySegment
 *
 * @typedef {{
 *   kind: 'synth',
 *   text: string,
 *   contextBefore: string | null,
 *   contextAfter: string | null,
 *   leadGap: number,
 *   trailGap: number,
 *   estimatedDuration: number
 * }} SynthSegment
 *
 * @typedef {CopySegment | SynthSegment} EdlSegment
 */

/**
 * Measure the speaker's articulation rate, used to predict how long inserted
 * text will take in their voice.
 * @param {Word[]} words
 * @returns {number} seconds per word
 */
export function speechRate(words) {
  if (!words || words.length < 4) return FALLBACK_SEC_PER_WORD;
  let spoken = 0;
  for (const w of words) spoken += Math.max(0, w.end - w.start);
  if (spoken <= 0) return FALLBACK_SEC_PER_WORD;
  // Include a share of inter-word gaps: real speech is not gapless.
  const span = words[words.length - 1].end - words[0].start;
  const gapShare = Math.max(0, span - spoken) / words.length;
  return spoken / words.length + Math.min(gapShare, 0.12);
}

/**
 * Median inter-word gap, used to space synthesised insertions naturally.
 * @param {Word[]} words
 */
export function medianGap(words) {
  if (!words || words.length < 2) return DEFAULT_SYNTH_GAP;
  const gaps = [];
  for (let i = 1; i < words.length; i++) {
    const g = words[i].start - words[i - 1].end;
    if (g >= 0 && g < 1.0) gaps.push(g);
  }
  if (gaps.length === 0) return DEFAULT_SYNTH_GAP;
  gaps.sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length / 2)];
}

/**
 * A candidate cut point must be at least this much quieter than the point we
 * were already going to cut at before it is worth moving to.
 */
const SNAP_IMPROVEMENT = 0.7;

/**
 * Find the quietest instant within a window, so cuts land in silence rather
 * than mid-phoneme.
 *
 * Snapping is deliberately conservative. The gap-midpoint target is already a
 * good cut; moving it is only justified by a clearly quieter neighbour, and
 * among equally quiet candidates the one nearest the target wins. Without this,
 * a flat (uniformly quiet) envelope would drag every cut to the edge of its
 * search window for no acoustic benefit.
 *
 * @param {number} target desired cut time
 * @param {number} lo earliest acceptable time
 * @param {number} hi latest acceptable time
 * @param {Envelope | null | undefined} envelope
 * @param {number} window search radius in seconds
 * @returns {number}
 */
export function snapToQuiet(target, lo, hi, envelope, window = DEFAULT_SNAP_WINDOW) {
  const fallback = clamp(target, lo, hi);
  if (!envelope || !envelope.rms || !envelope.fps || envelope.rms.length === 0) return fallback;

  const searchLo = clamp(target - window, lo, hi);
  const searchHi = clamp(target + window, lo, hi);
  if (searchHi <= searchLo) return fallback;

  const last = Math.min(envelope.rms.length - 1, Math.ceil(searchHi * envelope.fps));
  const first = Math.max(0, Math.min(last, Math.floor(searchLo * envelope.fps)));
  if (last <= first) return fallback;

  const targetIndex = clamp(Math.round(fallback * envelope.fps), first, last);
  const targetValue = envelope.rms[targetIndex];

  let bestValue = Infinity;
  for (let i = first; i <= last; i++) {
    if (envelope.rms[i] < bestValue) bestValue = envelope.rms[i];
  }
  if (!(bestValue < targetValue * SNAP_IMPROVEMENT)) return fallback;

  // Among near-optimal candidates prefer the least movement.
  const tolerance = bestValue * 1.05 + 1e-9;
  let bestIndex = targetIndex;
  let bestDistance = Infinity;
  for (let i = first; i <= last; i++) {
    if (envelope.rms[i] > tolerance) continue;
    const distance = Math.abs(i - targetIndex);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return clamp(bestIndex / envelope.fps, lo, hi);
}

/** @param {number} v @param {number} lo @param {number} hi */
function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Time at which playback of `words[index]` should begin.
 * @param {Word[]} words
 * @param {number} index
 * @param {{ maxPad: number, envelope?: Envelope | null, snapWindow: number, duration: number }} opts
 */
function boundaryBefore(words, index, opts) {
  const word = words[index];
  if (index === 0) {
    const pad = Math.min(opts.maxPad, word.start);
    return snapToQuiet(word.start - pad, Math.max(0, word.start - pad), word.start, opts.envelope, opts.snapWindow);
  }
  const prev = words[index - 1];
  const gap = word.start - prev.end;
  if (gap <= 0) return word.start;
  const pad = Math.min(opts.maxPad, gap / 2);
  const target = word.start - pad;
  return snapToQuiet(target, word.start - gap / 2, word.start, opts.envelope, opts.snapWindow);
}

/**
 * Time at which playback of `words[index]` should end.
 * @param {Word[]} words
 * @param {number} index
 * @param {{ maxPad: number, envelope?: Envelope | null, snapWindow: number, duration: number }} opts
 */
function boundaryAfter(words, index, opts) {
  const word = words[index];
  if (index === words.length - 1) {
    const room = Math.max(0, opts.duration - word.end);
    const pad = Math.min(opts.maxPad, room);
    return snapToQuiet(word.end + pad, word.end, word.end + pad, opts.envelope, opts.snapWindow);
  }
  const next = words[index + 1];
  const gap = next.start - word.end;
  if (gap <= 0) return word.end;
  const pad = Math.min(opts.maxPad, gap / 2);
  const target = word.end + pad;
  return snapToQuiet(target, word.end, word.end + gap / 2, opts.envelope, opts.snapWindow);
}

/**
 * Build an EDL from the original words and the user's edit tokens.
 *
 * @param {Word[]} words original transcript, in time order
 * @param {EditToken[]} tokens the edited document
 * @param {{
 *   duration?: number,
 *   envelope?: Envelope | null,
 *   maxPad?: number,
 *   snapWindow?: number
 * }} [options]
 * @returns {{
 *   segments: EdlSegment[],
 *   stats: {
 *     sourceWords: number, keptWords: number, deletedWords: number,
 *     insertedWords: number, sourceDuration: number, estimatedDuration: number,
 *     cuts: number
 *   }
 * }}
 */
export function buildEdl(words, tokens, options = {}) {
  const duration = options.duration ?? (words.length ? words[words.length - 1].end : 0);
  const opts = {
    maxPad: options.maxPad ?? DEFAULT_MAX_PAD,
    snapWindow: options.snapWindow ?? DEFAULT_SNAP_WINDOW,
    envelope: options.envelope ?? null,
    duration,
  };

  /** @type {Map<string, number>} */
  const indexById = new Map();
  words.forEach((w, i) => indexById.set(w.id, i));

  const gap = medianGap(words);
  const secPerWord = speechRate(words);

  // Normalise the token list into runs: consecutive refs whose source indices
  // are contiguous can be lifted from the original in one uninterrupted piece.
  /** @type {EdlSegment[]} */
  const segments = [];
  /** @type {number[]} */
  let run = [];
  let insertedWords = 0;
  const keptIndices = new Set();

  const flushRun = () => {
    if (run.length === 0) return;
    const first = run[0];
    const last = run[run.length - 1];
    segments.push({
      kind: 'copy',
      start: boundaryBefore(words, first, opts),
      end: boundaryAfter(words, last, opts),
      wordIds: run.map((i) => words[i].id),
      firstWordIndex: first,
      lastWordIndex: last,
    });
    run = [];
  };

  for (const token of tokens) {
    if ('ref' in token) {
      const index = indexById.get(token.ref);
      if (index === undefined) continue; // stale id — ignore rather than fail
      keptIndices.add(index);
      if (run.length > 0 && index !== run[run.length - 1] + 1) {
        flushRun(); // a cut: the kept words are not neighbours in the source
      }
      run.push(index);
    } else if ('insert' in token) {
      const text = (token.insert ?? '').trim();
      const wordCount = tokenizeText(text).length;
      if (wordCount === 0) continue;
      const contextBefore =
        run.length > 0 ? words[run[run.length - 1]].text : precedingWordText(segments, words);
      flushRun();
      insertedWords += wordCount;
      segments.push({
        kind: 'synth',
        text,
        contextBefore,
        contextAfter: null, // filled in below once we know what follows
        leadGap: gap,
        trailGap: gap,
        estimatedDuration: wordCount * secPerWord + gap,
      });
    }
  }
  flushRun();

  // Second pass: give each synth segment the word that follows it, so the
  // synthesiser can pick a coarticulation-friendly candidate from the voice bank.
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    if (seg.kind !== 'synth') continue;
    const next = segments[i + 1];
    if (next && next.kind === 'copy') {
      seg.contextAfter = words[next.firstWordIndex].text;
    }
    // At the very start or end of the timeline there is nothing to bridge to.
    if (i === 0) seg.leadGap = 0;
    if (i === segments.length - 1) seg.trailGap = 0;
  }

  let estimatedDuration = 0;
  let cuts = 0;
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    estimatedDuration += seg.kind === 'copy' ? seg.end - seg.start : seg.estimatedDuration;
    if (i > 0 && !isContiguous(segments[i - 1], seg)) cuts++;
  }

  return {
    segments,
    stats: {
      sourceWords: words.length,
      keptWords: keptIndices.size,
      deletedWords: words.length - keptIndices.size,
      insertedWords,
      sourceDuration: duration,
      estimatedDuration,
      cuts,
    },
  };
}

/**
 * Two segments are contiguous when playing them back to back reproduces the
 * original waveform exactly — no seam, so no crossfade needed.
 * @param {EdlSegment} a
 * @param {EdlSegment} b
 */
export function isContiguous(a, b) {
  if (a.kind !== 'copy' || b.kind !== 'copy') return false;
  if (b.firstWordIndex !== a.lastWordIndex + 1) return false;
  return Math.abs(b.start - a.end) < 1e-6;
}

/**
 * Surface text of the last original word emitted so far, or null if the
 * timeline so far contains nothing but synthesised speech.
 * @param {EdlSegment[]} segments
 * @param {Word[]} words
 * @returns {string | null}
 */
function precedingWordText(segments, words) {
  for (let i = segments.length - 1; i >= 0; i--) {
    const seg = segments[i];
    if (seg.kind === 'copy') return words[seg.lastWordIndex].text;
  }
  return null;
}

/**
 * Convenience: the identity edit, keeping every word.
 * @param {Word[]} words
 * @returns {EditToken[]}
 */
export function identityTokens(words) {
  return words.map((w) => ({ ref: w.id }));
}
