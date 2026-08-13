import test from 'node:test';
import assert from 'node:assert/strict';
import { buildEdl, diffTranscript, identityTokens, isContiguous, snapToQuiet, speechRate } from '../src/index.js';

/**
 * Words at a steady cadence: 0.4s of speech then a 0.1s gap.
 * w0 = [0.0, 0.4], w1 = [0.5, 0.9], ...
 */
function makeWords(sentence) {
  return sentence.split(' ').map((text, i) => ({
    id: `w${i}`,
    text,
    start: i * 0.5,
    end: i * 0.5 + 0.4,
  }));
}

const SENTENCE = 'four score and seven years ago our fathers brought forth';

test('an unedited document yields one uninterrupted copy segment', () => {
  const words = makeWords(SENTENCE);
  const { segments, stats } = buildEdl(words, identityTokens(words), { duration: 5.0 });
  assert.equal(segments.length, 1);
  assert.equal(segments[0].kind, 'copy');
  assert.equal(stats.cuts, 0);
  assert.equal(stats.deletedWords, 0);
  assert.equal(stats.insertedWords, 0);
});

test('cuts land in the gap between words, never mid-word', () => {
  const words = makeWords(SENTENCE);
  const tokens = diffTranscript(words, 'four score and seven years ago our fathers brought forth');
  const { segments } = buildEdl(words, tokens, { duration: 5.0 });
  const seg = segments[0];
  // Head padding may reach back before the first word but never past zero.
  assert.ok(seg.start >= 0);
  assert.ok(seg.start <= words[0].start);
  assert.ok(seg.end >= words[words.length - 1].end);
});

test('deleting a phrase removes exactly that span and creates one seam', () => {
  const words = makeWords(SENTENCE);
  // Drop "score and seven" (w1..w3).
  const tokens = diffTranscript(words, 'four years ago our fathers brought forth');
  const { segments, stats } = buildEdl(words, tokens, { duration: 5.0 });

  assert.equal(segments.length, 2);
  assert.ok(segments.every((s) => s.kind === 'copy'));
  assert.equal(stats.deletedWords, 3);
  assert.equal(stats.keptWords, 7);
  assert.equal(stats.cuts, 1);

  // First segment ends after "four" (w0.end = 0.4) and before "score" (w1.start = 0.5).
  assert.ok(segments[0].end > 0.4 && segments[0].end <= 0.5, `got ${segments[0].end}`);
  // Second segment starts in the gap before "years" (w4.start = 2.0).
  assert.ok(segments[1].start > 1.9 && segments[1].start <= 2.0, `got ${segments[1].start}`);

  // The removed audio really is gone.
  const kept = segments.reduce((sum, s) => sum + (s.end - s.start), 0);
  assert.ok(kept < 5.0 - 1.4, 'at least the three deleted words worth of audio removed');
});

test('padding never drags a long silence in with a deleted word', () => {
  const words = [
    { id: 'a', text: 'hello', start: 0.0, end: 0.5 },
    { id: 'b', text: 'um', start: 3.0, end: 3.3 }, // preceded by 2.5s of silence
    { id: 'c', text: 'world', start: 6.0, end: 6.5 },
  ];
  const { segments } = buildEdl(words, [{ ref: 'a' }, { ref: 'c' }], { duration: 7.0, maxPad: 0.12 });
  assert.equal(segments.length, 2);
  // Only up to maxPad of the 2.5s silence is retained on each side.
  assert.ok(segments[0].end <= 0.5 + 0.12 + 1e-9, `got ${segments[0].end}`);
  assert.ok(segments[1].start >= 6.0 - 0.12 - 1e-9, `got ${segments[1].start}`);
});

test('adjacent kept words never produce a seam even across separate refs', () => {
  const words = makeWords(SENTENCE);
  const tokens = words.map((w) => ({ ref: w.id }));
  const { segments, stats } = buildEdl(words, tokens, { duration: 5.0 });
  assert.equal(segments.length, 1, 'contiguous refs collapse into one segment');
  assert.equal(stats.cuts, 0);
});

test('inserted text becomes a synth segment carrying both neighbours', () => {
  const words = makeWords(SENTENCE);
  const tokens = diffTranscript(words, '246 years ago our fathers brought forth');
  const { segments, stats } = buildEdl(words, tokens, { duration: 5.0 });

  assert.equal(stats.insertedWords, 1);
  const synth = segments.find((s) => s.kind === 'synth');
  assert.ok(synth, 'a synth segment exists');
  assert.equal(synth.text, '246');
  assert.equal(synth.contextAfter, 'years');
  // Nothing precedes it on the timeline, so there is no left neighbour.
  assert.equal(segments[0], synth);
  assert.equal(synth.contextBefore, null);
  assert.equal(synth.leadGap, 0, 'no leading silence at the very start of the timeline');
});

test('a mid-sentence insertion sees the words on both sides', () => {
  const words = makeWords(SENTENCE);
  const tokens = diffTranscript(words, 'four score and seven long years ago our fathers brought forth');
  const { segments } = buildEdl(words, tokens, { duration: 5.0 });
  const synth = segments.find((s) => s.kind === 'synth');
  assert.equal(synth.text, 'long');
  assert.equal(synth.contextBefore, 'seven');
  assert.equal(synth.contextAfter, 'years');
  assert.ok(synth.estimatedDuration > 0);
});

test('estimated duration tracks the speaker’s own rate', () => {
  const words = makeWords(SENTENCE);
  const rate = speechRate(words);
  assert.ok(rate > 0.3 && rate < 0.7, `implausible rate ${rate}`);

  const tokens = diffTranscript(words, `${SENTENCE} and a few extra words here`);
  const { segments, stats } = buildEdl(words, tokens, { duration: 5.0 });
  const synth = segments.find((s) => s.kind === 'synth');
  assert.equal(stats.insertedWords, 6);
  // Six words at the measured rate, give or take the inter-word gap.
  assert.ok(Math.abs(synth.estimatedDuration - 6 * rate) < 0.5, `got ${synth.estimatedDuration}`);
  assert.ok(stats.estimatedDuration > stats.sourceDuration);
});

test('deleting words shortens the estimated result', () => {
  const words = makeWords(SENTENCE);
  const tokens = diffTranscript(words, 'four years ago');
  const { stats } = buildEdl(words, tokens, { duration: 5.0 });
  assert.ok(stats.estimatedDuration < stats.sourceDuration);
});

test('stale word ids are ignored rather than throwing', () => {
  const words = makeWords('one two three');
  const { segments, stats } = buildEdl(words, [{ ref: 'w0' }, { ref: 'gone' }, { ref: 'w2' }], {
    duration: 2.0,
  });
  assert.equal(stats.keptWords, 2);
  assert.equal(segments.length, 2);
});

test('blank insertions are dropped', () => {
  const words = makeWords('one two');
  const { segments, stats } = buildEdl(words, [{ ref: 'w0' }, { insert: '   ' }, { ref: 'w1' }], {
    duration: 1.5,
  });
  assert.equal(stats.insertedWords, 0);
  assert.equal(segments.length, 1, 'the two words rejoin as one contiguous copy');
});

test('an all-synthetic document produces no copy segments', () => {
  const words = makeWords('one two');
  const { segments, stats } = buildEdl(words, [{ insert: 'completely new line' }], { duration: 1.5 });
  assert.equal(segments.length, 1);
  assert.equal(segments[0].kind, 'synth');
  assert.equal(stats.keptWords, 0);
  assert.equal(stats.deletedWords, 2);
  assert.equal(stats.insertedWords, 3);
});

test('segments are ordered and never overlap in the source', () => {
  const words = makeWords(SENTENCE);
  const tokens = diffTranscript(words, 'four seven ago fathers forth');
  const { segments } = buildEdl(words, tokens, { duration: 5.0 });
  let prevEnd = -Infinity;
  for (const seg of segments) {
    if (seg.kind !== 'copy') continue;
    assert.ok(seg.start < seg.end, 'segment has positive duration');
    assert.ok(seg.start >= prevEnd - 1e-9, `segment ${seg.start} overlaps previous end ${prevEnd}`);
    prevEnd = seg.end;
  }
});

test('reordering words is expressed as copies, not resynthesis', () => {
  const words = makeWords('alpha bravo charlie');
  // Moving "charlie" to the front: the aligner keeps whichever run is longest
  // and re-synthesises the rest, but must still reproduce the target text.
  const tokens = diffTranscript(words, 'charlie alpha bravo');
  const { segments } = buildEdl(words, tokens, { duration: 1.5 });
  assert.ok(segments.length >= 1);
  const copiedWords = segments.filter((s) => s.kind === 'copy').flatMap((s) => s.wordIds);
  assert.ok(copiedWords.length >= 2, 'the alpha/bravo run is reused verbatim');
});

test('isContiguous only accepts genuinely adjacent copies', () => {
  const a = { kind: 'copy', start: 0, end: 1, wordIds: ['w0'], firstWordIndex: 0, lastWordIndex: 0 };
  const b = { kind: 'copy', start: 1, end: 2, wordIds: ['w1'], firstWordIndex: 1, lastWordIndex: 1 };
  const c = { kind: 'copy', start: 5, end: 6, wordIds: ['w9'], firstWordIndex: 9, lastWordIndex: 9 };
  assert.equal(isContiguous(a, b), true);
  assert.equal(isContiguous(a, c), false);
  assert.equal(isContiguous(a, { kind: 'synth', text: 'x' }), false);
});

test('snapToQuiet moves a cut to the local energy minimum', () => {
  // 100 fps envelope: loud everywhere except a dip at 1.00s.
  const rms = new Array(200).fill(0.5);
  rms[100] = 0.01;
  const envelope = { fps: 100, rms };
  const snapped = snapToQuiet(1.03, 0.9, 1.1, envelope, 0.06);
  assert.ok(Math.abs(snapped - 1.0) < 0.011, `expected ~1.00, got ${snapped}`);
});

test('snapToQuiet respects the permitted range and missing envelopes', () => {
  const rms = new Array(200).fill(0.5);
  rms[0] = 0.0; // a quiet point far outside the allowed range
  const envelope = { fps: 100, rms };
  const snapped = snapToQuiet(1.0, 0.98, 1.02, envelope, 0.5);
  assert.ok(snapped >= 0.98 && snapped <= 1.02, `escaped its bounds: ${snapped}`);
  assert.equal(snapToQuiet(1.0, 0.9, 1.1, null), 1.0);
});

test('an envelope shifts cut points without leaving the inter-word gap', () => {
  const words = makeWords(SENTENCE);
  const rms = new Array(600).fill(0.4);
  const envelope = { fps: 100, rms };
  const tokens = diffTranscript(words, 'four years ago our fathers brought forth');
  const { segments } = buildEdl(words, tokens, { duration: 5.0, envelope });
  assert.ok(segments[0].end > words[0].end && segments[0].end <= words[1].start);
});
