import test from 'node:test';
import assert from 'node:assert/strict';
import {
  __resetInsertIds,
  deleteRange,
  deletedSpans,
  docFromWords,
  docText,
  insertAt,
  localStats,
  replaceRange,
  restoreRange,
  toTokens,
  updateInsert,
  wordIndexAtTime,
} from '../src/lib/doc.js';

const WORDS = 'four score and seven years ago'.split(' ').map((text, i) => ({
  id: `w${i}`,
  text,
  start: i * 0.5,
  end: i * 0.5 + 0.4,
}));

test.beforeEach(() => __resetInsertIds());

test('a fresh document mirrors the transcript', () => {
  const doc = docFromWords(WORDS);
  assert.equal(doc.length, 6);
  assert.ok(doc.every((b) => b.kind === 'word' && !b.deleted));
  assert.equal(docText(doc), 'four score and seven years ago');
  assert.equal(toTokens(doc).length, 6);
});

test('segment indices are carried onto each word for paragraph layout', () => {
  const doc = docFromWords(WORDS, [
    { first_word: 0, last_word: 2 },
    { first_word: 3, last_word: 5 },
  ]);
  assert.deepEqual(doc.map((b) => b.segment), [0, 0, 0, 1, 1, 1]);
});

test('deleting marks words rather than dropping them, so it can be undone', () => {
  const doc = deleteRange(docFromWords(WORDS), 1, 3);
  assert.equal(doc.length, 6, 'blocks are retained');
  assert.deepEqual(doc.map((b) => b.deleted), [false, true, true, true, false, false]);
  assert.equal(docText(doc), 'four years ago');
  assert.deepEqual(toTokens(doc), [{ ref: 'w0' }, { ref: 'w4' }, { ref: 'w5' }]);
});

test('deleting accepts a backwards range', () => {
  const doc = deleteRange(docFromWords(WORDS), 3, 1);
  assert.equal(docText(doc), 'four years ago');
});

test('deleting clamps out-of-range indices', () => {
  const doc = deleteRange(docFromWords(WORDS), -5, 1);
  assert.equal(docText(doc), 'and seven years ago');
  assert.equal(deleteRange(docFromWords(WORDS), 10, 20).length, 6);
});

test('restoring brings deleted words back', () => {
  let doc = deleteRange(docFromWords(WORDS), 1, 3);
  doc = restoreRange(doc, 2, 2);
  assert.equal(docText(doc), 'four and years ago');
  doc = restoreRange(doc, 0, 5);
  assert.equal(docText(doc), 'four score and seven years ago');
});

test('the original document is never mutated', () => {
  const doc = docFromWords(WORDS);
  const snapshot = JSON.stringify(doc);
  deleteRange(doc, 0, 3);
  insertAt(doc, 2, 'hello');
  replaceRange(doc, 1, 2, 'x');
  assert.equal(JSON.stringify(doc), snapshot, 'editing must be pure for undo to work');
});

test('inserting text creates a synthesis block at the right position', () => {
  const doc = insertAt(docFromWords(WORDS), 0, '246');
  assert.equal(doc[0].kind, 'insert');
  assert.equal(docText(doc), '246 four score and seven years ago');
  assert.deepEqual(toTokens(doc)[0], { insert: '246' });
});

test('inserting at the end appends', () => {
  const doc = insertAt(docFromWords(WORDS), 6, 'amen');
  assert.equal(docText(doc), 'four score and seven years ago amen');
});

test('adjacent insertions merge into one phrase', () => {
  let doc = insertAt(docFromWords(WORDS), 2, 'brave');
  doc = insertAt(doc, 3, 'new');
  const inserts = doc.filter((b) => b.kind === 'insert');
  assert.equal(inserts.length, 1, 'the synthesiser should see one phrase, not two words');
  assert.equal(inserts[0].text, 'brave new');
  assert.equal(docText(doc), 'four score brave new and seven years ago');
});

test('an insertion merges with a following insert block too', () => {
  let doc = insertAt(docFromWords(WORDS), 2, 'world');
  doc = insertAt(doc, 2, 'hello');
  assert.equal(doc.filter((b) => b.kind === 'insert').length, 1);
  assert.equal(docText(doc), 'four score hello world and seven years ago');
});

test('blank insertions are ignored', () => {
  const doc = docFromWords(WORDS);
  assert.equal(insertAt(doc, 2, '   '), doc);
  assert.equal(insertAt(doc, 2, ''), doc);
});

test('editing an insert block updates it, and emptying it removes it', () => {
  let doc = insertAt(docFromWords(WORDS), 0, '246');
  const id = doc[0].id;
  doc = updateInsert(doc, id, '1776');
  assert.equal(docText(doc), '1776 four score and seven years ago');
  doc = updateInsert(doc, id, '  ');
  assert.ok(doc.every((b) => b.kind === 'word'));
});

test('typing over a selection strikes it out and puts the text in its place', () => {
  // The demo edit: "four score and seven years ago" -> "246 years ago".
  const doc = replaceRange(docFromWords(WORDS), 0, 3, '246');
  assert.equal(docText(doc), '246 years ago');
  assert.deepEqual(toTokens(doc), [{ insert: '246' }, { ref: 'w4' }, { ref: 'w5' }]);
  assert.equal(doc.filter((b) => b.kind === 'word' && b.deleted).length, 4);
});

test('replacing a middle run keeps both sides intact', () => {
  const doc = replaceRange(docFromWords(WORDS), 2, 3, 'plus');
  assert.equal(docText(doc), 'four score plus years ago');
});

test('replacing a range that contains an insert block lands the text correctly', () => {
  let doc = insertAt(docFromWords(WORDS), 2, 'temp');
  // doc: four score [temp] and seven years ago
  doc = replaceRange(doc, 1, 3, 'final');
  assert.equal(docText(doc), 'four final seven years ago');
  assert.equal(doc.filter((b) => b.kind === 'insert').length, 1);
});

test('deleting everything yields an empty edit, not a crash', () => {
  const doc = deleteRange(docFromWords(WORDS), 0, 5);
  assert.equal(docText(doc), '');
  assert.deepEqual(toTokens(doc), []);
});

test('local stats count words and surviving audio', () => {
  let doc = deleteRange(docFromWords(WORDS), 0, 1);
  doc = insertAt(doc, 0, 'two new words here');
  const stats = localStats(doc);
  assert.equal(stats.kept, 4);
  assert.equal(stats.deleted, 2);
  assert.equal(stats.inserted, 4);
  assert.ok(Math.abs(stats.keptSeconds - 1.6) < 1e-6);
});

test('the word under the playhead is found by time', () => {
  const doc = docFromWords(WORDS);
  assert.equal(wordIndexAtTime(doc, 0.1), 0);
  assert.equal(wordIndexAtTime(doc, 1.2), 2);
  assert.equal(wordIndexAtTime(doc, 0.45), -1, 'gaps between words match nothing');
  assert.equal(wordIndexAtTime(doc, 99), -1);
});

test('deleted spans merge when they are adjacent in the source', () => {
  const doc = deleteRange(docFromWords(WORDS), 1, 3);
  const spans = deletedSpans(doc);
  assert.equal(spans.length, 1, 'three consecutive deletions are one region on the waveform');
  assert.ok(Math.abs(spans[0].start - 0.5) < 1e-6);
  assert.ok(Math.abs(spans[0].end - 1.9) < 1e-6);
});

test('deleted spans stay separate when far apart', () => {
  let doc = deleteRange(docFromWords(WORDS), 0, 0);
  doc = deleteRange(doc, 5, 5);
  assert.equal(deletedSpans(doc).length, 2);
});
