import test from 'node:test';
import assert from 'node:assert/strict';
import { diffTranscript, alignTokens, normalizeToken, tokenizeText } from '../src/index.js';

/** Build a synthetic transcript with one word per 0.5s. */
function makeWords(sentence) {
  return sentence.split(' ').map((text, i) => ({
    id: `w${i}`,
    text,
    start: i * 0.5,
    end: i * 0.5 + 0.4,
  }));
}

/** Render an edit-token list back to a readable form for assertions. */
function describe(tokens, words) {
  const byId = new Map(words.map((w) => [w.id, w.text]));
  return tokens.map((t) => ('ref' in t ? byId.get(t.ref) : `+[${t.insert}]`)).join(' ');
}

test('normalizeToken strips case, punctuation and apostrophes', () => {
  assert.equal(normalizeToken(' The'), 'the');
  assert.equal(normalizeToken('space.'), 'space');
  assert.equal(normalizeToken("don't"), 'dont');
  assert.equal(normalizeToken('don’t'), 'dont');
  assert.equal(normalizeToken('—'), '');
  assert.equal(normalizeToken('2022'), '2022');
});

test('tokenizeText drops punctuation-only tokens', () => {
  assert.deepEqual(tokenizeText('hello,  world —  ok'), ['hello,', 'world', 'ok']);
  assert.deepEqual(tokenizeText('   '), []);
});

test('an unchanged transcript keeps every word and inserts nothing', () => {
  const words = makeWords('the goal of contrastive representation learning');
  const tokens = diffTranscript(words, 'the goal of contrastive representation learning');
  assert.equal(tokens.length, words.length);
  assert.ok(tokens.every((t) => 'ref' in t));
  assert.deepEqual(
    tokens.map((t) => t.ref),
    words.map((w) => w.id)
  );
});

test('deleting a middle phrase keeps the surrounding words intact', () => {
  const words = makeWords('the goal of contrastive representation learning is to learn');
  const tokens = diffTranscript(words, 'the goal is to learn');
  assert.equal(describe(tokens, words), 'the goal is to learn');
  assert.ok(tokens.every((t) => 'ref' in t), 'no synthesis needed for a pure deletion');
});

test('a pure deletion at the head is not misread as a replacement', () => {
  const words = makeWords('four score and seven years ago our fathers');
  const tokens = diffTranscript(words, 'our fathers');
  assert.deepEqual(
    tokens.map((t) => t.ref),
    ['w6', 'w7']
  );
});

test('inserted words become synthesis tokens and neighbours stay referenced', () => {
  const words = makeWords('four score and seven years ago our fathers');
  const tokens = diffTranscript(words, '246 years ago our fathers');
  assert.equal(describe(tokens, words), '+[246] years ago our fathers');
  const inserts = tokens.filter((t) => 'insert' in t);
  assert.equal(inserts.length, 1);
  assert.equal(inserts[0].insert, '246');
});

test('consecutive inserted words merge into a single synth token', () => {
  const words = makeWords('hello world');
  const tokens = diffTranscript(words, 'hello brave new world');
  assert.equal(describe(tokens, words), 'hello +[brave new] world');
});

test('punctuation and capitalisation changes do not force resynthesis', () => {
  const words = makeWords('all men are created equal');
  const tokens = diffTranscript(words, 'All men, are created equal!');
  assert.ok(tokens.every((t) => 'ref' in t));
  assert.equal(tokens.length, 5);
});

test('a word replaced in place yields exactly one insertion', () => {
  const words = makeWords('the quick brown fox jumps');
  const tokens = diffTranscript(words, 'the quick red fox jumps');
  assert.equal(describe(tokens, words), 'the quick +[red] fox jumps');
});

test('repeated words align to the nearest surviving occurrence', () => {
  const words = makeWords('the cat sat on the mat and the cat left');
  const tokens = diffTranscript(words, 'the cat sat on the mat and the cat left');
  assert.deepEqual(
    tokens.map((t) => t.ref),
    words.map((w) => w.id)
  );
});

test('emptying the transcript deletes everything and synthesises nothing', () => {
  const words = makeWords('one two three');
  assert.deepEqual(diffTranscript(words, ''), []);
  assert.deepEqual(diffTranscript(words, '   '), []);
});

test('typing into an empty transcript produces one insertion', () => {
  const tokens = diffTranscript([], 'brand new sentence');
  assert.deepEqual(tokens, [{ insert: 'brand new sentence' }]);
});

test('alignTokens is exact on a large document with a small edit', () => {
  // 4000 tokens exercises the patience-anchoring path, not plain Myers.
  const a = Array.from({ length: 4000 }, (_, i) => `tok${i}`);
  const b = a.slice();
  b.splice(2000, 3); // remove three words in the middle
  const ops = alignTokens(a, b);
  const equals = ops.filter((o) => o.type === 'equal').length;
  const deletes = ops.filter((o) => o.type === 'delete').length;
  const inserts = ops.filter((o) => o.type === 'insert').length;
  assert.equal(equals, 3997);
  assert.equal(deletes, 3);
  assert.equal(inserts, 0);
});

test('alignTokens handles a large document with heavy repetition', () => {
  // Almost no unique anchors: the fallback path must still produce a valid script.
  const a = Array.from({ length: 1500 }, (_, i) => (i % 3 === 0 ? 'a' : 'b'));
  const b = a.slice(0, 1400);
  const ops = alignTokens(a, b);
  const equals = ops.filter((o) => o.type === 'equal').length;
  assert.equal(equals + ops.filter((o) => o.type === 'delete').length, a.length);
  assert.equal(equals + ops.filter((o) => o.type === 'insert').length, b.length);
});

test('edit scripts always reconstruct the target sequence', () => {
  const cases = [
    ['a b c d e', 'a c e'],
    ['a b c', 'x y z'],
    ['', 'a b'],
    ['a b', ''],
    ['same words here', 'same words here'],
    ['one two two three', 'one two three'],
  ];
  for (const [left, right] of cases) {
    const a = left ? left.split(' ') : [];
    const b = right ? right.split(' ') : [];
    const ops = alignTokens(a, b);
    const rebuilt = ops
      .filter((o) => o.type !== 'delete')
      .map((o) => (o.type === 'equal' ? a[o.aIndex] : b[o.bIndex]));
    assert.deepEqual(rebuilt, b, `failed reconstructing ${right} from ${left}`);
    // Equal ops must reference genuinely identical tokens.
    for (const op of ops) {
      if (op.type === 'equal') assert.equal(a[op.aIndex], b[op.bIndex]);
    }
    // Indices must be strictly increasing on both sides.
    let ai = -1;
    let bi = -1;
    for (const op of ops) {
      if (op.type === 'equal') {
        assert.ok(op.aIndex > ai && op.bIndex > bi);
        ai = op.aIndex;
        bi = op.bIndex;
      } else if (op.type === 'delete') {
        assert.ok(op.aIndex > ai);
        ai = op.aIndex;
      } else {
        assert.ok(op.bIndex > bi);
        bi = op.bIndex;
      }
    }
  }
});
