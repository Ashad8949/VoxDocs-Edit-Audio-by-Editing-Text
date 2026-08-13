/**
 * Token normalisation.
 *
 * Alignment between the original transcript and the user's edited text has to
 * ignore differences that carry no audio consequence: case, punctuation,
 * smart quotes, and the leading space that ASR engines attach to each word.
 * Two tokens that normalise to the same key are considered the same spoken
 * word, so the original audio for it can be kept verbatim.
 */

/** Characters that are punctuation for our purposes but may live inside a word. */
const INNER_PUNCT = /[‘’ʼ']/g; // curly + straight apostrophes
const OUTER_PUNCT = /[^\p{L}\p{N}]+/gu;

/**
 * Reduce a surface word to its comparison key.
 * @param {string} raw
 * @returns {string} normalised key, possibly `''` for punctuation-only input
 */
export function normalizeToken(raw) {
  if (typeof raw !== 'string') return '';
  return raw
    .normalize('NFKC')
    .toLowerCase()
    .replace(INNER_PUNCT, '')
    .replace(OUTER_PUNCT, '')
    .trim();
}

/**
 * Split free text into surface words, preserving the original spelling so that
 * inserted text can be sent to the synthesiser exactly as the user typed it.
 * @param {string} text
 * @returns {string[]}
 */
export function tokenizeText(text) {
  if (!text) return [];
  return text
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .filter((t) => t.length > 0 && normalizeToken(t).length > 0);
}
