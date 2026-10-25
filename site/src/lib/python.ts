/**
 * A small highlighter for the Python of the site: keywords, strings, numbers, comments, calls and types.
 *
 * The landing page and the checked examples of the documentation both use
 * it, so code has the same colors everywhere on the site. Each token is a
 * span with a class `tk-<kind>`, and the stylesheet gives the colors.
 */

export function escape(text: string): string {
  return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
}

const KEYWORDS = new Set([
  'import',
  'from',
  'def',
  'class',
  'return',
  'as',
  'if',
  'elif',
  'else',
  'in',
  'is',
  'not',
  'and',
  'or',
  'None',
  'True',
  'False',
  'with',
  'for',
  'while',
  'lambda',
  'yield',
  'raise',
  'try',
  'except',
  'pass',
]);
const TOKEN = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z_0-9]*)|(\s+)|(.)/g;

function kind(match: RegExpExecArray, text: string): string {
  if (match[1]) return 'comment';
  if (match[2]) return 'string';
  if (match[3]) return 'number';
  const word = match[4];
  if (!word) return 'plain';
  if (KEYWORDS.has(word)) return 'keyword';
  if (/^[A-Z]/.test(word)) return 'type';
  return /^\s*\(/.test(text.slice(match.index + word.length)) ? 'call' : 'plain';
}

/** One line of Python as colored spans. */
export function tokens(text: string): string {
  let out = '';
  const pattern = new RegExp(TOKEN.source, 'g');
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    out += `<span class="tk-${kind(match, text)}">${escape(match[0])}</span>`;
  }
  return out;
}

/** Text with inline code: each part between backticks becomes a code span. */
export function parts(text: string, codeClass = 'ic'): string {
  return text
    .split('`')
    .map((piece, index) => (index % 2 === 1 ? `<code class="${codeClass}">${escape(piece)}</code>` : escape(piece)))
    .join('');
}
