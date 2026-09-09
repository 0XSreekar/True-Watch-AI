/**
 * The prototype expressed every style as a CSS declaration string, often computed
 * from state. Rather than rewrite hundreds of them by hand (and risk drifting from
 * the original design), we keep the strings and convert them to React style objects.
 *
 *   <div style={css(`padding: 12px; background: ${on ? '#FDFBF6' : '#FAF7F0'};`)} />
 */
const cache = new Map();

const toCamel = (prop) =>
  prop.startsWith('--') ? prop : prop.replace(/-([a-z])/g, (_, c) => c.toUpperCase());

export const css = (declarations) => {
  if (!declarations) return undefined;
  const cached = cache.get(declarations);
  if (cached) return cached;

  const style = {};
  for (const chunk of declarations.split(';')) {
    const decl = chunk.trim();
    if (!decl) continue;
    const idx = decl.indexOf(':');
    if (idx === -1) continue;
    style[toCamel(decl.slice(0, idx).trim())] = decl.slice(idx + 1).trim();
  }

  // Computed strings are unbounded, so keep the cache from growing without limit.
  if (cache.size > 4000) cache.clear();
  cache.set(declarations, style);
  return style;
};

export default css;
