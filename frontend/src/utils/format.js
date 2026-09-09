export const pad = (n) => (n < 10 ? `0${n}` : `${n}`);

export const stamp = (d = new Date()) =>
  `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;

export const hex = (n) => {
  const chars = '0123456789abcdef';
  let out = '';
  for (let i = 0; i < n; i += 1) out += chars[Math.floor(Math.random() * 16)];
  return out;
};

/** Builds the SVG path used by the sparklines in the console analytics panel. */
export const sparkPath = (values, w, h, max) =>
  values
    .map(
      (v, i) =>
        `${i ? 'L' : 'M'}${((i / (values.length - 1)) * w).toFixed(1)} ${(h - (v / max) * h).toFixed(1)}`,
    )
    .join(' ');
