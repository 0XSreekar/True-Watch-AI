/** Small helpers shared by the mock services. Delete once real data lands. */

export const hex = (n) => {
  const chars = '0123456789abcdef';
  let out = '';
  for (let i = 0; i < n; i += 1) out += chars[Math.floor(Math.random() * 16)];
  return out;
};

export const pad = (n) => (n < 10 ? `0${n}` : `${n}`);

export const stamp = (d = new Date()) =>
  `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;

export const pick = (list) => list[Math.floor(Math.random() * list.length)];

export const chance = (p) => Math.random() > p;
