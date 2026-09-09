import { createScene } from './core';

/**
 * The three small diagrams behind the "ground truth" cards on the landing page.
 * Options: { kind: 'open' | 'nopower' | 'chain' }.
 */
export const micro = createScene((S, optionsRef) => {
  const walkers = [];
  for (let i = 0; i < 13; i += 1) {
    walkers.push({
      x: Math.random(),
      y: 0.28 + Math.random() * 0.5,
      sp: (Math.random() > 0.5 ? 1 : -1) * (0.00009 + Math.random() * 0.00018),
    });
  }

  return (t) => {
    const { ctx, W, H, dpr } = S;
    if (!W) return;
    const kind = (optionsRef.current || {}).kind;

    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(43,37,33,0.1)';
    ctx.lineWidth = dpr * 0.6;
    for (let x = 0; x < W; x += 20 * dpr) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }

    if (kind === 'open') {
      // legal crossings streaming over an open border line
      ctx.strokeStyle = 'rgba(43,37,33,0.42)';
      ctx.lineWidth = dpr * 1.2;
      ctx.setLineDash([8 * dpr, 5 * dpr, 2 * dpr, 5 * dpr]);
      ctx.beginPath();
      ctx.moveTo(0, H * 0.5);
      ctx.lineTo(W, H * 0.5);
      ctx.stroke();
      ctx.setLineDash([]);
      walkers.forEach((w) => {
        w.x += w.sp * 16;
        if (w.x > 1) w.x = 0;
        if (w.x < 0) w.x = 1;
        const y = H * (0.5 + (w.y - 0.5) * 0.8);
        ctx.strokeStyle = 'rgba(31,111,122,0.2)';
        ctx.lineWidth = dpr * 0.7;
        ctx.beginPath();
        ctx.moveTo(w.x * W, y);
        ctx.lineTo(w.x * W, H * 0.5);
        ctx.stroke();
        ctx.fillStyle = 'rgba(31,111,122,0.9)';
        ctx.fillRect(w.x * W - 2 * dpr, y - 2 * dpr, 4 * dpr, 4 * dpr);
      });
    } else if (kind === 'nopower') {
      // masts flickering on intermittent power, one post with no road
      ctx.strokeStyle = 'rgba(43,37,33,0.32)';
      ctx.lineWidth = dpr;
      ctx.beginPath();
      ctx.moveTo(0, H * 0.78);
      ctx.lineTo(W, H * 0.78);
      ctx.stroke();
      for (let i = 0; i < 5; i += 1) {
        const x = W * (0.12 + i * 0.19);
        const on = Math.sin(t * 0.0015 + i * 1.7) > (i === 2 ? 0.85 : 0.05);
        ctx.strokeStyle = 'rgba(43,37,33,0.52)';
        ctx.lineWidth = dpr * 1.2;
        ctx.beginPath();
        ctx.moveTo(x, H * 0.78);
        ctx.lineTo(x, H * 0.32);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x - 4 * dpr, H * 0.5);
        ctx.lineTo(x + 4 * dpr, H * 0.56);
        ctx.stroke();
        ctx.strokeStyle = on ? 'rgba(181,115,36,0.95)' : 'rgba(43,37,33,0.26)';
        ctx.strokeRect(x - 4 * dpr, H * 0.32 - 8 * dpr, 8 * dpr, 8 * dpr);
        if (on) {
          ctx.fillStyle = 'rgba(181,115,36,0.85)';
          ctx.fillRect(x - 4 * dpr, H * 0.32 - 8 * dpr, 8 * dpr, 8 * dpr);
        }
        if (i === 2) {
          ctx.fillStyle = 'rgba(163,46,36,0.9)';
          ctx.font = `${8 * dpr}px "IBM Plex Mono", monospace`;
          ctx.fillText('NO ROAD', x - 15 * dpr, H * 0.92);
        }
      }
    } else {
      // a hash chain with a block that breaks
      for (let i = 0; i < 4; i += 1) {
        const x = W * (0.12 + i * 0.25);
        const y = H * 0.5;
        const s = 26 * dpr;
        const broken = i === 2 && Math.sin(t * 0.0016) > 0;
        ctx.strokeStyle = broken ? 'rgba(163,46,36,0.9)' : 'rgba(110,75,124,0.85)';
        ctx.lineWidth = dpr * 1.2;
        ctx.strokeRect(x - s / 2, y - s / 2, s, s);
        ctx.fillStyle = broken ? 'rgba(163,46,36,0.1)' : 'rgba(110,75,124,0.1)';
        ctx.fillRect(x - s / 2, y - s / 2, s, s);
        ctx.strokeStyle = broken ? 'rgba(163,46,36,0.55)' : 'rgba(110,75,124,0.45)';
        for (let h = 1; h < 4; h += 1) {
          ctx.beginPath();
          ctx.moveTo(x - s / 2, y - s / 2 + (s * h) / 4);
          ctx.lineTo(x + s / 2, y - s / 2 + (s * h) / 4);
          ctx.stroke();
        }
        if (i < 3) {
          const nb = broken || (i === 1 && Math.sin(t * 0.0016) > 0);
          ctx.strokeStyle = nb ? 'rgba(163,46,36,0.6)' : 'rgba(110,75,124,0.55)';
          ctx.setLineDash(nb ? [3 * dpr, 3 * dpr] : []);
          ctx.beginPath();
          ctx.moveTo(x + s / 2, y);
          ctx.lineTo(x + W * 0.25 - s / 2, y);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }
  };
});

export default micro;
