import { pointer } from '../hooks/usePointer';
import { createScene } from './core';

/**
 * Plan view: a ruled survey sheet with the border traverse, pillar marks and a
 * sweeping radar wedge. Used behind the auth screen (onDark) and in the console
 * sector map. Options: { onDark }.
 */
export const plan = createScene((S, optionsRef) => {
  const pillars = [];
  for (let i = 0; i < 11; i += 1) {
    pillars.push({ u: 0.06 + i * 0.088, v: 0.5 + Math.sin(i * 0.9) * 0.16 });
  }

  return (t) => {
    const { ctx, W, H, dpr } = S;
    if (!W) return;
    const o = optionsRef.current || {};
    const mx = pointer.x;
    const my = pointer.y;

    ctx.clearRect(0, 0, W, H);
    const ox = mx * 14 * dpr;
    const oy = my * 10 * dpr;

    // graph paper
    ctx.strokeStyle = o.onDark ? 'rgba(239,234,221,0.10)' : 'rgba(43,37,33,0.09)';
    ctx.lineWidth = dpr * 0.6;
    const g = 26 * dpr;
    for (let x = ox % g; x < W; x += g) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }
    for (let y = oy % g; y < H; y += g) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    // contour rings
    const cx = W * 0.5 + ox;
    const cy = H * 0.5 + oy;
    for (let i = 1; i <= 6; i += 1) {
      ctx.beginPath();
      for (let a = 0; a <= 64; a += 1) {
        const ang = (a / 64) * 6.283;
        const r = Math.min(W, H) * (0.07 + i * 0.055) * (1 + Math.sin(ang * 3 + i) * 0.1);
        const x = cx + Math.cos(ang) * r * 1.5;
        const y = cy + Math.sin(ang) * r * 0.85;
        if (a) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = o.onDark
        ? `rgba(196,164,85,${0.3 - i * 0.03})`
        : `rgba(43,37,33,${0.2 - i * 0.022})`;
      ctx.lineWidth = dpr * (i % 3 === 0 ? 1.2 : 0.7);
      ctx.stroke();
    }

    // border traverse + pillar marks
    ctx.beginPath();
    pillars.forEach((p, i) => {
      const x = p.u * W + ox;
      const y = p.v * H + oy;
      if (i) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
    });
    ctx.strokeStyle = o.onDark ? 'rgba(239,234,221,0.85)' : 'rgba(43,37,33,0.7)';
    ctx.lineWidth = dpr * 1.6;
    ctx.setLineDash([9 * dpr, 5 * dpr, 2 * dpr, 5 * dpr]);
    ctx.stroke();
    ctx.setLineDash([]);

    pillars.forEach((p, i) => {
      const x = p.u * W + ox;
      const y = p.v * H + oy;
      ctx.strokeStyle = o.onDark ? 'rgba(196,164,85,0.95)' : 'rgba(169,129,47,0.95)';
      ctx.lineWidth = dpr;
      ctx.strokeRect(x - 3 * dpr, y - 3 * dpr, 6 * dpr, 6 * dpr);
      if (i % 2 === 0) {
        ctx.fillStyle = o.onDark ? 'rgba(239,234,221,0.55)' : 'rgba(43,37,33,0.5)';
        ctx.font = `${8.5 * dpr}px "IBM Plex Mono", monospace`;
        ctx.fillText(`P${38 + i}`, x + 6 * dpr, y - 6 * dpr);
      }
    });

    // sweep
    const ang = t * 0.00035;
    const R = Math.min(W, H) * 0.46;
    ctx.save();
    ctx.translate(cx, cy);
    const grd = ctx.createLinearGradient(0, 0, Math.cos(ang) * R, Math.sin(ang) * R * 0.6);
    grd.addColorStop(0, o.onDark ? 'rgba(196,164,85,0.35)' : 'rgba(31,111,122,0.28)');
    grd.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.ellipse(0, 0, R, R * 0.6, 0, ang - 0.22, ang);
    ctx.closePath();
    ctx.fillStyle = grd;
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(Math.cos(ang) * R, Math.sin(ang) * R * 0.6);
    ctx.strokeStyle = o.onDark ? 'rgba(196,164,85,0.8)' : 'rgba(31,111,122,0.7)';
    ctx.lineWidth = dpr;
    ctx.stroke();
    ctx.restore();

    // bearing ticks
    for (let i = 0; i < 36; i += 1) {
      const a = (i / 36) * 6.283;
      const r1 = Math.min(W, H) * 0.44;
      const r2 = r1 + (i % 9 === 0 ? 9 : 4) * dpr;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * r1 * 1.5, cy + Math.sin(a) * r1 * 0.85);
      ctx.lineTo(cx + Math.cos(a) * r2 * 1.5, cy + Math.sin(a) * r2 * 0.85);
      ctx.strokeStyle = o.onDark ? 'rgba(239,234,221,0.28)' : 'rgba(43,37,33,0.24)';
      ctx.lineWidth = dpr * 0.8;
      ctx.stroke();
    }
  };
});

export default plan;
