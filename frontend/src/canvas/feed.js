import { createScene } from './core';

/**
 * Simulated CCTV feed: layered terrain, grain, scanlines and tracked bounding
 * boxes whose confidence collapses as the frame degrades.
 *
 * Options: { seed, ir, boxes, labels, getDegrade }.
 */
export const feed = createScene((S, optionsRef) => {
  const initial = optionsRef.current || {};
  const seed = initial.seed || 0;
  const count = initial.boxes == null ? 2 : initial.boxes;

  const boxes = [];
  for (let i = 0; i < count; i += 1) {
    boxes.push({
      x: Math.random(),
      y: 0.54 + Math.random() * 0.24,
      w: 0.05 + Math.random() * 0.06,
      sp: (Math.random() > 0.5 ? 1 : -1) * (0.00006 + Math.random() * 0.00008),
      id: `T${100 + Math.floor(Math.random() * 800)}`,
      veh: Math.random() > 0.6,
    });
  }

  return (t0) => {
    const { ctx, W, H, dpr } = S;
    if (!W) return;
    const o = optionsRef.current || {};
    const ir = !!o.ir;
    const t = t0 + seed * 100;
    const deg = o.getDegrade ? o.getDegrade() / 100 : 0;

    const sky = ctx.createLinearGradient(0, 0, 0, H);
    if (ir) {
      sky.addColorStop(0, '#1D2A2C');
      sky.addColorStop(0.55, '#2A3C3E');
      sky.addColorStop(1, '#141D1F');
    } else {
      sky.addColorStop(0, '#6B6455');
      sky.addColorStop(0.45, '#938B77');
      sky.addColorStop(1, '#413B31');
    }
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, W, H);

    for (let L = 0; L < 3; L += 1) {
      ctx.beginPath();
      ctx.moveTo(0, H);
      for (let x = 0; x <= W; x += W / 36) {
        const u = x / W;
        ctx.lineTo(
          x,
          H * (0.44 + L * 0.085) -
            Math.sin(u * (5 + L * 3) + seed + L) * H * 0.05 -
            Math.sin(u * 13 + L) * H * 0.016,
        );
      }
      ctx.lineTo(W, H);
      ctx.closePath();
      ctx.fillStyle = ir ? `rgba(14,24,26,${0.45 + L * 0.2})` : `rgba(38,33,27,${0.3 + L * 0.22})`;
      ctx.fill();
    }

    ctx.fillStyle = ir ? 'rgba(11,19,21,0.9)' : 'rgba(33,28,23,0.85)';
    ctx.fillRect(0, H * 0.74, W, H * 0.26);

    if (deg > 0) {
      ctx.fillStyle = ir ? `rgba(200,208,208,${deg * 0.4})` : `rgba(228,221,206,${deg * 0.48})`;
      ctx.fillRect(0, 0, W, H);
    }

    ctx.globalAlpha = 0.05 + deg * 0.08;
    for (let i = 0; i < 110; i += 1) {
      ctx.fillStyle = i % 2 ? '#fff' : '#000';
      ctx.fillRect(Math.random() * W, Math.random() * H, 1.5 * dpr, 1.5 * dpr);
    }
    ctx.globalAlpha = 1;

    ctx.fillStyle = 'rgba(239,234,221,0.035)';
    for (let y = 0; y < H; y += 3 * dpr) ctx.fillRect(0, y, W, dpr);

    boxes.forEach((b, i) => {
      b.x += b.sp * 16;
      if (b.x > 1.05) b.x = -0.05;
      if (b.x < -0.05) b.x = 1.05;

      const bw = b.w * W * (b.veh ? 1.7 : 1);
      const bh = b.w * W * (b.veh ? 1.05 : 2.05);
      const bx = b.x * W;
      const by = b.y * H - bh;
      const conf = Math.max(24, Math.round(94 - deg * 66 + Math.sin(t * 0.002 + i) * 2));
      const faded = deg > 0.62;
      const col = faded ? '198,192,176' : b.veh ? '196,164,85' : '110,168,178';

      ctx.strokeStyle = `rgba(${col},${faded ? 0.45 : 0.95})`;
      ctx.lineWidth = dpr;
      ctx.setLineDash(faded ? [3 * dpr, 3 * dpr] : []);
      ctx.strokeRect(bx, by, bw, bh);
      ctx.setLineDash([]);

      ctx.lineWidth = 2 * dpr;
      const k = Math.min(bw, bh) * 0.26;
      [
        [bx, by, 1, 1],
        [bx + bw, by, -1, 1],
        [bx, by + bh, 1, -1],
        [bx + bw, by + bh, -1, -1],
      ].forEach((c) => {
        ctx.beginPath();
        ctx.moveTo(c[0] + c[2] * k, c[1]);
        ctx.lineTo(c[0], c[1]);
        ctx.lineTo(c[0], c[1] + c[3] * k);
        ctx.stroke();
      });

      if (o.labels !== false) {
        const lab = `${b.id} · ${conf}%`;
        ctx.font = `${9 * dpr}px "IBM Plex Mono", monospace`;
        const tw = ctx.measureText(lab).width + 9 * dpr;
        ctx.fillStyle = `rgba(${col},0.92)`;
        ctx.fillRect(bx, by - 13 * dpr, tw, 12 * dpr);
        ctx.fillStyle = '#241F1B';
        ctx.fillText(lab, bx + 4.5 * dpr, by - 4 * dpr);
      }
    });
  };
});

export default feed;
