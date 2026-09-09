import { pointer } from '../hooks/usePointer';
import { createScene } from './core';

/**
 * Orthographic wireframe of the coverage volume above a post: ground grid,
 * camera frustums, tracked targets and a scan plane sweeping 0–40 m.
 *
 * Kept from the original prototype as an alternative hero backdrop — the hero
 * currently mounts `cine`. Swap the scene in Hero.jsx to use it.
 */
export const volume = createScene((S) => {
  const rigs = [
    { p: [-0.66, 0.44, -0.3], aim: [0.3, 0.65], tag: 'MAST 38' },
    { p: [-0.14, 0.52, 0.22], aim: [-0.34, 0.6], tag: 'MAST 41' },
    { p: [0.42, 0.46, -0.24], aim: [0.28, 0.62], tag: 'MAST 42' },
    { p: [0.82, 0.5, 0.28], aim: [-0.3, 0.58], tag: 'MAST 46' },
  ];

  const tracks = [];
  for (let i = 0; i < 9; i += 1) {
    tracks.push({
      x: -1 + Math.random() * 2,
      z: -1 + Math.random() * 2,
      y: 0.03 + Math.random() * 0.05,
      sp: (Math.random() > 0.5 ? 1 : -1) * (0.00007 + Math.random() * 0.00012),
      ax: Math.random() > 0.5,
      id: `T${100 + Math.floor(Math.random() * 800)}`,
    });
  }

  return (t) => {
    const { ctx, W, H, dpr } = S;
    if (!W) return;
    ctx.clearRect(0, 0, W, H);

    const yaw = t * 0.00013 + pointer.x * 0.5;
    const pitch = 0.5 + pointer.y * 0.16;
    const sc = Math.min(W / 2.7, H / 1.9);
    const cx = W * 0.6;
    const cy = H * 0.56;
    const cyw = Math.cos(yaw);
    const syw = Math.sin(yaw);
    const cp = Math.cos(pitch);
    const sp = Math.sin(pitch);

    const P = (x, y, z) => {
      const rx = x * cyw - z * syw;
      const rz = x * syw + z * cyw;
      const ry = y * cp - rz * sp;
      const rz2 = y * sp + rz * cp;
      const k = 1 / (2.35 + rz2 * 0.55);
      return [cx + rx * sc * k * 2.35, cy - ry * sc * k * 2.35, rz2];
    };
    const line = (a, b, col, w, dash) => {
      const p1 = P(a[0], a[1], a[2]);
      const p2 = P(b[0], b[1], b[2]);
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.strokeStyle = col;
      ctx.lineWidth = (w || 1) * dpr;
      if (dash) ctx.setLineDash(dash.map((v) => v * dpr));
      ctx.stroke();
      ctx.setLineDash([]);
    };

    // ground grid
    const N = 12;
    for (let i = 0; i <= N; i += 1) {
      const u = -1 + (2 * i) / N;
      const major = i % 3 === 0;
      const c = `rgba(43,37,33,${major ? 0.3 : 0.14})`;
      line([u, 0, -1], [u, 0, 1], c, major ? 0.9 : 0.6);
      line([-1, 0, u], [1, 0, u], c, major ? 0.9 : 0.6);
    }

    // coverage volume box
    const top = 0.92;
    const corners = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ];
    corners.forEach((c1, i) => {
      const c2 = corners[(i + 1) % 4];
      line([c1[0], 0, c1[1]], [c2[0], 0, c2[1]], 'rgba(43,37,33,0.55)', 1.2);
      line([c1[0], top, c1[1]], [c2[0], top, c2[1]], 'rgba(43,37,33,0.22)', 0.8, [6, 5]);
      line([c1[0], 0, c1[1]], [c1[0], top, c1[1]], 'rgba(43,37,33,0.28)', 0.8, [6, 5]);
    });

    // border traverse on the floor
    ctx.beginPath();
    for (let i = 0; i <= 40; i += 1) {
      const u = -1 + (2 * i) / 40;
      const p = P(u, 0.004, Math.sin(u * 2.4) * 0.3);
      if (i) ctx.lineTo(p[0], p[1]);
      else ctx.moveTo(p[0], p[1]);
    }
    ctx.strokeStyle = 'rgba(169,129,47,0.95)';
    ctx.lineWidth = 1.8 * dpr;
    ctx.setLineDash([10 * dpr, 5 * dpr, 2 * dpr, 5 * dpr]);
    ctx.stroke();
    ctx.setLineDash([]);

    // sweeping scan plane
    const sh = (Math.sin(t * 0.00035) * 0.5 + 0.5) * top * 0.85 + 0.04;
    const q = [P(-1, sh, -1), P(1, sh, -1), P(1, sh, 1), P(-1, sh, 1)];
    ctx.beginPath();
    ctx.moveTo(q[0][0], q[0][1]);
    for (let i = 1; i < 4; i += 1) ctx.lineTo(q[i][0], q[i][1]);
    ctx.closePath();
    ctx.fillStyle = 'rgba(31,111,122,0.10)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(31,111,122,0.75)';
    ctx.lineWidth = 1.2 * dpr;
    ctx.stroke();
    for (let i = 1; i < 6; i += 1) {
      const u = -1 + (2 * i) / 6;
      line([u, sh, -1], [u, sh, 1], 'rgba(31,111,122,0.22)', 0.6);
    }
    ctx.font = `${9.5 * dpr}px "IBM Plex Mono", monospace`;
    ctx.fillStyle = 'rgba(31,111,122,0.95)';
    ctx.fillText(`SCAN ${(sh * 44).toFixed(1)} m`, q[3][0] + 6 * dpr, q[3][1] - 6 * dpr);

    // camera frustums
    rigs.forEach((r, ri) => {
      const a = r.p;
      const reach = 1.05;
      const fx = a[0] + r.aim[0] * reach;
      const fz = a[2] + r.aim[1] * reach;
      const half = 0.3;
      const foot = [
        [fx - half, 0, fz - half * 0.6],
        [fx + half, 0, fz - half * 0.6],
        [fx + half, 0, fz + half * 0.6],
        [fx - half, 0, fz + half * 0.6],
      ];

      let hot = -1;
      tracks.forEach((tr, ti) => {
        if (Math.abs(tr.x - fx) < half && Math.abs(tr.z - fz) < half * 0.6) hot = ti;
      });
      const base = hot >= 0 ? '181,115,36' : '62,81,105';

      ctx.beginPath();
      const pa = P(a[0], a[1], a[2]);
      foot.forEach((c) => {
        const pc = P(c[0], c[1], c[2]);
        ctx.moveTo(pa[0], pa[1]);
        ctx.lineTo(pc[0], pc[1]);
      });
      ctx.strokeStyle = `rgba(${base},0.5)`;
      ctx.lineWidth = 0.8 * dpr;
      ctx.stroke();

      ctx.beginPath();
      foot.forEach((c, i) => {
        const pc = P(c[0], 0.002, c[2]);
        if (i) ctx.lineTo(pc[0], pc[1]);
        else ctx.moveTo(pc[0], pc[1]);
      });
      ctx.closePath();
      ctx.fillStyle = `rgba(${base},${hot >= 0 ? 0.2 : 0.09})`;
      ctx.fill();
      ctx.strokeStyle = `rgba(${base},0.85)`;
      ctx.lineWidth = 1.1 * dpr;
      ctx.stroke();

      line([a[0], 0, a[2]], [a[0], a[1], a[2]], 'rgba(43,37,33,0.7)', 1.3);
      for (let k = 1; k < 4; k += 1) {
        const y1 = (a[1] * k) / 4;
        const y2 = (a[1] * (k + 0.5)) / 4;
        line([a[0] - 0.035, y1, a[2]], [a[0] + 0.035, y2, a[2]], 'rgba(43,37,33,0.4)', 0.7);
      }

      ctx.fillStyle = hot >= 0 ? '#B57324' : '#3E5169';
      ctx.fillRect(pa[0] - 4 * dpr, pa[1] - 4 * dpr, 8 * dpr, 8 * dpr);
      ctx.strokeStyle = 'rgba(43,37,33,0.85)';
      ctx.lineWidth = dpr;
      ctx.strokeRect(pa[0] - 4 * dpr, pa[1] - 4 * dpr, 8 * dpr, 8 * dpr);
      ctx.font = `${9 * dpr}px "IBM Plex Mono", monospace`;
      ctx.fillStyle = 'rgba(43,37,33,0.6)';
      ctx.fillText(r.tag, pa[0] - 14 * dpr, pa[1] - 11 * dpr);

      if (hot >= 0) {
        const tr = tracks[hot];
        const tp = P(tr.x, tr.y, tr.z);
        const lw = 168 * dpr;
        const lh = 32 * dpr;
        const lx = tp[0] + 14 * dpr;
        const ly = tp[1] - lh - 12 * dpr;
        ctx.fillStyle = 'rgba(250,247,240,0.96)';
        ctx.fillRect(lx, ly, lw, lh);
        ctx.strokeStyle = 'rgba(181,115,36,0.95)';
        ctx.lineWidth = dpr;
        ctx.strokeRect(lx, ly, lw, lh);
        ctx.fillStyle = '#7A4E14';
        ctx.font = `${10 * dpr}px "IBM Plex Mono", monospace`;
        ctx.fillText(`PROVISIONAL · 31 ms · ${tr.id}`, lx + 8 * dpr, ly + 13 * dpr);
        ctx.fillStyle = 'rgba(43,37,33,0.62)';
        ctx.fillText(
          ri % 2 ? 'MOTION + APPEARANCE AGREE' : 'BOTH CHANNELS AGREE',
          lx + 8 * dpr,
          ly + 25 * dpr,
        );
        ctx.strokeStyle = 'rgba(181,115,36,0.7)';
        ctx.beginPath();
        ctx.moveTo(tp[0] + 5 * dpr, tp[1] - 5 * dpr);
        ctx.lineTo(lx, ly + lh);
        ctx.stroke();
      }
    });

    // tracks
    tracks.forEach((tr) => {
      tr.x += tr.sp * 16;
      if (tr.x > 1) tr.x = -1;
      if (tr.x < -1) tr.x = 1;
      const g = P(tr.x, 0.002, tr.z);
      const h = P(tr.x, tr.y + 0.1, tr.z);
      line([tr.x, 0, tr.z], [tr.x, tr.y + 0.1, tr.z], 'rgba(43,37,33,0.2)', 0.7);
      ctx.strokeStyle = tr.ax ? 'rgba(62,81,105,0.9)' : 'rgba(31,111,122,0.9)';
      ctx.lineWidth = 1.1 * dpr;
      ctx.beginPath();
      ctx.arc(g[0], g[1], 3.4 * dpr, 0, 6.3);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(g[0] - 7 * dpr, g[1]);
      ctx.lineTo(g[0] - 4.6 * dpr, g[1]);
      ctx.moveTo(g[0] + 4.6 * dpr, g[1]);
      ctx.lineTo(g[0] + 7 * dpr, g[1]);
      ctx.stroke();
      if (Math.abs(sh - (tr.y + 0.1)) < 0.05) {
        ctx.fillStyle = 'rgba(163,46,36,0.9)';
        ctx.fillRect(h[0] - 3 * dpr, h[1] - 3 * dpr, 6 * dpr, 6 * dpr);
        ctx.strokeStyle = 'rgba(163,46,36,0.5)';
        ctx.beginPath();
        ctx.arc(h[0], h[1], 11 * dpr, 0, 6.3);
        ctx.stroke();
      }
    });

    // corner registration ticks
    ctx.strokeStyle = 'rgba(43,37,33,0.42)';
    ctx.lineWidth = dpr;
    const m = 22 * dpr;
    const L = 13 * dpr;
    [
      [m, m, 1, 1],
      [W - m, m, -1, 1],
      [m, H - m, 1, -1],
      [W - m, H - m, -1, -1],
    ].forEach((c) => {
      ctx.beginPath();
      ctx.moveTo(c[0] + c[2] * L, c[1]);
      ctx.lineTo(c[0], c[1]);
      ctx.lineTo(c[0], c[1] + c[3] * L);
      ctx.stroke();
    });
  };
});

export default volume;
