import { pointer } from '../hooks/usePointer';
import { createScene, pad2 } from './core';

/**
 * The hero's cinematic camera feed: a perspective road scene that cycles
 * day → dusk → IR night, tracks actors with bracket overlays, and runs the
 * two-stage alert banner (provisional → confirmed) on the nearest vehicle.
 */
export const cine = createScene((S) => {
  const mkActor = (kind, dir) => {
    const lane =
      kind === 'ped'
        ? Math.random() > 0.5
          ? -5.4 - Math.random() * 1.6
          : 5.0 + Math.random() * 1.8
        : dir > 0
          ? -2.1
          : 2.1;
    return {
      kind,
      dir,
      x: lane,
      z: dir > 0 ? 78 + Math.random() * 40 : 6 + Math.random() * 30,
      sp:
        kind === 'ped'
          ? 0.0016 + Math.random() * 0.0012
          : (kind === 'truck' ? 0.0045 : 0.0075) + Math.random() * 0.003,
      w: kind === 'ped' ? 0.58 : kind === 'truck' ? 2.5 : kind === 'bike' ? 0.8 : 1.95,
      h: kind === 'ped' ? 1.72 : kind === 'truck' ? 3.1 : kind === 'bike' ? 1.5 : 1.62,
      len: kind === 'ped' ? 0.5 : kind === 'truck' ? 7.4 : kind === 'bike' ? 1.9 : 4.4,
      id: `T${100 + Math.floor(Math.random() * 890)}`,
      ph: Math.random() * 6.3,
      loaded: Math.random() > 0.55,
      conf: 0,
    };
  };

  const actors = [];
  ['truck', 'car', 'bike', 'ped', 'ped', 'car', 'ped'].forEach((k, i) =>
    actors.push(mkActor(k, i % 2 ? 1 : -1)),
  );

  const trees = [];
  for (let i = 0; i < 26; i += 1) {
    trees.push({
      x: (Math.random() > 0.5 ? 1 : -1) * (7.5 + Math.random() * 9),
      z: 6 + Math.random() * 86,
      h: 4 + Math.random() * 5.5,
      w: 1.4 + Math.random() * 2,
    });
  }

  let focusIdx = -1;
  let focusT = 0;
  let motes = null;

  return (t) => {
    const { ctx, W, H, dpr } = S;
    if (!W) return;

    // time-of-day cycle
    const cyc = (t % 46000) / 46000;
    const night = cyc > 0.52 && cyc < 0.92;
    const dusk = (cyc > 0.4 && cyc <= 0.52) || (cyc >= 0.92 && cyc < 0.99);

    const mx = pointer.x;
    const my = pointer.y;
    const panv = Math.sin(t * 0.00013) * 2.4 + mx * 1.6;
    const f = W * (0.78 + Math.sin(t * 0.00009) * 0.05);
    const shake = Math.sin(t * 0.013) * 0.5 * dpr + Math.sin(t * 0.031) * 0.3 * dpr;
    const camH = 6.4;
    const hy = H * (0.395 + my * 0.02) + shake;

    const P = (x, z, y) => {
      const zz = Math.max(1.2, z);
      return [W / 2 + ((x - panv) * f) / zz + shake, hy + ((camH - (y || 0)) * f) / zz];
    };
    const sz = (m, z) => (m * f) / Math.max(1.2, z);

    // sky
    const sky = ctx.createLinearGradient(0, 0, 0, hy + 4);
    if (night) {
      sky.addColorStop(0, '#0E1518');
      sky.addColorStop(0.7, '#16211F');
      sky.addColorStop(1, '#1D2A26');
    } else if (dusk) {
      sky.addColorStop(0, '#4A4038');
      sky.addColorStop(0.55, '#8A6A4A');
      sky.addColorStop(1, '#B58A5E');
    } else {
      sky.addColorStop(0, '#7E8B90');
      sky.addColorStop(0.6, '#A9B0AC');
      sky.addColorStop(1, '#C6C4B4');
    }
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, W, hy + 6);

    if (!night) {
      for (let i = 0; i < 4; i += 1) {
        const cw = W * (0.16 + i * 0.03);
        const cxp = ((((t * 0.0000045 + i * 0.27) % 1.3) - 0.15)) * W;
        const cyp = hy * (0.14 + i * 0.11);
        const cg = ctx.createRadialGradient(cxp, cyp, 1, cxp, cyp, cw * 0.6);
        cg.addColorStop(0, dusk ? 'rgba(210,180,150,0.28)' : 'rgba(255,255,250,0.32)');
        cg.addColorStop(1, 'rgba(255,255,250,0)');
        ctx.fillStyle = cg;
        ctx.beginPath();
        ctx.ellipse(cxp, cyp, cw * 0.6, cw * 0.16, 0, 0, 6.3);
        ctx.fill();
      }
    }

    // sun / moon glare
    const sunX = W * (0.72 - panv * 0.01);
    const sunY = hy * (night ? 0.12 : 0.22);
    const sg2 = ctx.createRadialGradient(sunX, sunY, 1, sunX, sunY, W * 0.26);
    sg2.addColorStop(
      0,
      night ? 'rgba(200,225,215,0.14)' : dusk ? 'rgba(255,210,150,0.32)' : 'rgba(255,250,230,0.24)',
    );
    sg2.addColorStop(1, 'rgba(255,250,230,0)');
    ctx.fillStyle = sg2;
    ctx.beginPath();
    ctx.arc(sunX, sunY, W * 0.26, 0, 6.3);
    ctx.fill();

    // distant ridge lines
    for (let L = 0; L < 2; L += 1) {
      ctx.beginPath();
      ctx.moveTo(0, hy + 2);
      for (let x = 0; x <= W; x += W / 44) {
        const u = x / W + panv * 0.006;
        ctx.lineTo(
          x,
          hy - Math.abs(Math.sin(u * (3.1 + L * 2.2) + L)) * H * (0.075 - L * 0.028) - L * 3 * dpr,
        );
      }
      ctx.lineTo(W, hy + 2);
      ctx.closePath();
      ctx.fillStyle = night
        ? `rgba(12,20,20,${0.75 + L * 0.2})`
        : dusk
          ? `rgba(58,44,36,${0.6 + L * 0.25})`
          : `rgba(96,100,92,${0.45 + L * 0.3})`;
      ctx.fill();
    }

    // ground
    const gnd = ctx.createLinearGradient(0, hy, 0, H);
    if (night) {
      gnd.addColorStop(0, '#1A241F');
      gnd.addColorStop(1, '#0C1310');
    } else if (dusk) {
      gnd.addColorStop(0, '#5A4A3A');
      gnd.addColorStop(1, '#2E271F');
    } else {
      gnd.addColorStop(0, '#8E8770');
      gnd.addColorStop(1, '#4A4437');
    }
    ctx.fillStyle = gnd;
    ctx.fillRect(0, hy, W, H - hy);

    // carriageway
    const quad = (a, b, c, d, fill) => {
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.lineTo(c[0], c[1]);
      ctx.lineTo(d[0], d[1]);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    };
    quad(
      P(-4.6, 96),
      P(4.6, 96),
      P(4.6, 3),
      P(-4.6, 3),
      night ? '#111A18' : dusk ? '#3A322A' : '#6E6A5C',
    );
    for (let z = 4; z < 92; z += 6) {
      quad(
        P(-0.09, z),
        P(0.09, z),
        P(0.09, z + 3),
        P(-0.09, z + 3),
        night ? 'rgba(160,190,180,0.30)' : 'rgba(232,226,208,0.70)',
      );
    }

    // border line + pillar 42
    const bl1 = P(-9, 26);
    const bl2 = P(9, 26);
    ctx.beginPath();
    ctx.moveTo(bl1[0], bl1[1]);
    ctx.lineTo(bl2[0], bl2[1]);
    ctx.strokeStyle = 'rgba(196,164,85,0.85)';
    ctx.lineWidth = Math.max(1, sz(0.28, 26));
    ctx.setLineDash([sz(1.6, 26), sz(1.1, 26)]);
    ctx.stroke();
    ctx.setLineDash([]);

    const pb = P(-5.2, 26);
    const pt = P(-5.2, 26, 2.1);
    ctx.fillStyle = night ? '#4A5A52' : '#D8D2BC';
    ctx.fillRect(pb[0] - sz(0.28, 26), pt[1], sz(0.56, 26), pb[1] - pt[1]);
    ctx.strokeStyle = 'rgba(43,37,33,0.7)';
    ctx.lineWidth = dpr;
    ctx.strokeRect(pb[0] - sz(0.28, 26), pt[1], sz(0.56, 26), pb[1] - pt[1]);
    ctx.font = `${Math.max(8 * dpr, sz(0.5, 26))}px "IBM Plex Mono", monospace`;
    ctx.fillStyle = night ? 'rgba(200,220,210,0.8)' : 'rgba(43,37,33,0.8)';
    ctx.fillText('P42', pb[0] + sz(0.5, 26), pt[1] - 4 * dpr);

    // scrub, far to near
    trees
      .sort((a, b) => b.z - a.z)
      .forEach((tr) => {
        const b = P(tr.x, tr.z);
        const tp = P(tr.x, tr.z, tr.h);
        const w = sz(tr.w, tr.z);
        ctx.beginPath();
        ctx.moveTo(b[0], b[1]);
        ctx.lineTo(b[0] - w / 2, b[1] - (b[1] - tp[1]) * 0.55);
        ctx.lineTo(b[0], tp[1]);
        ctx.lineTo(b[0] + w / 2, b[1] - (b[1] - tp[1]) * 0.55);
        ctx.closePath();
        ctx.fillStyle = night
          ? 'rgba(10,20,17,0.9)'
          : dusk
            ? 'rgba(40,34,26,0.85)'
            : 'rgba(58,62,48,0.8)';
        ctx.fill();
      });

    // actors
    actors.sort((a, b) => b.z - a.z);
    let bestIdx = -1;
    let bestZ = 999;
    actors.forEach((a) => {
      a.z -= a.dir * a.sp * 16;
      if (a.z < 3) {
        Object.assign(a, mkActor(a.kind, 1));
        a.z = 96;
      }
      if (a.z > 100) {
        Object.assign(a, mkActor(a.kind, -1));
        a.z = 5;
      }
      if (a.kind === 'ped') a.x += Math.sin(t * 0.0004 + a.ph) * 0.004;

      const zf = a.z + a.len;
      const bn = P(a.x, a.z);
      const bf = P(a.x, zf);
      const wn = sz(a.w, a.z);
      const wf = sz(a.w, zf);
      const hn = sz(a.h, a.z);
      const hf = sz(a.h, zf);
      const body = night
        ? 'rgba(150,180,168,0.55)'
        : a.kind === 'ped'
          ? 'rgba(52,48,40,0.92)'
          : a.loaded
            ? 'rgba(90,78,56,0.95)'
            : 'rgba(120,116,102,0.95)';

      // top and side faces give the box some solidity
      ctx.beginPath();
      ctx.moveTo(bn[0] - wn / 2, bn[1] - hn);
      ctx.lineTo(bf[0] - wf / 2, bf[1] - hf);
      ctx.lineTo(bf[0] + wf / 2, bf[1] - hf);
      ctx.lineTo(bn[0] + wn / 2, bn[1] - hn);
      ctx.closePath();
      ctx.fillStyle = night ? 'rgba(120,150,140,0.4)' : 'rgba(0,0,0,0.18)';
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(bf[0] - wf / 2, bf[1] - hf);
      ctx.lineTo(bf[0] - wf / 2, bf[1]);
      ctx.lineTo(bn[0] - wn / 2, bn[1]);
      ctx.lineTo(bn[0] - wn / 2, bn[1] - hn);
      ctx.closePath();
      ctx.fillStyle = night ? 'rgba(110,140,130,0.5)' : 'rgba(0,0,0,0.3)';
      ctx.fill();

      if (a.kind === 'ped') {
        const bob = Math.sin(t * 0.006 + a.ph) * hn * 0.03;
        ctx.fillStyle = body;
        ctx.fillRect(bn[0] - wn / 2, bn[1] - hn + bob, wn, hn * 0.62);
        const legS = Math.sin(t * 0.008 + a.ph) * wn * 0.55;
        ctx.fillRect(bn[0] - wn * 0.18 + legS * 0.4, bn[1] - hn * 0.42 + bob, wn * 0.22, hn * 0.42);
        ctx.fillRect(bn[0] - wn * 0.04 - legS * 0.4, bn[1] - hn * 0.42 + bob, wn * 0.22, hn * 0.42);
        ctx.beginPath();
        ctx.arc(bn[0], bn[1] - hn * 1.02 + bob, wn * 0.3, 0, 6.3);
        ctx.fill();
      } else {
        ctx.fillStyle = body;
        const rr = Math.min(wn, hn) * 0.14;
        ctx.beginPath();
        ctx.moveTo(bn[0] - wn / 2, bn[1]);
        ctx.lineTo(bn[0] - wn / 2, bn[1] - hn + rr);
        ctx.quadraticCurveTo(bn[0] - wn / 2, bn[1] - hn, bn[0] - wn / 2 + rr, bn[1] - hn);
        ctx.lineTo(bn[0] + wn / 2 - rr, bn[1] - hn);
        ctx.quadraticCurveTo(bn[0] + wn / 2, bn[1] - hn, bn[0] + wn / 2, bn[1] - hn + rr);
        ctx.lineTo(bn[0] + wn / 2, bn[1]);
        ctx.closePath();
        ctx.fill();

        ctx.fillStyle = night ? 'rgba(190,215,205,0.35)' : 'rgba(180,190,190,0.55)';
        ctx.fillRect(bn[0] - wn * 0.38, bn[1] - hn * 0.92, wn * 0.76, hn * 0.34);
        ctx.fillStyle = 'rgba(20,18,16,0.9)';
        ctx.fillRect(bn[0] - wn * 0.46, bn[1] - hn * 0.14, wn * 0.2, hn * 0.16);
        ctx.fillRect(bn[0] + wn * 0.26, bn[1] - hn * 0.14, wn * 0.2, hn * 0.16);

        if (night || dusk) {
          const lg = ctx.createRadialGradient(
            bn[0],
            bn[1] - hn * 0.5,
            1,
            bn[0],
            bn[1] - hn * 0.5,
            wn * 2.6,
          );
          lg.addColorStop(0, `rgba(255,238,190,${night ? 0.5 : 0.25})`);
          lg.addColorStop(1, 'rgba(255,238,190,0)');
          ctx.fillStyle = lg;
          ctx.beginPath();
          ctx.arc(bn[0], bn[1] - hn * 0.5, wn * 2.6, 0, 6.3);
          ctx.fill();
          ctx.fillStyle = 'rgba(255,246,214,0.95)';
          ctx.fillRect(bn[0] - wn * 0.42, bn[1] - hn * 0.42, wn * 0.16, hn * 0.1);
          ctx.fillRect(bn[0] + wn * 0.26, bn[1] - hn * 0.42, wn * 0.16, hn * 0.1);
        }

        ctx.fillStyle = 'rgba(0,0,0,0.22)';
        ctx.beginPath();
        ctx.ellipse(bn[0], bn[1], wn * 0.6, sz(0.35, a.z), 0, 0, 6.3);
        ctx.fill();
      }

      // detection bracket; confidence ramps in as the actor enters the frame
      const inFrame = a.z > 8 && a.z < 62 && bn[0] > W * 0.04 && bn[0] < W * 0.96;
      a.conf += ((inFrame ? (night ? 82 : 94) - Math.min(28, a.z * 0.32) : 0) - a.conf) * 0.05;
      if (a.conf > 22) {
        const col = a.kind === 'ped' ? '110,168,178' : '196,164,85';
        const bx = bn[0] - wn / 2 - 3 * dpr;
        const by = bn[1] - hn - 3 * dpr;
        const bw = wn + 6 * dpr;
        const bh = hn + 6 * dpr;
        ctx.strokeStyle = `rgba(${col},${Math.min(0.95, a.conf / 90)})`;
        ctx.lineWidth = 1.2 * dpr;
        const k = Math.min(bw, bh) * 0.28;
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
        if (bh > 26 * dpr) {
          const lab = `${a.id} · ${Math.round(a.conf)}%`;
          ctx.font = `${9 * dpr}px "IBM Plex Mono", monospace`;
          const tw = ctx.measureText(lab).width + 8 * dpr;
          ctx.fillStyle = `rgba(${col},0.9)`;
          ctx.fillRect(bx, by - 12 * dpr, tw, 12 * dpr);
          ctx.fillStyle = '#1C1815';
          ctx.fillText(lab, bx + 4 * dpr, by - 3.5 * dpr);
        }
        if (a.z < bestZ && a.kind !== 'ped') {
          bestZ = a.z;
          bestIdx = actors.indexOf(a);
        }
      }
    });

    // lock-on: provisional → confirmed banner on the nearest vehicle
    if (bestIdx >= 0 && focusIdx < 0 && Math.random() < 0.02) {
      focusIdx = bestIdx;
      focusT = t;
    }
    if (focusIdx >= 0) {
      const el = t - focusT;
      if (el > 7200) {
        focusIdx = -1;
      } else {
        const a = actors[focusIdx];
        if (a) {
          const bn = P(a.x, a.z);
          const wn = sz(a.w, a.z);
          const hn = sz(a.h, a.z);
          const stage = el < 2600 ? 0 : 1;
          const col = stage ? '163,46,36' : '181,115,36';
          ctx.strokeStyle = `rgba(${col},0.95)`;
          ctx.lineWidth = 1.6 * dpr;
          ctx.strokeRect(
            bn[0] - wn / 2 - 6 * dpr,
            bn[1] - hn - 6 * dpr,
            wn + 12 * dpr,
            hn + 12 * dpr,
          );
          const r = 8 * dpr + ((el % 900) / 900) * 16 * dpr;
          ctx.strokeStyle = `rgba(${col},${0.5 - ((el % 900) / 900) * 0.5})`;
          ctx.beginPath();
          ctx.arc(bn[0], bn[1] - hn / 2, r + Math.max(wn, hn) * 0.6, 0, 6.3);
          ctx.stroke();

          const bw2 = Math.min(W * 0.62, 330 * dpr);
          const bh2 = 34 * dpr;
          const bx2 = W * 0.5 - bw2 / 2;
          const by2 = H - 84 * dpr;
          ctx.fillStyle = 'rgba(28,24,21,0.86)';
          ctx.fillRect(bx2, by2, bw2, bh2);
          ctx.fillStyle = `rgba(${col},1)`;
          ctx.fillRect(bx2, by2, 3 * dpr, bh2);
          ctx.font = `${10.5 * dpr}px "IBM Plex Mono", monospace`;
          ctx.fillStyle = `rgba(${col},1)`;
          ctx.fillText(
            stage ? 'CONFIRMED · 3.1 s · 94%' : 'PROVISIONAL · 31 ms',
            bx2 + 11 * dpr,
            by2 + 14 * dpr,
          );
          ctx.fillStyle = 'rgba(236,230,216,0.92)';
          ctx.font = `${10 * dpr}px "IBM Plex Mono", monospace`;
          ctx.fillText(
            stage
              ? 'LOADED VEHICLE NORTHBOUND, OUTSIDE HOURS'
              : 'APPEARANCE ✓   MOTION ✓   AWAITING JUDGE',
            bx2 + 11 * dpr,
            by2 + 27 * dpr,
          );
        }
      }
    }

    // colour grade
    if (night) {
      ctx.globalCompositeOperation = 'saturation';
      ctx.fillStyle = 'rgba(120,190,170,0.55)';
      ctx.fillRect(0, 0, W, H);
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = 'rgba(24,60,52,0.18)';
      ctx.fillRect(0, 0, W, H);
    } else if (dusk) {
      ctx.fillStyle = 'rgba(150,96,44,0.10)';
      ctx.fillRect(0, 0, W, H);
    }

    // grain, scanlines, vignette
    ctx.globalAlpha = night ? 0.05 : 0.025;
    for (let i = 0; i < 90; i += 1) {
      ctx.fillStyle = i % 2 ? '#fff' : '#000';
      ctx.fillRect(Math.random() * W, Math.random() * H, 1.1 * dpr, 1.1 * dpr);
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = 'rgba(0,0,0,0.05)';
    for (let y = (t * 0.06) % (3 * dpr); y < H; y += 3 * dpr) ctx.fillRect(0, y, W, dpr);
    const vg = ctx.createRadialGradient(
      W / 2,
      H / 2,
      Math.min(W, H) * 0.3,
      W / 2,
      H / 2,
      Math.max(W, H) * 0.72,
    );
    vg.addColorStop(0, 'rgba(0,0,0,0)');
    vg.addColorStop(1, 'rgba(0,0,0,0.5)');
    ctx.fillStyle = vg;
    ctx.fillRect(0, 0, W, H);

    // HUD
    const p = 12 * dpr;
    ctx.font = `${10 * dpr}px "IBM Plex Mono", monospace`;
    const dt = new Date();
    const tc = `${pad2(night ? 2 : dusk ? 18 : 11)}:${pad2(dt.getMinutes())}:${pad2(
      dt.getSeconds(),
    )}:${pad2(Math.floor((t % 1000) / 40))}`;
    ctx.fillStyle = 'rgba(236,230,216,0.9)';
    ctx.fillText(tc, p, H - p - 2 * dpr);
    ctx.fillText(
      `${night ? 'IR ON · GAIN 18 dB' : dusk ? 'AUTO IRIS · f2.0' : 'DAY · f5.6'}   ZOOM ${(
        (f / W) *
        3.4
      ).toFixed(1)}x`,
      p,
      H - p - 16 * dpr,
    );
    ctx.fillStyle = `rgba(163,46,36,${0.4 + 0.6 * Math.abs(Math.sin(t * 0.002))})`;
    ctx.beginPath();
    ctx.arc(W - p - 42 * dpr, p + 6 * dpr, 3.6 * dpr, 0, 6.3);
    ctx.fill();
    ctx.fillStyle = 'rgba(236,230,216,0.9)';
    ctx.fillText('REC', W - p - 34 * dpr, p + 10 * dpr);

    // DORI strip
    const sw = W - p * 2;
    const sy2 = H - p - 30 * dpr;
    let cxp = p;
    [
      ['DETECT', 0.4, '62,81,105'],
      ['OBSERVE', 0.26, '31,111,122'],
      ['RECOGNISE', 0.2, '169,129,47'],
      ['IDENTIFY', 0.14, '163,46,36'],
    ].forEach((seg) => {
      const w2 = sw * seg[1];
      ctx.fillStyle = `rgba(${seg[2]},0.75)`;
      ctx.fillRect(cxp, sy2, w2 - 2 * dpr, 4 * dpr);
      cxp += w2;
    });

    // lens motes
    if (!motes) {
      motes = [];
      for (let i = 0; i < 40; i += 1) {
        motes.push({
          x: Math.random(),
          y: Math.random(),
          s: 0.4 + Math.random() * 1.1,
          sp: 0.00008 + Math.random() * 0.00014,
          dx: (Math.random() - 0.5) * 0.6,
        });
      }
    }
    motes.forEach((m) => {
      m.x += m.sp * 16 * m.dx * 2 + m.sp * 4;
      m.y -= m.sp * 16 * 0.3;
      if (m.y < -0.05) m.y = 1.05;
      if (m.x > 1.05) m.x = -0.05;
      if (m.x < -0.05) m.x = 1.05;
      ctx.fillStyle = night ? 'rgba(170,210,195,0.16)' : 'rgba(255,250,235,0.14)';
      ctx.beginPath();
      ctx.arc(m.x * W, m.y * H, m.s * dpr, 0, 6.3);
      ctx.fill();
    });

    // reticule
    ctx.strokeStyle = 'rgba(236,230,216,0.35)';
    ctx.lineWidth = dpr;
    ctx.beginPath();
    ctx.moveTo(W / 2 - 9 * dpr, H * 0.52);
    ctx.lineTo(W / 2 + 9 * dpr, H * 0.52);
    ctx.moveTo(W / 2, H * 0.52 - 9 * dpr);
    ctx.lineTo(W / 2, H * 0.52 + 9 * dpr);
    ctx.stroke();
  };
});

export default cine;
