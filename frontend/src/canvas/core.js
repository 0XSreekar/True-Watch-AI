import { prefersReducedMotion } from '../hooks/useReducedMotion';

/** Sizes a canvas to its box in device pixels and keeps it in sync. */
export const setupCanvas = (cv) => {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const state = { ctx: cv.getContext('2d'), dpr, W: 0, H: 0, dispose: () => {} };

  const fit = () => {
    state.W = cv.width = Math.max(1, Math.round(cv.clientWidth * dpr));
    state.H = cv.height = Math.max(1, Math.round(cv.clientHeight * dpr));
  };
  fit();

  try {
    const ro = new ResizeObserver(fit);
    ro.observe(cv);
    state.dispose = () => ro.disconnect();
  } catch {
    window.addEventListener('resize', fit);
    state.dispose = () => window.removeEventListener('resize', fit);
  }

  return state;
};

/**
 * Drives a scene at animation-frame cadence with the prototype's fixed 16 ms tick.
 * With reduced motion the scene is painted once and left static.
 */
export const runLoop = (draw) => {
  if (prefersReducedMotion()) {
    draw(0);
    return () => {};
  }

  let t = 0;
  let raf = 0;
  let stopped = false;

  const step = () => {
    if (stopped) return;
    t += 16;
    draw(t);
    raf = requestAnimationFrame(step);
  };
  raf = requestAnimationFrame(step);

  return () => {
    stopped = true;
    cancelAnimationFrame(raf);
  };
};

/** Wires setup + loop + teardown into the shape useCanvasScene expects. */
export const createScene = (paint) => (cv, optionsRef) => {
  const S = setupCanvas(cv);
  const stop = runLoop(paint(S, optionsRef));
  return () => {
    stop();
    S.dispose();
  };
};

export const pad2 = (n) => (n < 10 ? `0${n}` : `${n}`);
