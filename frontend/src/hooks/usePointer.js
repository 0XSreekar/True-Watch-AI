import { useEffect } from 'react';

/**
 * Normalised cursor position (-0.5 … 0.5 on each axis), shared by every canvas
 * scene so they all parallax off the same pointer — as in the original prototype.
 */
export const pointer = { x: 0, y: 0 };

let listeners = 0;
let handler = null;

export const usePointerTracking = () => {
  useEffect(() => {
    if (listeners === 0) {
      handler = (e) => {
        pointer.x = e.clientX / window.innerWidth - 0.5;
        pointer.y = e.clientY / window.innerHeight - 0.5;
      };
      window.addEventListener('mousemove', handler);
    }
    listeners += 1;
    return () => {
      listeners -= 1;
      if (listeners === 0 && handler) {
        window.removeEventListener('mousemove', handler);
        handler = null;
      }
    };
  }, []);
};

export default usePointerTracking;
