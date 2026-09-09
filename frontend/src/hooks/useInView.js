import { useCallback, useRef, useState } from 'react';

/** Fires once when the element scrolls into view — used by the counting stats band. */
export const useInView = ({ threshold = 0.2 } = {}) => {
  const [inView, setInView] = useState(false);
  const observerRef = useRef(null);

  const ref = useCallback(
    (el) => {
      if (observerRef.current) {
        observerRef.current.disconnect();
        observerRef.current = null;
      }
      if (!el) return;
      try {
        const io = new IntersectionObserver(
          (entries) => {
            if (entries[0].isIntersecting) {
              setInView(true);
              io.disconnect();
            }
          },
          { threshold },
        );
        io.observe(el);
        observerRef.current = io;
      } catch {
        setInView(true);
      }
    },
    [threshold],
  );

  return [ref, inView];
};

export default useInView;
