import { useEffect, useRef } from 'react';

/**
 * Mounts one of the canvas scenes in src/canvas onto a <canvas> element.
 * The scene owns its own animation loop and is torn down on unmount.
 *
 *   const ref = useCanvasScene(feed, { ir: true, boxes: 2 });
 *   <canvas ref={ref} />
 */
export const useCanvasScene = (scene, options) => {
  const ref = useRef(null);
  // Options are re-read every frame, so a change never needs to restart the scene.
  const optionsRef = useRef(options);
  optionsRef.current = options;

  useEffect(() => {
    const el = ref.current;
    if (!el || !scene) return undefined;
    const stop = scene(el, optionsRef);
    return () => {
      if (typeof stop === 'function') stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- scene identity is stable per component
  }, [scene]);

  return ref;
};

export default useCanvasScene;
