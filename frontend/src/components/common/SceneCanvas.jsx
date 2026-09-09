import useCanvasScene from '../../hooks/useCanvasScene';
import { css } from '../../utils/css';

const FILL = 'position: absolute; inset: 0; width: 100%; height: 100%; display: block;';

/**
 * A <canvas> running one of the scenes in src/canvas.
 * `style` is a CSS declaration string, like everything else in this UI.
 */
export const SceneCanvas = ({ scene, options, style }) => {
  const ref = useCanvasScene(scene, options);
  return <canvas ref={ref} style={{ ...css(FILL), ...css(style) }} />;
};

export default SceneCanvas;
