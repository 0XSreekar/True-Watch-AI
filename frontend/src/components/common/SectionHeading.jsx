import { css } from '../../utils/css';

/** The "01 ——— THE GROUND TRUTH" rule that opens each landing section. */
export const SectionHeading = ({ index, label, color = '#8A6A22' }) => (
  <div
    style={css(
      `display: flex; align-items: baseline; gap: 14px; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.2em; color: ${color};`,
    )}
  >
    <span>{index}</span>
    <span style={css('flex: 1; height: 1px; background: rgba(43,37,33,0.22);')} />
    <span>{label}</span>
  </div>
);

export default SectionHeading;
