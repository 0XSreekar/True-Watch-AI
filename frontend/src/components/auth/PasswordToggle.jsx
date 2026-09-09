import { css } from '../../utils/css';

export const PasswordToggle = ({ shown, onToggle }) => (
  <button
    type="button"
    onClick={onToggle}
    style={css(
      "position: absolute; right: 10px; top: 34px; padding: 5px 8px; border: 1px solid rgba(43,37,33,0.2); background: transparent; color: #8A6A22; font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.06em; cursor: pointer;",
    )}
  >
    {shown ? 'HIDE' : 'SHOW'}
  </button>
);

export default PasswordToggle;
