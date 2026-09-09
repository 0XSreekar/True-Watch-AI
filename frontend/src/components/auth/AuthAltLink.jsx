import { css } from '../../utils/css';

export const AuthAltLink = ({ prompt, linkLabel, onClick }) => (
  <div
    style={css(
      'display: flex; justify-content: center; gap: 6px; margin-top: 18px; font-size: 12.5px; color: rgba(43,37,33,0.58);',
    )}
  >
    <span>{prompt}</span>
    <a
      href="#alt"
      onClick={(e) => {
        e.preventDefault();
        onClick();
      }}
      style={css('color: #8A6A22; font-weight: 500;')}
    >
      {linkLabel}
    </a>
  </div>
);

export default AuthAltLink;
