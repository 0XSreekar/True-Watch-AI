import { css } from '../../utils/css';

/** The paper card itself — shakes on a failed validation pass. */
export const AuthCard = ({ shake, title, subtitle, children, footer, steps }) => (
  <div
    style={css(
      `position: relative; width: 100%; max-width: 462px; padding: 32px; background: #FAF7F0; border: 1px solid rgba(43,37,33,0.3); box-shadow: 8px 8px 0 rgba(43,37,33,0.09); animation: ${
        shake ? 'sdShake .4s' : 'sdRise .5s'
      } ease both;`,
    )}
    key={shake}
  >
    {steps}
    <h2 style={css('font-family: Spectral, serif; font-size: 25px; letter-spacing: -0.02em; margin: 0;')}>
      {title}
    </h2>
    <p style={css('font-size: 13.5px; line-height: 1.6; color: rgba(43,37,33,0.62); margin: 8px 0 24px;')}>
      {subtitle}
    </p>
    {children}
    {footer}
  </div>
);

export default AuthCard;
