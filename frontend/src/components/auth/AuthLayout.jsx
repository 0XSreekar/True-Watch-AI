import { Link } from 'react-router-dom';

import { plan } from '../../canvas';
import { css } from '../../utils/css';
import SceneCanvas from '../common/SceneCanvas';

/** Split screen: survey-sheet panel on the left, the credential card on the right. */
export const AuthLayout = ({ children }) => (
  <div
    style={css(
      'display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); min-height: 100vh; background: #E9E3D6;',
    )}
  >
    <div style={css('position: relative; overflow: hidden; min-height: 42vh; background: #201B17;')}>
      <SceneCanvas scene={plan} options={{ onDark: true }} />
      <div
        style={css(
          'position: absolute; inset: 0; background: linear-gradient(160deg, rgba(32,27,23,0.5), rgba(32,27,23,0.15) 50%, rgba(32,27,23,0.6));',
        )}
      />

      <div style={css('position: absolute; left: 36px; top: 32px; display: flex; align-items: center; gap: 11px;')}>
        <div
          style={css(
            'width: 30px; height: 30px; border: 1.5px solid rgba(239,234,221,0.7); display: grid; place-items: center;',
          )}
        >
          <div style={css('width: 11px; height: 11px; border-radius: 50%; border: 1.5px solid #C4A455;')} />
        </div>
        <Link
          to="/"
          style={css(
            'font-family: Spectral, serif; font-weight: 600; font-size: 15px; letter-spacing: 0.03em; color: #EFEADD;',
          )}
        >
          TRUE WATCH
        </Link>
      </div>

      <div style={css('position: absolute; left: 36px; right: 36px; bottom: 36px;')}>
        <div style={css("font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.2em; color: rgba(239,234,221,0.5);")}>
          SECURE TERMINAL · SSB NETWORK
        </div>
        <div
          style={css(
            'font-family: Spectral, serif; font-size: clamp(22px,2.4vw,30px); line-height: 1.14; letter-spacing: -0.015em; color: #EFEADD; margin-top: 12px; max-width: 400px;',
          )}
        >
          2,450 km watched by two channels that must agree.
        </div>
        <div
          style={css(
            "display: flex; gap: 20px; margin-top: 20px; flex-wrap: wrap; font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.08em; color: rgba(239,234,221,0.55);",
          )}
        >
          <span>734 BOP ONLINE</span>
          <span>NEPAL · BHUTAN FRONTIER</span>
          <span>SIH26187</span>
        </div>
      </div>
    </div>

    <div style={css('position: relative; display: grid; place-items: center; padding: 56px 30px;')}>
      {children}
    </div>
  </div>
);

export default AuthLayout;
