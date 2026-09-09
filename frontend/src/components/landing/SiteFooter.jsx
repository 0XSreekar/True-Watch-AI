import { useNavigate } from 'react-router-dom';

import { useAppShell } from '../../context/AppShellContext';
import { css } from '../../utils/css';

const linkStyle = css('color: rgba(239,234,221,0.82);');
const monoLabel = css(
  "font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.19em; color: rgba(239,234,221,0.44);",
);

export const SiteFooter = () => {
  const navigate = useNavigate();
  const { enterConsole } = useAppShell();

  const go = (path) => (e) => {
    e.preventDefault();
    navigate(path);
  };

  return (
    <footer style={css('position: relative; padding: 58px 26px 36px; background: #2B2521; color: #EFEADD;')}>
      <div
        style={css(
          'position: relative; max-width: 1300px; margin: 0 auto; display: grid; grid-template-columns: repeat(auto-fit,minmax(226px,1fr)); gap: 32px; align-items: start;',
        )}
      >
        <div>
          <div style={css('font-family: Spectral, serif; font-weight: 600; font-size: 20px; letter-spacing: 0.03em;')}>
            TRUE WATCH
          </div>
          <p
            style={css(
              'font-size: 13px; line-height: 1.7; color: rgba(239,234,221,0.6); margin: 11px 0 0; max-width: 290px;',
            )}
          >
            Edge-resident border video analytics for the open Nepal and Bhutan frontier.
          </p>
        </div>

        <div style={css('display: flex; flex-direction: column; gap: 9px; font-size: 12.5px; color: rgba(239,234,221,0.64);')}>
          <span style={monoLabel}>PROBLEM STATEMENT</span>
          <span style={css("font-family: 'IBM Plex Mono', monospace; font-size: 15px; color: #C4A455;")}>
            SIH26187
          </span>
          <span>Ministry of Home Affairs</span>
          <span>Sashastra Seema Bal</span>
        </div>

        <div style={css('display: flex; flex-direction: column; gap: 9px; font-size: 12.5px;')}>
          <span style={monoLabel}>SCREENS</span>
          <a href="/login" onClick={go('/login')} style={linkStyle}>
            Sign in
          </a>
          <a href="/signup" onClick={go('/signup')} style={linkStyle}>
            Create account
          </a>
          <a href="/console" onClick={enterConsole} style={linkStyle}>
            Command console
          </a>
        </div>

        <div style={css('display: flex; justify-content: flex-end;')}>
          <div style={css('padding: 16px 20px; border: 1px solid rgba(239,234,221,0.3); text-align: center;')}>
            <div style={css('font-family: Spectral, serif; font-size: 25px; font-weight: 500; letter-spacing: -0.02em;')}>
              SIH 2026
            </div>
            <div
              style={css(
                "font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 0.19em; color: rgba(239,234,221,0.6); margin-top: 5px;",
              )}
            >
              SMART INDIA HACKATHON
            </div>
          </div>
        </div>
      </div>

      <div
        style={css(
          "position: relative; max-width: 1300px; margin: 34px auto 0; padding-top: 18px; border-top: 1px solid rgba(239,234,221,0.18); display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.13em; color: rgba(239,234,221,0.48);",
        )}
      >
        <span>NO LICENCE COST · RUNS ON EXISTING CCTV</span>
        <span>PROTOTYPE UI · MOCK DATA</span>
      </div>
    </footer>
  );
};

export default SiteFooter;
