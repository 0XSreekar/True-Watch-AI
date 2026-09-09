import { useNavigate } from 'react-router-dom';

import { useAppShell } from '../../context/AppShellContext';
import { css } from '../../utils/css';

export const TopNav = () => {
  const navigate = useNavigate();
  const { toggleDim, dimLabel, enterConsole } = useAppShell();

  return (
    <div
      style={css(
        'position: fixed; top: 0; left: 0; right: 0; z-index: 60; display: flex; align-items: center; gap: 16px; padding: 13px 26px; background: rgba(243,239,230,0.94); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(43,37,33,0.18);',
      )}
    >
      <div style={css('display: flex; align-items: center; gap: 12px;')}>
        <div
          style={css(
            'position: relative; width: 32px; height: 32px; border: 1.5px solid #2B2521; display: grid; place-items: center;',
          )}
        >
          <div style={css('width: 13px; height: 13px; border-radius: 50%; border: 1.5px solid #A9812F;')} />
          <div style={css('position: absolute; left: 50%; top: 2px; bottom: 2px; width: 1px; background: rgba(43,37,33,0.3);')} />
          <div style={css('position: absolute; top: 50%; left: 2px; right: 2px; height: 1px; background: rgba(43,37,33,0.3);')} />
        </div>
        <div style={css('display: flex; flex-direction: column; line-height: 1.15;')}>
          <span style={css("font-family: Spectral, serif; font-weight: 600; font-size: 16.5px; letter-spacing: 0.03em;")}>
            TRUE WATCH
          </span>
          <span style={css("font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 0.2em; color: rgba(43,37,33,0.52);")}>
            SIH26187 · MHA / SSB
          </span>
        </div>
      </div>

      <div style={css('flex: 1;')} />

      <button
        type="button"
        className="tw-ghost-btn"
        onClick={toggleDim}
        style={css(
          'display: flex; align-items: center; gap: 8px; padding: 8px 12px; border: 1px solid rgba(43,37,33,0.24); background: transparent; color: #2B2521; font-size: 11.5px; letter-spacing: 0.05em; cursor: pointer; transition: all .2s;',
        )}
      >
        <span style={css('width: 7px; height: 7px; background: #A9812F;')} />
        {dimLabel}
      </button>

      <button
        type="button"
        className="tw-ghost-btn"
        onClick={() => navigate('/login')}
        style={css(
          'padding: 9px 15px; border: 1px solid rgba(43,37,33,0.24); background: transparent; color: #2B2521; font-size: 12px; letter-spacing: 0.05em; cursor: pointer; transition: all .2s;',
        )}
      >
        Sign in
      </button>

      <button
        type="button"
        className="tw-solid-btn"
        onClick={enterConsole}
        style={css(
          'padding: 10px 17px; border: 1px solid #2B2521; background: #2B2521; color: #F3EFE6; font-size: 12px; letter-spacing: 0.06em; cursor: pointer; box-shadow: 3px 3px 0 rgba(169,129,47,0.45); transition: all .18s;',
        )}
      >
        COMMAND CONSOLE
      </button>
    </div>
  );
};

export default TopNav;
