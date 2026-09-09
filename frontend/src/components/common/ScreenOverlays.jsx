import { css } from '../../utils/css';
import { useAppShell } from '../../context/AppShellContext';

/** DIM wash + the radar wipe shown while entering the console. */
export const ScreenOverlays = () => {
  const { dim, transitioning } = useAppShell();

  return (
    <>
      <div
        style={css(
          `position: fixed; inset: 0; pointer-events: none; z-index: 200; background: #3A3129; mix-blend-mode: multiply; opacity: ${
            dim ? 0.2 : 0
          }; transition: opacity .45s;`,
        )}
      />
      <div
        style={css(
          `position: fixed; inset: 0; z-index: 300; display: ${
            transitioning ? 'grid' : 'none'
          }; place-items: center; background: #2B2521;`,
        )}
      >
        <div
          style={css(
            'width: 210px; height: 210px; border: 1px solid rgba(239,234,221,0.45); border-radius: 50%; display: grid; place-items: center; animation: sdSweep 1.2s linear infinite;',
          )}
        >
          <div style={css('width: 8px; height: 8px; background: #C4A455;')} />
        </div>
        <div
          style={css(
            'position: absolute; font-family: \'IBM Plex Mono\', monospace; font-size: 10.5px; letter-spacing: 0.24em; color: rgba(239,234,221,0.78);',
          )}
        >
          ENTERING SECTOR · RAXAUL
        </div>
      </div>
    </>
  );
};

export default ScreenOverlays;
