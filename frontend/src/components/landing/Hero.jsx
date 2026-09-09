import { cine } from '../../canvas';
import { useAppShell } from '../../context/AppShellContext';
import { css } from '../../utils/css';
import SceneCanvas from '../common/SceneCanvas';

const LATENCY_CHIPS = [
  { dot: '#B57324', color: '#7A4E14', label: 'PROVISIONAL ~30 ms', pulse: true },
  { dot: '#A32E24', color: '#7C231B', label: 'CONFIRMED + REASON ~3 s' },
  { dot: '#6E4B7C', color: '#573C63', label: 'HASH-CHAINED AT CAPTURE' },
];

export const Hero = ({ onSeeHowItWorks }) => {
  const { enterConsole } = useAppShell();

  return (
    <section
      style={css(
        'position: relative; min-height: 100vh; display: flex; align-items: center; overflow: hidden; padding: 116px 26px 56px; background: linear-gradient(172deg,#DBD3C2 0%, #E4DDCE 30%, #E9E3D6 62%, #DED6C6 100%);',
      )}
    >
      <div
        style={css(
          'position: absolute; inset: 0; opacity: 0.55; background-image: linear-gradient(rgba(43,37,33,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(43,37,33,0.05) 1px, transparent 1px); background-size: 32px 32px;',
        )}
      />
      <SceneCanvas scene={cine} />
      <div
        style={css(
          'position: absolute; inset: 0; background: linear-gradient(100deg, rgba(43,37,33,0.32) 0%, rgba(43,37,33,0.16) 30%, rgba(43,37,33,0.02) 56%, transparent 72%);',
        )}
      />
      <div
        style={css(
          "position: absolute; right: 22px; top: 90px; padding: 5px 10px; border: 1px solid rgba(43,37,33,0.3); background: rgba(250,247,240,0.75); font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.12em; color: #2B2521; z-index: 3;",
        )}
      >
        CAM RXL-01 · LIVE FEED SIMULATION
      </div>

      <div style={css('position: relative; width: 100%; max-width: 1300px; margin: 0 auto;')}>
        <div style={css('max-width: 700px;')}>
          <div
            style={css(
              'display: inline-flex; align-items: center; gap: 11px; padding: 7px 13px; border: 1px solid rgba(43,37,33,0.3); background: rgba(250,247,240,0.72); animation: sdRise .6s ease both;',
            )}
          >
            <span style={css('position: relative; width: 7px; height: 7px; background: #1F6F7A;')}>
              <span style={css('position: absolute; inset: -4px; border: 1px solid #1F6F7A; animation: sdRing 2.6s ease-out infinite;')} />
            </span>
            <span style={css("font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.19em; color: #2B2521;")}>
              EDGE-RESIDENT · 734 BOP READY
            </span>
          </div>

          <h1
            style={css(
              'font-family: Spectral, serif; font-size: clamp(38px, 5.9vw, 78px); line-height: 1.02; letter-spacing: -0.022em; font-weight: 500; margin: 24px 0 0; animation: sdRise .7s .06s ease both; color: #F3EFE6; text-shadow: 0 3px 26px rgba(20,17,14,0.85), 0 1px 3px rgba(20,17,14,0.9);',
            )}
          >
            Two channels that see.
            <br />
            <span style={css('font-style: italic; font-weight: 400; color: #E0B662;')}>
              One model that judges.
            </span>
          </h1>

          <div style={css('width: 96px; height: 1px; background: #E0B662; margin: 26px 0 0; animation: sdRise .7s .1s ease both;')} />

          <p
            style={css(
              'font-family: Spectral, serif; font-size: clamp(16px, 1.4vw, 20px); line-height: 1.66; color: rgba(243,239,230,0.92); max-width: 588px; margin: 22px 0 0; animation: sdRise .7s .14s ease both; text-shadow: 0 2px 18px rgba(20,17,14,0.75);',
            )}
          >
            TRUE WATCH turns the CCTV the Sashastra Seema Bal already owns into an intelligent
            surveillance network. An appearance channel and a motion channel must agree before
            anything is raised — then a vision-language model reads the evidence and writes the
            reason in plain words.
          </p>

          <div style={css('display: flex; flex-wrap: wrap; gap: 13px; margin-top: 34px; animation: sdRise .7s .2s ease both;')}>
            <button
              type="button"
              className="tw-solid-btn-lg"
              onClick={enterConsole}
              style={css(
                'display: inline-flex; align-items: center; gap: 13px; padding: 16px 26px; border: 1px solid #2B2521; background: #2B2521; color: #F3EFE6; font-size: 13.5px; letter-spacing: 0.09em; cursor: pointer; box-shadow: 4px 4px 0 rgba(169,129,47,0.5); transition: all .2s;',
              )}
            >
              ENTER COMMAND CONSOLE <span style={css('font-size: 15px; line-height: 1;')}>&#8594;</span>
            </button>
            <button
              type="button"
              className="tw-ghost-btn"
              onClick={onSeeHowItWorks}
              style={css(
                'display: inline-flex; align-items: center; gap: 10px; padding: 16px 24px; border: 1px solid rgba(43,37,33,0.32); background: rgba(250,247,240,0.5); color: #2B2521; font-size: 13.5px; letter-spacing: 0.09em; cursor: pointer; transition: all .2s;',
              )}
            >
              SEE HOW IT WORKS
            </button>
          </div>

          <div
            style={css(
              'display: flex; flex-wrap: wrap; margin-top: 34px; border: 1px solid rgba(43,37,33,0.2); background: rgba(250,247,240,0.72); animation: sdRise .7s .26s ease both;',
            )}
          >
            {LATENCY_CHIPS.map((chip, i) => (
              <div
                key={chip.label}
                style={css(
                  `display: flex; align-items: center; gap: 9px; padding: 11px 15px;${
                    i < LATENCY_CHIPS.length - 1 ? ' border-right: 1px solid rgba(43,37,33,0.16);' : ''
                  }`,
                )}
              >
                <span
                  style={css(
                    `width: 7px; height: 7px; background: ${chip.dot};${
                      chip.pulse ? ' animation: sdPulse 1.5s ease-in-out infinite;' : ''
                    }`,
                  )}
                />
                <span
                  style={css(
                    `font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.1em; color: ${chip.color};`,
                  )}
                >
                  {chip.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div
        style={css(
          'position: absolute; bottom: 22px; left: 50%; transform: translateX(-50%); display: flex; flex-direction: column; align-items: center; gap: 7px;',
        )}
      >
        <span style={css("font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 0.26em; color: rgba(43,37,33,0.5);")}>
          SCROLL
        </span>
        <div
          style={css(
            'width: 19px; height: 30px; border: 1px solid rgba(43,37,33,0.3); display: grid; place-items: start center; padding-top: 5px;',
          )}
        >
          <span style={css('width: 3px; height: 8px; background: #A9812F; animation: sdScroll 2s ease-in-out infinite;')} />
        </div>
      </div>

      <div
        style={css(
          "position: absolute; right: 26px; bottom: 26px; font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.16em; color: rgba(43,37,33,0.5); text-align: right; line-height: 1.85;",
        )}
      >
        FIG. 1 — COVERAGE VOLUME, BOP RAXAUL
        <br />
        ORTHOGRAPHIC · SCAN PLANE SWEEPING 0–40 m
      </div>
    </section>
  );
};

export default Hero;
