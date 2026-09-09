import { useCallback, useRef, useState } from 'react';

import { feed } from '../../canvas';
import { css } from '../../utils/css';
import SceneCanvas from '../common/SceneCanvas';
import SectionHeading from '../common/SectionHeading';

const label = (d) =>
  d < 20
    ? 'CLEAR · 60 m · DAY'
    : d < 45
      ? 'LIGHT HAZE · 110 m'
      : d < 70
        ? 'RIVER FOG · 160 m · IR'
        : d < 88
          ? 'DENSE FOG · 210 m · IR'
          : 'NEAR-ZERO · 260 m';

const Bar = ({ name, value, color }) => (
  <div>
    <div style={css('display: flex; justify-content: space-between; font-size: 12.5px; margin-bottom: 7px;')}>
      <span style={css('font-weight: 500;')}>{name}</span>
      <span style={css(`font-family: 'IBM Plex Mono', monospace; color: ${color};`)}>{value}%</span>
    </div>
    <div style={css('height: 7px; background: rgba(43,37,33,0.1); overflow: hidden;')}>
      <div style={css(`height: 100%; width: ${value}%; background: ${color}; transition: width .3s;`)} />
    </div>
  </div>
);

/**
 * The interactive proof that two channels beat one: degrade the frame and watch
 * the appearance channel collapse while the motion channel holds.
 */
export const ChannelsSection = () => {
  const [degrade, setDegrade] = useState(18);
  // The canvas reads this each frame instead of restarting on every slider tick.
  const degradeRef = useRef(degrade);
  degradeRef.current = degrade;
  const getDegrade = useCallback(() => degradeRef.current, []);

  const appConf = Math.max(11, Math.round(96 - degrade * 0.85));
  const motConf = Math.max(62, Math.round(94 - degrade * 0.19));
  const appColor = appConf > 70 ? '#1F6F7A' : appConf > 45 ? '#B57324' : '#A32E24';
  const motColor = motConf > 70 ? '#1F6F7A' : '#B57324';
  const agree = appConf >= 45;

  return (
    <section style={css('position: relative; padding: 92px 26px; background: #E9E3D6; overflow: hidden;')}>
      <div
        style={css(
          'position: relative; max-width: 1300px; margin: 0 auto; display: grid; grid-template-columns: repeat(auto-fit, minmax(354px,1fr)); gap: 44px; align-items: center;',
        )}
      >
        <div>
          <SectionHeading index="04" label="WHY TWO CHANNELS" />
          <h2
            style={css(
              'font-family: Spectral, serif; font-weight: 500; font-size: clamp(27px,3.3vw,43px); letter-spacing: -0.02em; line-height: 1.12; margin: 20px 0 0;',
            )}
          >
            One channel fails quietly. Two fail loudly.
          </h2>
          <p
            style={css(
              'font-family: Spectral, serif; font-size: 16.5px; line-height: 1.68; color: rgba(43,37,33,0.72); margin: 16px 0 0; max-width: 470px;',
            )}
          >
            Drag the slider to degrade the frame the way real conditions do — river fog at
            Panitanki, no moon over the Jogbani ridge, a target at 180 m. The appearance channel
            loses confidence fast. The motion channel barely moves, because physics does not care
            what something looks like.
          </p>

          <div
            style={css(
              'margin-top: 28px; padding: 20px; border: 1px solid rgba(43,37,33,0.24); background: #FAF7F0; box-shadow: 5px 5px 0 rgba(43,37,33,0.06);',
            )}
          >
            <div
              style={css(
                "display: flex; align-items: center; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.14em; color: rgba(43,37,33,0.58);",
              )}
            >
              <span>FRAME CONDITION</span>
              <span style={css('color: #2B2521;')}>{label(degrade)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={degrade}
              onChange={(e) => setDegrade(Number(e.target.value))}
              style={css('width: 100%; margin: 14px 0 20px; accent-color: #8A6A22; height: 3px;')}
            />
            <div style={css('display: flex; flex-direction: column; gap: 15px;')}>
              <Bar name="Appearance channel" value={appConf} color={appColor} />
              <Bar name="Motion channel" value={motConf} color={motColor} />
            </div>
            <div
              style={css(
                `margin-top: 18px; padding: 12px 14px; font-size: 12.5px; line-height: 1.6; border: 1px solid ${
                  agree ? 'rgba(31,111,122,0.5)' : 'rgba(181,115,36,0.55)'
                }; border-left: 3px solid ${agree ? '#1F6F7A' : '#B57324'}; background: ${
                  agree ? 'rgba(31,111,122,0.07)' : 'rgba(181,115,36,0.08)'
                }; color: ${agree ? '#1A555D' : '#7A4E14'}; transition: all .3s;`,
              )}
            >
              {agree
                ? 'Both channels agree — alert raised, and the judge has enough to write a reason.'
                : 'Appearance has collapsed, motion still holds. The gate withholds the alert instead of guessing — no false alarm spent.'}
            </div>
          </div>
        </div>

        <div
          style={css(
            'position: relative; border: 1px solid rgba(43,37,33,0.3); box-shadow: 7px 7px 0 rgba(43,37,33,0.08); aspect-ratio: 4 / 3; background: #241F1B; overflow: hidden;',
          )}
        >
          <SceneCanvas scene={feed} options={{ seed: 3.3, ir: false, boxes: 2, getDegrade }} />
          <div style={css('position: absolute; left: 11px; top: 11px; display: flex; gap: 7px;')}>
            <span
              style={css(
                "padding: 5px 9px; border: 1px solid rgba(239,234,221,0.42); font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.12em; color: #EFEADD; background: rgba(36,31,27,0.6);",
              )}
            >
              CAM PNT-07 · PANITANKI
            </span>
            <span
              style={css(
                `padding: 5px 9px; font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.12em; color: #EFEADD; border: 1px solid ${
                  degrade > 45 ? 'rgba(110,168,178,0.75)' : 'rgba(196,164,85,0.75)'
                }; background: rgba(36,31,27,0.6);`,
              )}
            >
              {degrade > 45 ? 'IR ACTIVE' : 'DAY MODE'}
            </span>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ChannelsSection;
