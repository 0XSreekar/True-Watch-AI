import { feed } from '../../canvas';
import { css } from '../../utils/css';
import SceneCanvas from '../common/SceneCanvas';

const STAGE_COLOR = { prov: '#B57324', sealed: '#6E4B7C', conf: '#A32E24', gone: '#A32E24' };
const STAGE_LABEL = {
  prov: 'PROVISIONAL · ~30 ms',
  sealed: 'SEALED · EVIDENCE BLOCK',
  conf: 'CONFIRMED · ~3 s',
};

const CHANNEL_CHIPS = [
  { k: 'APPEARANCE ✓', c: '#3E5169' },
  { k: 'MOTION ✓', c: '#1F6F7A' },
];

const AlertCard = ({ alert, onAct, onDismiss }) => {
  const stage = alert.stage;
  const col = STAGE_COLOR[stage] || '#A32E24';

  return (
    <div
      style={css(
        `position: relative; padding: 14px; background: #FAF7F0; border: 1px solid rgba(43,37,33,0.22); border-left: 3px solid ${col}; box-shadow: ${
          stage === 'prov' ? `0 0 0 3px ${col}22` : '3px 3px 0 rgba(43,37,33,0.07)'
        }; opacity: ${stage === 'gone' ? 0 : 1}; transform: translateX(${
          stage === 'gone' ? 40 : 0
        }px); transition: all .35s; animation: sdSlideIn .4s ease both;`,
      )}
    >
      <div style={css('display: flex; align-items: center; justify-content: space-between; gap: 8px;')}>
        <span
          style={css(
            `display: inline-flex; align-items: center; gap: 7px; padding: 3px 8px; font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 0.12em; color: ${col}; border: 1px solid ${col}66;`,
          )}
        >
          <span
            style={css(
              `width: 5px; height: 5px; background: ${col}${
                stage === 'prov' ? '; animation: sdPulse 1.2s ease-in-out infinite' : ''
              }`,
            )}
          />
          {STAGE_LABEL[stage] || STAGE_LABEL.conf}
        </span>
        <span style={css("font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: rgba(43,37,33,0.45);")}>
          {alert.time}
        </span>
      </div>

      <div style={css('display: flex; gap: 10px; margin-top: 10px;')}>
        <div style={css('position: relative; width: 64px; height: 46px; flex: 0 0 auto; background: #1C1815; overflow: hidden;')}>
          <SceneCanvas
            scene={feed}
            options={{ seed: alert.seed, ir: alert.camera.ir, boxes: 1, labels: false }}
          />
        </div>
        <div style={css('min-width: 0;')}>
          <div style={css('font-size: 12px; font-weight: 500;')}>{alert.camera.name}</div>
          <div style={css("font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; color: rgba(43,37,33,0.5); margin-top: 2px;")}>
            {alert.camera.id}
            {alert.plate ? ` · ${alert.plate}` : ''}
          </div>
          {stage === 'prov' ? (
            <div style={css('display: flex; gap: 6px; margin-top: 7px;')}>
              {CHANNEL_CHIPS.map((chip, ci) => (
                <span
                  key={chip.k}
                  style={css(
                    `padding: 3px 8px; font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 0.08em; color: ${
                      chip.c
                    }; border: 1px solid ${chip.c}55; background: ${chip.c}10; animation: sdRise .35s ${
                      0.1 + ci * 0.15
                    }s ease both;`,
                  )}
                >
                  {chip.k}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      </div>

      <div
        style={css(
          `font-family: Spectral, serif; font-size: 14.5px; line-height: 1.55; color: #2B2521; margin-top: 10px; max-height: ${
            stage === 'prov' ? 0 : 110
          }px; opacity: ${stage === 'prov' ? 0 : 1}; overflow: hidden; transition: all .4s;`,
        )}
      >
        {alert.reason.slice(0, alert.typed)}
      </div>

      <div
        style={css(
          `display: ${stage === 'prov' ? 'none' : 'flex'}; align-items: center; gap: 9px; margin-top: 11px;`,
        )}
      >
        <div style={css('flex: 1; height: 5px; background: rgba(43,37,33,0.1);')}>
          <div
            style={css(
              `height: 5px; width: ${alert.conf || 0}%; background: ${col}; transition: width 1s cubic-bezier(.2,.8,.2,1);`,
            )}
          />
        </div>
        <span
          style={css(
            "font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; color: rgba(43,37,33,0.55); white-space: nowrap;",
          )}
        >
          {alert.conf ? `${alert.conf}% CONFIDENCE` : ''}
        </span>
      </div>

      <div style={css(`display: ${stage === 'conf' ? 'flex' : 'none'}; gap: 8px; margin-top: 12px;`)}>
        <button
          type="button"
          onClick={() => onAct(alert.id)}
          style={css(
            'flex: 1; padding: 9px; border: 1px solid #2B2521; background: #2B2521; color: #F3EFE6; font-size: 11px; letter-spacing: 0.06em; cursor: pointer;',
          )}
        >
          ACT · SEAL EVIDENCE
        </button>
        <button
          type="button"
          onClick={() => onDismiss(alert.id)}
          style={css(
            'padding: 9px 14px; border: 1px solid rgba(43,37,33,0.28); background: transparent; color: #2B2521; font-size: 11px; letter-spacing: 0.06em; cursor: pointer;',
          )}
        >
          DISMISS
        </button>
      </div>

      <div
        style={css(
          `display: ${
            stage === 'sealed' ? 'flex' : 'none'
          }; align-items: center; gap: 9px; margin-top: 11px; padding: 9px 11px; border: 1px solid rgba(110,75,124,0.42); background: rgba(110,75,124,0.07); font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #573C63;`,
        )}
      >
        <span>⛓</span>
        <span>SEALED · {alert.hash}</span>
      </div>
    </div>
  );
};

export const AlertQueue = ({ alerts, onAct, onDismiss }) => (
  <div>
    <div style={css('display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 10px;')}>
      <span style={css("font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.16em; color: #A32E24;")}>
        ALERT QUEUE
      </span>
      <span style={css("font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: rgba(43,37,33,0.5);")}>
        {alerts.filter((a) => a.stage !== 'gone').length} ACTIVE
      </span>
    </div>
    <div
      style={css(
        'display: flex; flex-direction: column; gap: 10px; max-height: 620px; overflow-y: auto; padding-right: 4px;',
      )}
    >
      {alerts.map((alert) => (
        <AlertCard key={alert.id} alert={alert} onAct={onAct} onDismiss={onDismiss} />
      ))}
    </div>
  </div>
);

export default AlertQueue;
