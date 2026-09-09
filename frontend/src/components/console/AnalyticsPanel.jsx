import { css } from '../../utils/css';
import { sparkPath } from '../../utils/format';

const Card = ({ children }) => (
  <div style={css('padding: 16px; border: 1px solid rgba(43,37,33,0.22); background: #FAF7F0;')}>
    {children}
  </div>
);

const CardTitle = ({ children, value, valueColor }) => (
  <div
    style={css(
      value
        ? 'display: flex; justify-content: space-between; font-size: 11.5px; font-weight: 500; margin-bottom: 8px;'
        : 'font-size: 11.5px; font-weight: 500; margin-bottom: 12px;',
    )}
  >
    <span>{children}</span>
    {value ? (
      <span style={css(`font-family: 'IBM Plex Mono', monospace; color: ${valueColor};`)}>{value}</span>
    ) : null}
  </div>
);

export const AnalyticsPanel = ({ analytics, traffic }) => {
  const { camBars, dayNight, agreement } = analytics;

  return (
    <div>
      <div style={css("font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.16em; color: #3E5169; margin-bottom: 10px;")}>
        ANALYTICS
      </div>
      <div style={css('display: grid; grid-template-columns: repeat(auto-fit, minmax(230px,1fr)); gap: 12px;')}>
        <Card>
          <CardTitle>Alerts / camera vs budget</CardTitle>
          <div style={css('display: flex; align-items: flex-end; gap: 8px; height: 74px;')}>
            {camBars.map((bar) => (
              <div
                key={bar.id}
                style={css(
                  'flex: 1; display: flex; flex-direction: column; align-items: center; height: 100%; justify-content: flex-end;',
                )}
              >
                <div
                  style={css(
                    `width: 100%; height: ${(bar.v / 12) * 100}%; background: ${
                      bar.v >= 10 ? '#B57324' : '#1F6F7A'
                    }; border: 1px solid rgba(43,37,33,0.28); border-bottom: none; transition: height .7s cubic-bezier(.2,.8,.2,1);`,
                  )}
                />
                <span
                  style={css(
                    "font-family: 'IBM Plex Mono', monospace; font-size: 7.5px; margin-top: 4px; color: rgba(43,37,33,0.5);",
                  )}
                >
                  {bar.id}
                </span>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <CardTitle>Day vs night split</CardTitle>
          <div style={css('display: flex; flex-direction: column; gap: 10px;')}>
            {dayNight.map((b) => (
              <div key={b.k}>
                <div
                  style={css(
                    "display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 9px; margin-bottom: 4px; color: rgba(43,37,33,0.6);",
                  )}
                >
                  <span>{b.k}</span>
                  <span>{b.v}%</span>
                </div>
                <div style={css('height: 9px; background: rgba(43,37,33,0.08);')}>
                  <div
                    style={css(
                      `height: 9px; width: ${b.v}%; background: ${b.c}; transition: width .9s cubic-bezier(.2,.8,.2,1);`,
                    )}
                  />
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <CardTitle value={`${agreement[agreement.length - 1]}%`} valueColor="#1F6F7A">
            Channel agreement rate
          </CardTitle>
          <svg viewBox="0 0 300 66" style={css('width: 100%; height: 74px; display: block;')}>
            <path d={sparkPath(agreement, 300, 66, 100)} fill="none" stroke="#1F6F7A" strokeWidth="2" />
          </svg>
        </Card>

        <Card>
          <CardTitle value={`${traffic[traffic.length - 1]} TRACKS/MIN`} valueColor="#B57324">
            Live traffic flow
          </CardTitle>
          <svg viewBox="0 0 300 66" style={css('width: 100%; height: 74px; display: block;')}>
            <path d={sparkPath(traffic, 300, 66, 16)} fill="none" stroke="#B57324" strokeWidth="2" />
          </svg>
        </Card>
      </div>
    </div>
  );
};

export default AnalyticsPanel;
