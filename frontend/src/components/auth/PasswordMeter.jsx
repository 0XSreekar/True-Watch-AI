import { css } from '../../utils/css';

const LABELS = ['ENTER A PASSWORD', 'VERY WEAK', 'WEAK', 'FAIR', 'STRONG', 'VERY STRONG'];
const COLORS = ['#A32E24', '#A32E24', '#B57324', '#B57324', '#1F6F7A', '#1F6F7A'];

export const PasswordMeter = ({ score }) => {
  const color = COLORS[score];
  return (
    <>
      <div style={css('display: flex; gap: 5px; margin-top: 9px;')}>
        {[0, 1, 2, 3, 4].map((i) => (
          <span
            key={i}
            style={css(
              `flex: 1; height: 4px; background: ${
                i < score ? color : 'rgba(43,37,33,0.14)'
              }; transition: background .25s ${i * 0.04}s;`,
            )}
          />
        ))}
      </div>
      <div
        style={css(
          `font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.08em; margin-top: 7px; color: ${color};`,
        )}
      >
        {LABELS[score]}
      </div>
    </>
  );
};

export default PasswordMeter;
