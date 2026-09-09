import { css } from '../../utils/css';

const STEPS = ['IDENTITY', 'CREDENTIALS', 'ROLE & TERMS'];

export const StepDots = ({ step }) => (
  <div style={css('display: flex; align-items: center; gap: 7px; margin-bottom: 24px;')}>
    {STEPS.map((label, i) => (
      <div key={label} style={css('display: flex; align-items: center; gap: 8px; flex: 1;')}>
        <span
          style={css(
            `display: grid; place-items: center; width: 22px; height: 22px; flex: 0 0 auto; font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: ${
              i <= step ? '#FAF7F0' : 'rgba(43,37,33,0.45)'
            }; background: ${i <= step ? '#2B2521' : 'transparent'}; border: 1px solid ${
              i <= step ? '#2B2521' : 'rgba(43,37,33,0.32)'
            }; transition: all .3s;`,
          )}
        >
          {i + 1}
        </span>
        <span
          style={css(
            `font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.13em; white-space: nowrap; color: ${
              i <= step ? '#2B2521' : 'rgba(43,37,33,0.42)'
            };`,
          )}
        >
          {label}
        </span>
        <span
          style={css(
            `flex: 1; height: 1px; min-width: 10px; background: ${
              i < step ? '#A9812F' : 'rgba(43,37,33,0.22)'
            }; transition: background .35s;`,
          )}
        />
      </div>
    ))}
  </div>
);

export default StepDots;
