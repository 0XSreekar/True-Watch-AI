import { Link } from 'react-router-dom';

import { CONSOLE_NAV } from '../../data/landingContent';
import { css } from '../../utils/css';

/** Collapsible icon rail. Items are inert until their modules are built. */
export const ConsoleRail = ({ open, onToggle }) => (
  <div
    style={css(
      `position: sticky; top: 0; align-self: start; height: 100vh; flex: 0 0 auto; width: ${
        open ? 206 : 60
      }px; padding: 15px 9px; display: flex; flex-direction: column; gap: 4px; background: #F3EFE6; border-right: 1px solid rgba(43,37,33,0.2); transition: width .32s cubic-bezier(.2,.8,.2,1); overflow: hidden; z-index: 40;`,
    )}
  >
    <div
      onClick={onToggle}
      style={css('display: flex; align-items: center; gap: 12px; padding: 10px 11px; margin-bottom: 10px; cursor: pointer;')}
    >
      <div
        style={css(
          'width: 20px; height: 20px; border: 1.5px solid #2B2521; display: grid; place-items: center; flex: 0 0 auto;',
        )}
      >
        <div style={css('width: 7px; height: 7px; border-radius: 50%; border: 1.5px solid #A9812F;')} />
      </div>
      <span
        style={css(
          `font-family: Spectral, serif; font-weight: 600; font-size: 13px; opacity: ${
            open ? 1 : 0
          }; white-space: nowrap;`,
        )}
      >
        TRUE WATCH
      </span>
    </div>

    {CONSOLE_NAV.map((item, i) => (
      <div
        key={item.t}
        style={css(
          `display: flex; align-items: center; gap: 13px; padding: 10px 11px; cursor: pointer; color: ${
            i === 0 ? '#FAF7F0' : 'rgba(43,37,33,0.7)'
          }; background: ${i === 0 ? '#2B2521' : 'transparent'}; border-left: 2px solid ${
            i === 0 ? '#A9812F' : 'transparent'
          }; transition: all .22s;`,
        )}
      >
        <span style={css('flex: 0 0 auto; width: 18px; text-align: center; font-size: 14px;')}>
          {item.g}
        </span>
        <span
          style={css(
            `font-size: 12.5px; white-space: nowrap; opacity: ${open ? 1 : 0}; transition: opacity .25s;`,
          )}
        >
          {item.t}
        </span>
      </div>
    ))}

    <div style={css('flex: 1;')} />

    <Link
      to="/"
      style={css(
        'display: flex; align-items: center; gap: 13px; padding: 10px 11px; color: rgba(43,37,33,0.5); font-size: 12px;',
      )}
    >
      <span style={css('width: 18px; text-align: center;')}>⏎</span>
      <span style={css(`opacity: ${open ? 1 : 0};`)}>Exit console</span>
    </Link>
  </div>
);

export default ConsoleRail;
