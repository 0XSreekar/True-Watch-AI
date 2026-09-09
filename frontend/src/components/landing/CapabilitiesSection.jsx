import { useState } from 'react';

import { CAPABILITIES } from '../../data/landingContent';
import { css } from '../../utils/css';
import SectionHeading from '../common/SectionHeading';

export const CapabilitiesSection = () => {
  const [hover, setHover] = useState(-1);

  return (
    <section
      style={css(
        'position: relative; padding: 92px 26px; background: #F3EFE6; border-top: 1px solid rgba(43,37,33,0.18); border-bottom: 1px solid rgba(43,37,33,0.18);',
      )}
    >
      <div style={css('max-width: 1300px; margin: 0 auto;')}>
        <SectionHeading index="03" label="CAPABILITIES" />
        <div
          style={css(
            'display: flex; align-items: flex-end; justify-content: space-between; gap: 30px; flex-wrap: wrap; margin-top: 20px;',
          )}
        >
          <h2
            style={css(
              'font-family: Spectral, serif; font-weight: 500; font-size: clamp(28px,3.6vw,47px); letter-spacing: -0.02em; line-height: 1.1; margin: 0;',
            )}
          >
            Eight capabilities. All in software.
          </h2>
          <div style={css('font-size: 13px; color: rgba(43,37,33,0.64); max-width: 300px; line-height: 1.65;')}>
            No new cameras, no new poles, no per-seat licence. Hover a tile for the detail.
          </div>
        </div>

        <div
          style={css(
            'display: grid; grid-template-columns: repeat(auto-fit, minmax(246px,1fr)); margin-top: 40px; border-top: 1px solid rgba(43,37,33,0.22); border-left: 1px solid rgba(43,37,33,0.22);',
          )}
        >
          {CAPABILITIES.map((c, i) => {
            const on = hover === i;
            return (
              <div
                key={c.code}
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(-1)}
                style={css(
                  `position: relative; padding: 22px; min-height: 208px; background: ${
                    on ? '#FDFBF6' : '#FAF7F0'
                  }; border-right: 1px solid rgba(43,37,33,0.22); border-bottom: 1px solid rgba(43,37,33,0.22); box-shadow: ${
                    on ? `inset 0 0 0 1px ${c.a}` : 'none'
                  }; transition: all .28s; overflow: hidden;`,
                )}
              >
                <div style={css('position: relative; display: flex; align-items: flex-start; justify-content: space-between;')}>
                  <span
                    style={css(
                      `display: grid; place-items: center; width: 40px; height: 40px; font-size: 17px; color: ${
                        on ? '#FAF7F0' : c.a
                      }; background: ${on ? c.a : 'transparent'}; border: 1px solid ${
                        c.a
                      }; transition: all .28s;`,
                    )}
                  >
                    {c.glyph}
                  </span>
                  <span
                    style={css(
                      "font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 0.16em; color: rgba(43,37,33,0.42);",
                    )}
                  >
                    {c.code}
                  </span>
                </div>
                <div
                  style={css(
                    'position: relative; font-family: Spectral, serif; font-size: 18.5px; font-weight: 500; letter-spacing: -0.012em; margin-top: 36px; line-height: 1.24;',
                  )}
                >
                  {c.title}
                </div>
                <div
                  style={css(
                    `position: relative; font-size: 13px; line-height: 1.62; color: rgba(43,37,33,${
                      on ? 0.74 : 0.52
                    }); margin-top: 10px; transition: color .28s;`,
                  )}
                >
                  {c.desc}
                </div>
                <div
                  style={css(
                    `position: absolute; left: 22px; bottom: 0; height: 2px; width: ${
                      on ? 'calc(100% - 44px)' : '0'
                    }; background: ${c.a}; transition: width .35s;`,
                  )}
                />
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default CapabilitiesSection;
