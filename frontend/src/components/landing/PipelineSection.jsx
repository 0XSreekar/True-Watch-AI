import { forwardRef, useState } from 'react';

import { PIPELINE } from '../../data/landingContent';
import { css } from '../../utils/css';
import SectionHeading from '../common/SectionHeading';

const LEGEND = [
  { c: '#3E5169', t: 'Appearance rail — neural detector' },
  { c: '#1F6F7A', t: 'Motion rail — optical flow + physics' },
  { c: '#6E4B7C', t: 'Evidence rail — hash chain' },
];

export const PipelineSection = forwardRef((_props, ref) => {
  const [hover, setHover] = useState(-1);

  return (
    <section ref={ref} style={css('position: relative; padding: 92px 26px 100px; background: #E9E3D6;')}>
      <div style={css('max-width: 1300px; margin: 0 auto;')}>
        <SectionHeading index="02" label="THE PIPELINE" />
        <h2
          style={css(
            'font-family: Spectral, serif; font-weight: 500; font-size: clamp(28px,3.6vw,47px); letter-spacing: -0.02em; line-height: 1.1; margin: 20px 0 0; max-width: 780px;',
          )}
        >
          Frame in. Judgement out. Sealed on the way past.
        </h2>
        <p
          style={css(
            'font-family: Spectral, serif; font-size: 17px; line-height: 1.66; color: rgba(43,37,33,0.7); max-width: 600px; margin: 15px 0 0;',
          )}
        >
          Everything below runs on a single edge box at the post — no uplink required. Hover any
          stage to read what it does.
        </p>

        <div
          style={css(
            'position: relative; margin-top: 46px; padding: 40px 22px 28px; border: 1px solid rgba(43,37,33,0.24); background: #FAF7F0; box-shadow: 6px 6px 0 rgba(43,37,33,0.07); overflow: hidden;',
          )}
        >
          <div
            style={css(
              'position: absolute; left: 22px; right: 22px; top: 50%; height: 1px; background: repeating-linear-gradient(90deg, rgba(43,37,33,0.34) 0 7px, transparent 7px 13px);',
            )}
          />
          <div style={css('position: absolute; left: 22px; right: 22px; top: 50%; height: 0;')}>
            <span
              style={css(
                'position: absolute; top: -4px; width: 8px; height: 8px; background: #A9812F; box-shadow: 0 0 0 4px rgba(169,129,47,0.22); animation: sdPacket 8s linear infinite;',
              )}
            />
          </div>

          <div
            style={css(
              'position: relative; display: grid; grid-template-columns: repeat(auto-fit, minmax(138px, 1fr)); gap: 12px;',
            )}
          >
            {PIPELINE.map((node, i) => {
              const on = hover === i;
              return (
                <div
                  key={node.title}
                  onMouseEnter={() => setHover(i)}
                  onMouseLeave={() => setHover(-1)}
                  style={css(
                    `position: relative; padding: 15px 14px; background: ${
                      on ? '#FDFBF6' : '#F3EFE6'
                    }; border: 1px solid ${on ? node.dot : 'rgba(43,37,33,0.2)'}; box-shadow: ${
                      on ? `4px 4px 0 ${node.dot}33` : 'none'
                    }; transform: translateY(${on ? -3 : 0}px); transition: all .25s;`,
                  )}
                >
                  <div style={css('display: flex; align-items: center; justify-content: space-between; gap: 8px;')}>
                    <span
                      style={css(
                        "font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 0.16em; color: rgba(43,37,33,0.45);",
                      )}
                    >
                      {`0${i + 1}`}
                    </span>
                    <span
                      style={css(
                        `width: 8px; height: 8px; background: ${node.dot}; box-shadow: 0 0 0 ${
                          on ? 5 : 3
                        }px ${node.dot}26; transition: box-shadow .25s;`,
                      )}
                    />
                  </div>
                  <div
                    style={css(
                      'font-family: Spectral, serif; font-size: 16px; font-weight: 500; letter-spacing: -0.01em; margin-top: 12px; line-height: 1.22;',
                    )}
                  >
                    {node.title}
                  </div>
                  <div
                    style={css(
                      `font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.12em; margin-top: 7px; color: ${node.accent};`,
                    )}
                  >
                    {node.meta}
                  </div>
                  <div
                    style={css(
                      `font-size: 12.5px; line-height: 1.6; color: rgba(43,37,33,0.7); overflow: hidden; max-height: ${
                        on ? 150 : 0
                      }px; opacity: ${on ? 1 : 0}; margin-top: ${on ? 11 : 0}px; transition: all .3s;`,
                    )}
                  >
                    {node.body}
                  </div>
                </div>
              );
            })}
          </div>

          <div
            style={css(
              'display: flex; gap: 24px; margin-top: 26px; flex-wrap: wrap; padding-top: 18px; border-top: 1px solid rgba(43,37,33,0.16);',
            )}
          >
            {LEGEND.map((l) => (
              <div key={l.t} style={css('display: flex; align-items: center; gap: 9px; font-size: 12px; color: rgba(43,37,33,0.68);')}>
                <span style={css(`width: 20px; height: 2px; background: ${l.c};`)} />
                {l.t}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
});

PipelineSection.displayName = 'PipelineSection';

export default PipelineSection;
