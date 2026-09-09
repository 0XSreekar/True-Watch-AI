import { useEffect, useState } from 'react';

import { feed } from '../../canvas';
import { FALLBACK_CHAIN } from '../../data/fallbacks';
import { evidenceApi, withFallback } from '../../services';
import { css } from '../../utils/css';
import SceneCanvas from '../common/SceneCanvas';
import SectionHeading from '../common/SectionHeading';

const ChainBlock = ({ block, index, hovered, onEnter, onLeave }) => {
  const on = hovered;
  return (
    <div style={css('display: flex; align-items: center; flex: 0 0 auto;')}>
      <div
        onMouseEnter={onEnter}
        onMouseLeave={onLeave}
        style={css(
          `flex: 0 0 auto; width: 224px; padding: 15px; background: ${
            on ? '#FDFBF6' : '#FAF7F0'
          }; border: 1px solid ${
            on ? '#6E4B7C' : 'rgba(110,75,124,0.35)'
          }; box-shadow: ${
            on ? '5px 5px 0 rgba(110,75,124,0.28)' : 'none'
          }; transform: translateY(${on ? -5 : 0}px); transition: all .28s;`,
        )}
      >
        <div
          style={css(
            `position: relative; height: ${on ? 92 : 0}px; margin-bottom: ${
              on ? 13 : 0
            }px; overflow: hidden; opacity: ${on ? 1 : 0}; transition: all .3s; background: #241F1B;`,
          )}
        >
          {on ? (
            <SceneCanvas
              scene={feed}
              options={{ seed: index * 1.7, ir: index % 2 === 1, boxes: 1, labels: false }}
            />
          ) : null}
          <span
            style={css(
              "position: absolute; left: 7px; bottom: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: rgba(239,234,221,0.9);",
            )}
          >
            {block.camera}
          </span>
        </div>

        <div style={css('display: flex; align-items: center; justify-content: space-between;')}>
          <span style={css("font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.14em; color: #6E4B7C;")}>
            BLOCK {block.n}
          </span>
          <span
            style={css(
              'width: 17px; height: 17px; border: 1px solid rgba(110,75,124,0.5); display: grid; place-items: center; font-size: 9px; color: #6E4B7C;',
            )}
          >
            &#9852;
          </span>
        </div>
        <div
          style={css(
            "font-family: 'IBM Plex Mono', monospace; font-size: 11px; margin-top: 10px; color: #2B2521; word-break: break-all; line-height: 1.55;",
          )}
        >
          {block.hash}
        </div>
        <div
          style={css(
            "font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; margin-top: 7px; letter-spacing: 0.08em; color: rgba(43,37,33,0.52);",
          )}
        >
          {block.time}
        </div>
      </div>
      <div
        style={css(
          'width: 36px; height: 1px; background: repeating-linear-gradient(90deg, rgba(110,75,124,0.65) 0 5px, transparent 5px 9px); flex: 0 0 auto;',
        )}
      />
    </div>
  );
};

export const EvidenceSection = () => {
  const [chain, setChain] = useState(FALLBACK_CHAIN);
  const [hover, setHover] = useState(-1);

  useEffect(() => {
    let alive = true;
    withFallback(evidenceApi.fetchChain(), FALLBACK_CHAIN, 'evidence chain').then((data) => {
      if (alive) setChain(data);
    });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <section
      style={css(
        'position: relative; padding: 92px 26px 100px; background: #F3EFE6; border-top: 1px solid rgba(43,37,33,0.18);',
      )}
    >
      <div style={css('max-width: 1300px; margin: 0 auto;')}>
        <SectionHeading index="05" label="EVIDENCE INTEGRITY" color="#6E4B7C" />
        <h2
          style={css(
            'font-family: Spectral, serif; font-weight: 500; font-size: clamp(28px,3.6vw,47px); letter-spacing: -0.02em; line-height: 1.1; margin: 20px 0 0; max-width: 760px;',
          )}
        >
          Sealed at the moment of capture, not at the moment of upload.
        </h2>
        <p
          style={css(
            'font-family: Spectral, serif; font-size: 17px; line-height: 1.66; color: rgba(43,37,33,0.7); max-width: 620px; margin: 15px 0 0;',
          )}
        >
          Each clip&apos;s SHA-256 digest carries the previous digest forward. Alter one frame and
          every block after it breaks — which is what makes the footage admissible.
        </p>

        <div style={css('display: flex; margin-top: 44px; overflow-x: auto; padding: 6px 2px 22px;')}>
          {chain.map((block, i) => (
            <ChainBlock
              key={block.n}
              block={block}
              index={i}
              hovered={hover === i}
              onEnter={() => setHover(i)}
              onLeave={() => setHover(-1)}
            />
          ))}
          <div
            style={css(
              "display: grid; place-items: center; flex: 0 0 auto; width: 128px; height: 124px; border: 1px dashed rgba(110,75,124,0.55); font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.1em; color: rgba(110,75,124,0.9); text-align: center; line-height: 1.8; padding: 12px;",
            )}
          >
            NEXT BLOCK
            <br />
            ON CAPTURE
          </div>
        </div>
      </div>
    </section>
  );
};

export default EvidenceSection;
