import { feed } from '../../canvas';
import { DORI_BANDS } from '../../data/landingContent';
import { css } from '../../utils/css';
import SceneCanvas from '../common/SceneCanvas';

/** The 3-up camera wall. Clicking a tile expands it across two columns. */
export const LiveWall = ({ cameras, sector, expanded, onExpand, onClearSector }) => (
  <div>
    <div style={css('display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 10px;')}>
      <span style={css("font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.16em; color: #8A6A22;")}>
        LIVE WALL · {sector === 'all' ? 'ALL SECTORS' : `${sector.toUpperCase()} SECTOR`}
      </span>
      {sector !== 'all' ? (
        <a
          href="#clear"
          onClick={(e) => {
            e.preventDefault();
            onClearSector();
          }}
          style={css('font-size: 11px; color: #A32E24;')}
        >
          CLEAR FILTER
        </a>
      ) : null}
    </div>

    <div style={css('display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;')}>
      {cameras.map((cam, i) => {
        const open = expanded === i;
        return (
          <div
            key={cam.id}
            onClick={() => onExpand(open ? -1 : i)}
            style={css(
              `position: relative; overflow: hidden; cursor: pointer; background: #241F1B; border: 1px solid ${
                open ? '#A9812F' : 'rgba(43,37,33,0.32)'
              }; box-shadow: ${
                open ? '5px 5px 0 rgba(169,129,47,0.4)' : 'none'
              }; grid-column: ${open ? 'span 2' : 'span 1'}; aspect-ratio: ${
                open ? '32 / 11' : '16 / 10'
              }; transition: all .4s cubic-bezier(.2,.8,.2,1);`,
            )}
          >
            <SceneCanvas scene={feed} options={{ seed: i * 2.3, ir: cam.ir, boxes: 1 + (i % 3) }} />
            <span
              style={css(
                "position: absolute; left: 8px; top: 7px; padding: 3px 7px; font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 0.08em; color: #ECE6D8; background: rgba(28,24,21,0.55); border: 1px solid rgba(236,230,216,0.3);",
              )}
            >
              {cam.id}
            </span>
            <span
              style={css(
                `position: absolute; right: 8px; top: 7px; padding: 3px 7px; font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 0.12em; color: #EFEADD; border: 1px solid ${
                  cam.ir ? 'rgba(110,168,178,0.75)' : 'rgba(196,164,85,0.75)'
                }; background: rgba(36,31,27,0.55);`,
              )}
            >
              {cam.ir ? 'IR' : 'DAY'}
            </span>
            <div style={css('position: absolute; left: 0; right: 0; bottom: 0; display: flex; height: 4px;')}>
              {DORI_BANDS.map((band) => (
                <div
                  key={band.k}
                  style={css(`flex: 0 0 auto; height: 100%; background: ${band.c}; width: ${band.w}%;`)}
                />
              ))}
            </div>
            <span
              style={css(
                "position: absolute; left: 8px; bottom: 8px; font-family: 'IBM Plex Mono', monospace; font-size: 8.5px; color: rgba(236,230,216,0.85);",
              )}
            >
              {cam.name}
            </span>
          </div>
        );
      })}
    </div>
  </div>
);

export default LiveWall;
