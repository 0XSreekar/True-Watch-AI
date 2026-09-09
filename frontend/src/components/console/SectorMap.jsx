import { plan } from '../../canvas';
import { css } from '../../utils/css';
import SceneCanvas from '../common/SceneCanvas';

/** Post markers over the survey plan. Clicking one filters the live wall. */
export const SectorMap = ({ posts, sector, onSelect }) => (
  <div>
    <div style={css("font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.16em; color: #6E4B7C; margin-bottom: 10px;")}>
      SECTOR POST MAP
    </div>
    <div
      style={css(
        'position: relative; height: 260px; border: 1px solid rgba(43,37,33,0.24); background: #FAF7F0; overflow: hidden;',
      )}
    >
      <SceneCanvas scene={plan} />
      {posts.map((p) => {
        const selected = sector === p.name;
        return (
          <div
            key={p.id}
            onClick={() => onSelect(selected ? 'all' : p.name)}
            style={css(
              `position: absolute; left: ${p.x}%; top: ${
                p.y
              }%; transform: translate(-50%,-50%); display: flex; flex-direction: column; align-items: center; gap: 4px; cursor: pointer; z-index: ${
                selected ? 5 : 2
              };`,
            )}
          >
            <span
              style={css(
                `position: relative; width: ${selected ? 13 : 10}px; height: ${
                  selected ? 13 : 10
                }px; border: 1px solid #2B2521; background: ${
                  p.alert ? '#A32E24' : selected ? '#A9812F' : '#1F6F7A'
                }; transition: all .25s;`,
              )}
            >
              <span
                style={css(
                  `position: absolute; inset: -5px; border: 1px solid ${
                    p.alert ? '#A32E24' : '#1F6F7A'
                  }; animation: sdRing ${p.alert ? 1.5 : 2.8}s ease-out infinite;`,
                )}
              />
            </span>
            <span
              style={css(
                `padding: 2px 6px; white-space: nowrap; font-family: 'IBM Plex Mono', monospace; font-size: 8.5px; letter-spacing: 0.08em; color: ${
                  selected ? '#FAF7F0' : '#2B2521'
                }; background: ${
                  selected ? '#2B2521' : 'rgba(250,247,240,0.92)'
                }; border: 1px solid rgba(43,37,33,0.28); transition: all .25s;`,
              )}
            >
              {p.name}
            </span>
            <span
              style={css(
                `font-family: 'IBM Plex Mono', monospace; font-size: 8px; letter-spacing: 0.1em; color: ${
                  p.road ? '#1F6F7A' : '#A32E24'
                };`,
              )}
            >
              {p.road ? 'ROAD' : 'NO ROAD'}
            </span>
          </div>
        );
      })}
    </div>
  </div>
);

export default SectorMap;
