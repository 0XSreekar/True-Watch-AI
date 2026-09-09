import { feed } from '../../canvas';
import { css } from '../../utils/css';
import SceneCanvas from '../common/SceneCanvas';

/** Plain-language search over stored footage, with its own result strip. */
export const FootageSearch = ({ query, onQueryChange, placeholder, onRun, searching, results }) => (
  <div>
    <div style={css("font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.16em; color: #8A6A22; margin-bottom: 10px;")}>
      FOOTAGE SEARCH — PLAIN LANGUAGE
    </div>

    <div style={css('display: flex; gap: 8px;')}>
      <div
        style={css(
          'flex: 1; display: flex; align-items: center; gap: 8px; padding: 12px 14px; border: 1px solid rgba(43,37,33,0.24); background: #FAF7F0;',
        )}
      >
        <span style={css('font-size: 13px; color: rgba(43,37,33,0.4);')}>⌕</span>
        <input
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && onRun()}
          placeholder={placeholder}
          style={css('border: none; outline: none; background: transparent; font-size: 13px; flex: 1; color: #2B2521;')}
        />
      </div>
      <button
        type="button"
        onClick={onRun}
        style={css(
          'padding: 12px 20px; border: 1px solid #2B2521; background: #2B2521; color: #F3EFE6; font-size: 12px; letter-spacing: 0.08em; cursor: pointer;',
        )}
      >
        SEARCH
      </button>
    </div>

    <div style={css(`display: ${searching > 0 ? 'block' : 'none'}; margin-top: 13px;`)}>
      <div
        style={css(
          "display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.08em; color: rgba(43,37,33,0.55); margin-bottom: 6px;",
        )}
      >
        <span>SEARCHING 6 WEEKS OF FOOTAGE…</span>
        <span>{Math.round(searching)}%</span>
      </div>
      <div style={css('height: 5px; background: rgba(43,37,33,0.1);')}>
        <div style={css(`height: 100%; width: ${searching}%; background: #A9812F; transition: width .2s;`)} />
      </div>
    </div>

    {results && results.length ? (
      <div style={css('display: flex; gap: 10px; margin-top: 16px; overflow-x: auto; padding-bottom: 6px;')}>
        {results.map((r, i) => (
          <div
            key={r.id}
            style={css(
              `flex: 0 0 auto; width: 212px; background: #FAF7F0; border: 1px solid rgba(43,37,33,0.22); box-shadow: 3px 3px 0 rgba(43,37,33,0.06); animation: sdRise .4s ${
                i * 0.07
              }s ease both; cursor: pointer;`,
            )}
          >
            <div style={css('position: relative; height: 90px;')}>
              <SceneCanvas
                scene={feed}
                options={{ seed: i * 3.1, ir: r.camera.ir, boxes: 1, labels: false }}
              />
              <span
                style={css(
                  "position: absolute; right: 6px; top: 6px; padding: 2px 6px; font-family: 'IBM Plex Mono', monospace; font-size: 8.5px; color: #ECE6D8; background: rgba(28,24,21,0.6);",
                )}
              >
                {r.score}% MATCH
              </span>
            </div>
            <div style={css('padding: 9px 10px;')}>
              <div style={css("font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: rgba(43,37,33,0.5);")}>
                {r.time}
              </div>
              <div style={css('font-size: 11.5px; font-weight: 500; margin-top: 3px;')}>
                {r.camera.name}
              </div>
              <div style={css('display: flex; flex-wrap: wrap; gap: 4px; margin-top: 7px;')}>
                {r.chips.map((chip) => (
                  <span
                    key={chip}
                    style={css(
                      "padding: 2px 6px; font-family: 'IBM Plex Mono', monospace; font-size: 8.5px; color: #8A6A22; border: 1px solid rgba(169,129,47,0.4);",
                    )}
                  >
                    {chip}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    ) : null}
  </div>
);

export default FootageSearch;
