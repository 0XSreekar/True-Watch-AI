import { css } from '../../utils/css';

/** Blocks for the selected event, plus the walk-the-chain verification. */
export const EvidenceChainPanel = ({ chain, verifying, verified, onVerify }) => {
  const allVerified = verified >= chain.length && chain.length > 0;

  return (
    <div>
      <div style={css("font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.16em; color: #6E4B7C; margin-bottom: 10px;")}>
        EVIDENCE CHAIN — SELECTED EVENT
      </div>
      <div style={css('padding: 16px; border: 1px solid rgba(43,37,33,0.24); background: #FAF7F0;')}>
        <div style={css('display: flex; flex-direction: column; gap: 8px;')}>
          {chain.map((block, i) => {
            const done = i < verified;
            const active = verifying === i;
            return (
              <div
                key={block.n}
                style={css(
                  `display: flex; gap: 11px; padding: 11px 12px; background: ${
                    done ? 'rgba(31,111,122,0.07)' : '#FAF7F0'
                  }; border: 1px solid ${
                    done ? 'rgba(31,111,122,0.5)' : active ? '#6E4B7C' : 'rgba(110,75,124,0.28)'
                  }; box-shadow: ${
                    active ? '3px 3px 0 rgba(110,75,124,0.3)' : 'none'
                  }; transition: all .28s;`,
                )}
              >
                <span
                  style={css(
                    `display: grid; place-items: center; width: 24px; height: 24px; flex: 0 0 auto; font-size: 11px; color: ${
                      done || active ? '#FAF7F0' : '#6E4B7C'
                    }; background: ${
                      done ? '#1F6F7A' : active ? '#6E4B7C' : 'transparent'
                    }; border: 1px solid ${done ? '#1F6F7A' : '#6E4B7C'}; transition: all .28s;`,
                  )}
                >
                  {done ? '✓' : active ? '◌' : '⛓'}
                </span>
                <div style={css('min-width: 0;')}>
                  <div style={css("font-family: 'IBM Plex Mono', monospace; font-size: 10.5px;")}>
                    BLOCK {block.n} · {block.camera}
                  </div>
                  <div
                    style={css(
                      "font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: rgba(43,37,33,0.5); margin-top: 2px; word-break: break-all;",
                    )}
                  >
                    {block.hash} · {block.time}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <button
          type="button"
          onClick={onVerify}
          style={css(
            `width: 100%; margin-top: 13px; padding: 12px; border: 1px solid ${
              allVerified ? '#1F6F7A' : '#6E4B7C'
            }; background: ${
              allVerified ? '#1F6F7A' : '#6E4B7C'
            }; color: #FAF7F0; font-size: 11.5px; letter-spacing: 0.11em; cursor: pointer; box-shadow: 3px 3px 0 rgba(43,37,33,0.14); transition: all .3s;`,
          )}
        >
          {allVerified
            ? `CHAIN INTACT — ${chain.length} / ${chain.length} VERIFIED`
            : verifying >= 0
              ? `VERIFYING BLOCK ${verifying + 1} / ${chain.length}…`
              : 'VERIFY CHAIN'}
        </button>
      </div>
    </div>
  );
};

export default EvidenceChainPanel;
