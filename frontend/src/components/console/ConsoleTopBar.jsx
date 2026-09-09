import { css } from '../../utils/css';

const BudgetMeter = ({ budget }) => {
  const { used, total, near } = budget;
  return (
    <div
      style={css(
        `display: flex; flex-direction: column; gap: 7px; min-width: 248px; padding: 9px 13px; background: #FAF7F0; border: 1px solid ${
          near ? '#B57324' : 'rgba(43,37,33,0.22)'
        }; transition: all .4s;`,
      )}
    >
      <div style={css('display: flex; justify-content: space-between; align-items: baseline;')}>
        <span style={css("font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 0.1em; color: rgba(43,37,33,0.5);")}>
          SHIFT ALARM BUDGET
        </span>
        <span style={css("font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 500;")}>
          {used} / {total}
        </span>
      </div>
      <div style={css('display: flex; height: 18px;')}>
        {Array.from({ length: total }, (_, i) => (
          <div
            key={i}
            style={css(
              `flex: 1; height: 18px; background: ${
                i < used
                  ? near && i >= total - 3
                    ? '#B57324'
                    : '#1F6F7A'
                  : 'rgba(43,37,33,0.1)'
              }; border-right: 1px solid #F3EFE6; transition: background .4s ${i * 0.03}s;`,
            )}
          />
        ))}
      </div>
      <span
        style={css(
          `font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.13em; color: ${
            near ? '#7A4E14' : 'rgba(43,37,33,0.5)'
          };`,
        )}
      >
        {near ? 'APPROACHING SHIFT LIMIT' : 'ALERTS RAISED THIS SHIFT'}
      </span>
    </div>
  );
};

export const ConsoleTopBar = ({ query, onQueryChange, onSubmitQuery, placeholder, budget, clock, operator }) => (
  <div
    style={css(
      'position: sticky; top: 0; z-index: 30; display: flex; align-items: center; gap: 18px; padding: 12px 24px; background: rgba(243,239,230,0.95); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(43,37,33,0.18); flex-wrap: wrap;',
    )}
  >
    <div
      style={css(
        'display: flex; align-items: center; gap: 8px; padding: 8px 12px; border: 1px solid rgba(43,37,33,0.22); background: #FAF7F0; flex: 1; min-width: 220px; max-width: 380px;',
      )}
    >
      <span style={css('font-size: 12px; color: rgba(43,37,33,0.4);')}>⌕</span>
      <input
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && onSubmitQuery()}
        placeholder={placeholder}
        style={css('border: none; outline: none; background: transparent; font-size: 12.5px; flex: 1; color: #2B2521;')}
      />
    </div>

    <BudgetMeter budget={budget} />

    <div style={css('flex: 1;')} />

    <div
      style={css(
        "display: flex; align-items: center; gap: 14px; font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.06em; color: rgba(43,37,33,0.55);",
      )}
    >
      <span>{clock}</span>
      <span style={css('display: flex; align-items: center; gap: 6px;')}>
        <span style={css('width: 6px; height: 6px; background: #1F6F7A;')} />
        LINK OK
      </span>
    </div>

    <div
      style={css(
        'display: flex; align-items: center; gap: 9px; padding-left: 14px; border-left: 1px solid rgba(43,37,33,0.18);',
      )}
    >
      <div
        style={css(
          'width: 28px; height: 28px; border: 1px solid #2B2521; display: grid; place-items: center; font-family: Spectral, serif; font-size: 12px; background: #2B2521; color: #FAF7F0;',
        )}
      >
        {operator.initials}
      </div>
      <div style={css('display: flex; flex-direction: column; line-height: 1.25;')}>
        <span style={css('font-size: 11.5px; font-weight: 500;')}>{operator.name}</span>
        <span style={css("font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: rgba(43,37,33,0.5);")}>
          {operator.role}
        </span>
      </div>
    </div>
  </div>
);

export default ConsoleTopBar;
