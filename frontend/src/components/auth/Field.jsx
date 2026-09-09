import { css } from '../../utils/css';

const labelStyle = css(
  "display: block; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.08em; color: rgba(43,37,33,0.58); margin-bottom: 7px;",
);

export const fieldStyles = (bad, good) => ({
  input: `width: 100%; padding: 13px 40px 13px 13px; border: 1px solid ${
    bad ? '#A32E24' : good ? '#1F6F7A' : 'rgba(43,37,33,0.3)'
  }; background: ${bad ? 'rgba(163,46,36,0.04)' : '#FDFBF6'}; font-size: 14px; color: #2B2521; outline: none; transition: all .2s;`,
  err: `font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.06em; color: #A32E24; margin-top: 6px; height: ${
    bad ? 15 : 0
  }px; opacity: ${bad ? 1 : 0}; overflow: hidden; transition: all .22s;`,
  check: `position: absolute; right: 12px; top: 36px; width: 17px; height: 17px; border: 1px solid #1F6F7A; background: #1F6F7A; color: #FAF7F0; font-size: 10px; display: grid; place-items: center; opacity: ${
    good ? 1 : 0
  }; transform: scale(${good ? 1 : 0.5}); transition: all .25s;`,
});

/** Labelled input with the inline tick / error affordances from the prototype. */
export const Field = ({
  label,
  value,
  onChange,
  placeholder,
  type = 'text',
  error,
  valid,
  showCheck = true,
  children,
}) => {
  const bad = !!error;
  const good = !!valid && !bad;
  const s = fieldStyles(bad, good);

  return (
    <div style={css('position: relative;')}>
      <label style={labelStyle}>{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        style={css(s.input)}
      />
      {showCheck ? <span style={css(s.check)}>✓</span> : null}
      {children}
      <div style={css(s.err)}>{error || ''}</div>
    </div>
  );
};

export default Field;
