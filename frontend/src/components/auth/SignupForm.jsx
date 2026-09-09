import { css } from '../../utils/css';
import Field, { fieldStyles } from './Field';
import PasswordMeter from './PasswordMeter';
import PasswordToggle from './PasswordToggle';

const slide = (visible) =>
  `display: ${visible ? 'flex' : 'none'}; flex-direction: column; gap: 13px; animation: sdRise .35s ease both;`;

const labelStyle = css(
  "display: block; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.08em; color: rgba(43,37,33,0.58); margin-bottom: 7px;",
);

export const SignupForm = ({
  step,
  form,
  setField,
  errors,
  ok,
  showPw,
  toggleShowPw,
  pwScore,
  units,
  roles,
}) => (
  <div>
    {/* Step 1 — identity */}
    <div style={css(slide(step === 0))}>
      <Field
        label="FULL NAME"
        value={form.name}
        onChange={(v) => setField('name', v)}
        placeholder="Ranjit Kumar Thapa"
        error={errors.name}
        valid={ok.name}
      />
      <Field
        label="OFFICIAL ID"
        value={form.id}
        onChange={(v) => setField('id', v)}
        placeholder="SSB/104382"
        error={errors.id}
        valid={ok.id}
      />
      <div style={css('position: relative;')}>
        <label style={labelStyle}>UNIT / SECTOR</label>
        <select
          value={form.unit}
          onChange={(e) => setField('unit', e.target.value)}
          style={css(fieldStyles(!!errors.unit, !!ok.unit && !errors.unit).input)}
        >
          <option value="">Select your posting</option>
          {units.map((u) => (
            <option key={u} value={u}>
              {u}
            </option>
          ))}
        </select>
        <div style={css(fieldStyles(!!errors.unit, false).err)}>{errors.unit || ''}</div>
      </div>
    </div>

    {/* Step 2 — credentials */}
    <div style={css(slide(step === 1))}>
      <Field
        label="OFFICIAL EMAIL"
        value={form.email}
        onChange={(v) => setField('email', v)}
        placeholder="r.thapa@ssb.gov.in"
        error={errors.email}
        valid={ok.email}
      />
      <Field
        label="PASSWORD"
        type={showPw ? 'text' : 'password'}
        value={form.np}
        onChange={(v) => setField('np', v)}
        placeholder="Minimum 10 characters"
        error={errors.np}
        valid={ok.np}
        showCheck={false}
      >
        <PasswordToggle shown={showPw} onToggle={toggleShowPw} />
        <PasswordMeter score={pwScore} />
      </Field>
      <Field
        label="CONFIRM PASSWORD"
        type={showPw ? 'text' : 'password'}
        value={form.cp}
        onChange={(v) => setField('cp', v)}
        placeholder="Repeat password"
        error={errors.cp}
        valid={ok.cp}
      />
    </div>

    {/* Step 3 — role and undertaking */}
    <div style={css(slide(step === 2))}>
      <div>
        <label
          style={css(
            "display: block; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.08em; color: rgba(43,37,33,0.58); margin-bottom: 9px;",
          )}
        >
          ROLE
        </label>
        <div style={css('display: flex; flex-direction: column; gap: 8px;')}>
          {roles.map((r) => {
            const picked = form.role === r.k;
            return (
              <div
                key={r.k}
                onClick={() => setField('role', r.k)}
                style={css(
                  `display: flex; align-items: center; gap: 12px; padding: 12px 14px; cursor: pointer; border: 1px solid ${
                    picked ? '#2B2521' : 'rgba(43,37,33,0.24)'
                  }; background: ${
                    picked ? 'rgba(169,129,47,0.12)' : 'transparent'
                  }; transition: all .22s;`,
                )}
              >
                <span
                  style={css(
                    `width: 14px; height: 14px; flex: 0 0 auto; border: 1px solid ${
                      picked ? '#2B2521' : 'rgba(43,37,33,0.36)'
                    }; background: ${
                      picked ? '#A9812F' : 'transparent'
                    }; box-shadow: inset 0 0 0 2px #FAF7F0; transition: all .22s;`,
                  )}
                />
                <span style={css('display: flex; flex-direction: column; gap: 2px;')}>
                  <span style={css('font-size: 13px; font-weight: 500;')}>{r.k}</span>
                  <span style={css('font-size: 11.5px; color: rgba(43,37,33,0.56);')}>{r.dd}</span>
                </span>
              </div>
            );
          })}
        </div>
        <div style={css(fieldStyles(!!errors.role, false).err)}>{errors.role || ''}</div>
      </div>

      <div
        onClick={() => setField('terms', !form.terms)}
        style={css('display: flex; align-items: flex-start; gap: 10px; cursor: pointer; margin-top: 6px;')}
      >
        <span
          style={css(
            `width: 18px; height: 18px; flex: 0 0 auto; display: grid; place-items: center; cursor: pointer; font-size: 11px; color: #FAF7F0; border: 1px solid ${
              form.terms ? '#2B2521' : 'rgba(43,37,33,0.36)'
            }; background: ${form.terms ? '#2B2521' : 'transparent'}; transition: all .2s;`,
          )}
        >
          {form.terms ? '✓' : ''}
        </span>
        <span style={css('font-size: 12px; line-height: 1.55; color: rgba(43,37,33,0.66);')}>
          I confirm this account is for official SSB duty and accept the terminal usage undertaking.
        </span>
      </div>
      <div style={css(fieldStyles(!!errors.terms, false).err)}>{errors.terms || ''}</div>
    </div>
  </div>
);

export default SignupForm;
