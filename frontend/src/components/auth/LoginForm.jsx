import { css } from '../../utils/css';
import Field from './Field';
import PasswordToggle from './PasswordToggle';

export const LoginForm = ({ form, setField, errors, ok, showPw, toggleShowPw, remember, toggleRemember }) => (
  <div style={css('display: flex; flex-direction: column; gap: 13px;')}>
    <Field
      label="POST ID / OFFICIAL ID"
      value={form.id}
      onChange={(v) => setField('id', v)}
      placeholder="SSB/104382"
      error={errors.id}
      valid={ok.id}
    />

    <Field
      label="PASSWORD"
      type={showPw ? 'text' : 'password'}
      value={form.pw}
      onChange={(v) => setField('pw', v)}
      placeholder="••••••••••"
      error={errors.pw}
      valid={ok.pw}
      showCheck={false}
    >
      <PasswordToggle shown={showPw} onToggle={toggleShowPw} />
    </Field>

    <div style={css('display: flex; align-items: center; justify-content: space-between; margin-top: 3px;')}>
      <div
        onClick={toggleRemember}
        style={css('display: flex; align-items: center; gap: 9px; cursor: pointer;')}
      >
        <span
          style={css(
            `position: relative; width: 34px; height: 18px; cursor: pointer; border: 1px solid ${
              remember ? '#2B2521' : 'rgba(43,37,33,0.32)'
            }; background: ${remember ? '#2B2521' : 'transparent'}; transition: all .25s;`,
          )}
        >
          <span
            style={css(
              `position: absolute; top: 2px; left: ${
                remember ? 18 : 2
              }px; width: 12px; height: 12px; background: ${
                remember ? '#C4A455' : 'rgba(43,37,33,0.4)'
              }; transition: all .25s;`,
            )}
          />
        </span>
        <span style={css('font-size: 12.5px; color: rgba(43,37,33,0.68);')}>
          Remember this terminal
        </span>
      </div>
    </div>
  </div>
);

export default LoginForm;
