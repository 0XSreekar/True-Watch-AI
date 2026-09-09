import { css } from '../../utils/css';

export const SubmitRow = ({ label, loading, onSubmit, showBack, onBack }) => (
  <>
    <button
      type="button"
      onClick={onSubmit}
      style={css(
        `width: 100%; margin-top: 22px; padding: 15px; border: 1px solid #2B2521; background: ${
          loading ? 'rgba(43,37,33,0.55)' : '#2B2521'
        }; color: #F3EFE6; font-size: 12.5px; letter-spacing: 0.11em; cursor: ${
          loading ? 'wait' : 'pointer'
        }; box-shadow: 4px 4px 0 rgba(169,129,47,0.5); transition: all .2s;`,
      )}
    >
      {label}
    </button>
    <button
      type="button"
      onClick={onBack}
      style={css(
        `display: ${
          showBack ? 'block' : 'none'
        }; width: 100%; margin-top: 9px; padding: 12px; border: 1px solid rgba(43,37,33,0.3); background: transparent; color: #2B2521; font-size: 12px; letter-spacing: 0.09em; cursor: pointer;`,
      )}
    >
      BACK
    </button>
  </>
);

export default SubmitRow;
