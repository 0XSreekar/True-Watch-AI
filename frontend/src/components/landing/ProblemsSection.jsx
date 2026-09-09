import { useState } from 'react';

import { micro } from '../../canvas';
import { PROBLEMS } from '../../data/landingContent';
import { css } from '../../utils/css';
import SceneCanvas from '../common/SceneCanvas';
import SectionHeading from '../common/SectionHeading';

const CELL = 'position: relative; padding: 22px; border-right: 1px solid rgba(43,37,33,0.2); background: #FAF7F0;';

const ProblemCard = ({ problem }) => {
  const [tilt, setTilt] = useState(null);

  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    setTilt([
      ((e.clientY - r.top) / r.height - 0.5) * -5,
      ((e.clientX - r.left) / r.width - 0.5) * 5,
    ]);
  };

  return (
    <div
      onMouseMove={onMove}
      onMouseLeave={() => setTilt(null)}
      style={css(
        `${CELL}transform: perspective(1000px) rotateX(${tilt ? tilt[0] : 0}deg) rotateY(${
          tilt ? tilt[1] : 0
        }deg) translateY(${tilt ? -4 : 0}px); background: ${
          tilt ? '#FDFBF6' : '#FAF7F0'
        }; box-shadow: ${
          tilt ? 'inset 0 0 0 1px rgba(169,129,47,0.55)' : 'none'
        }; transition: transform .3s ease-out, background .3s, box-shadow .3s;`,
      )}
    >
      <div
        style={css(
          'position: relative; height: 128px; overflow: hidden; border: 1px solid rgba(43,37,33,0.18); background: #EFEADD;',
        )}
      >
        <SceneCanvas scene={micro} options={{ kind: problem.kind }} />
        <span
          style={css(
            "position: absolute; left: 8px; top: 7px; font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 0.16em; color: rgba(43,37,33,0.55);",
          )}
        >
          {problem.fig}
        </span>
      </div>

      <div style={css('display: flex; align-items: center; gap: 10px; margin-top: 18px;')}>
        <span style={css(`width: 18px; height: 1px; background: ${problem.accent};`)} />
        <span
          style={css(
            `font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.18em; color: ${problem.accent};`,
          )}
        >
          {problem.tag}
        </span>
      </div>

      <h3
        style={css(
          'font-family: Spectral, serif; font-weight: 500; font-size: 21px; letter-spacing: -0.01em; margin: 11px 0 0; line-height: 1.26;',
        )}
      >
        {problem.title}
      </h3>
      <p style={css('font-size: 14px; line-height: 1.66; color: rgba(43,37,33,0.7); margin: 10px 0 0;')}>
        {problem.body}
      </p>
    </div>
  );
};

export const ProblemsSection = () => (
  <section
    style={css(
      'position: relative; padding: 96px 26px; background: #F3EFE6; border-top: 1px solid rgba(43,37,33,0.18);',
    )}
  >
    <div style={css('position: relative; max-width: 1220px; margin: 0 auto;')}>
      <SectionHeading index="01" label="THE GROUND TRUTH" />
      <h2
        style={css(
          'font-family: Spectral, serif; font-weight: 500; font-size: clamp(28px,3.6vw,47px); letter-spacing: -0.02em; line-height: 1.1; margin: 20px 0 0; max-width: 740px;',
        )}
      >
        A border where crossing is legal, and recording is not enough.
      </h2>
      <div
        style={css(
          'display: grid; grid-template-columns: repeat(auto-fit, minmax(298px, 1fr)); margin-top: 44px; border: 1px solid rgba(43,37,33,0.2);',
        )}
      >
        {PROBLEMS.map((problem) => (
          <ProblemCard key={problem.fig} problem={problem} />
        ))}
      </div>
    </div>
  </section>
);

export default ProblemsSection;
