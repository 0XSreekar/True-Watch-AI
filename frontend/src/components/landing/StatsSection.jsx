import { STATS } from '../../data/landingContent';
import useInView from '../../hooks/useInView';
import { css } from '../../utils/css';

/** Dark band of headline figures; the numbers flip in when scrolled into view. */
export const StatsSection = () => {
  const [ref, inView] = useInView();

  return (
    <section ref={ref} style={css('position: relative; background: #2B2521; color: #EFEADD;')}>
      <div style={css('max-width: 1220px; margin: 0 auto; padding: 46px 26px;')}>
        <div style={css('display: grid; grid-template-columns: repeat(auto-fit, minmax(178px,1fr));')}>
          {STATS.map((stat, i) => (
            <div
              key={stat.label}
              style={css(
                `padding: 18px 20px; border-left: 1px solid ${
                  i === 0 ? 'transparent' : 'rgba(239,234,221,0.2)'
                };`,
              )}
            >
              <div
                style={css(
                  'display: flex; align-items: baseline; gap: 3px; font-family: Spectral, serif; font-weight: 400; font-size: clamp(32px,3.3vw,46px); letter-spacing: -0.025em; color: #FAF7F0;',
                )}
              >
                <span>{stat.prefix || ''}</span>
                <span>
                  {inView
                    ? stat.target >= 1000
                      ? stat.target.toLocaleString('en-IN')
                      : `${stat.target}`
                    : '0'}
                </span>
                <span
                  style={css(
                    "font-size: 0.45em; font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.08em; color: #C4A455; margin-left: 5px;",
                  )}
                >
                  {stat.suffix}
                </span>
              </div>
              <div
                style={css(
                  'font-size: 12.5px; line-height: 1.55; color: rgba(239,234,221,0.62); margin-top: 9px; max-width: 200px;',
                )}
              >
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default StatsSection;
