import AlertQueue from '../components/console/AlertQueue';
import AnalyticsPanel from '../components/console/AnalyticsPanel';
import ConsoleRail from '../components/console/ConsoleRail';
import ConsoleTopBar from '../components/console/ConsoleTopBar';
import EvidenceChainPanel from '../components/console/EvidenceChainPanel';
import FootageSearch from '../components/console/FootageSearch';
import LiveWall from '../components/console/LiveWall';
import SectorMap from '../components/console/SectorMap';
import useConsole from '../hooks/useConsole';
import { css } from '../utils/css';

// Replace with the signed-in user once auth is real.
const OPERATOR = { name: 'Ct. R. K. Thapa', role: 'Operator · 20th Bn Raxaul', initials: 'RT' };

export const ConsolePage = () => {
  const c = useConsole();

  return (
    <div style={css('display: flex; min-height: 100vh; background: #E9E3D6;')}>
      <ConsoleRail open={c.railOpen} onToggle={c.toggleRail} />

      <div style={css('flex: 1; min-width: 0;')}>
        <ConsoleTopBar
          query={c.query}
          onQueryChange={c.setQuery}
          onSubmitQuery={c.runSearch}
          placeholder={c.placeholder}
          budget={c.budget}
          clock={c.clock}
          operator={OPERATOR}
        />

        <div
          style={css(
            'padding: 22px 24px 40px; display: grid; grid-template-columns: 1.6fr 1fr; gap: 20px; align-items: start;',
          )}
        >
          <div style={css('display: flex; flex-direction: column; gap: 20px; min-width: 0;')}>
            <LiveWall
              cameras={c.cameras}
              sector={c.sector}
              expanded={c.expanded}
              onExpand={c.setExpanded}
              onClearSector={() => c.setSector('all')}
            />
            <SectorMap
              posts={c.posts}
              sector={c.sector}
              onSelect={(s) => {
                c.setSector(s);
                c.setExpanded(-1);
              }}
            />
            <FootageSearch
              query={c.query}
              onQueryChange={c.setQuery}
              placeholder={c.placeholder}
              onRun={c.runSearch}
              searching={c.searching}
              results={c.results}
            />
            <AnalyticsPanel analytics={c.analytics} traffic={c.traffic} />
          </div>

          <div style={css('display: flex; flex-direction: column; gap: 20px; min-width: 0;')}>
            <AlertQueue alerts={c.alerts} onAct={c.actOnAlert} onDismiss={c.dismissAlert} />
            <EvidenceChainPanel
              chain={c.chain}
              verifying={c.verifying}
              verified={c.verified}
              onVerify={c.verifyChain}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConsolePage;
