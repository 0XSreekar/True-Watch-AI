import { useCallback, useRef } from 'react';

import CapabilitiesSection from '../components/landing/CapabilitiesSection';
import ChannelsSection from '../components/landing/ChannelsSection';
import EvidenceSection from '../components/landing/EvidenceSection';
import Hero from '../components/landing/Hero';
import PipelineSection from '../components/landing/PipelineSection';
import ProblemsSection from '../components/landing/ProblemsSection';
import SiteFooter from '../components/landing/SiteFooter';
import StatsSection from '../components/landing/StatsSection';
import TopNav from '../components/landing/TopNav';
import { css } from '../utils/css';

export const LandingPage = () => {
  const pipelineRef = useRef(null);

  const scrollToPipeline = useCallback(() => {
    if (pipelineRef.current) {
      window.scrollTo({ top: pipelineRef.current.offsetTop - 64, behavior: 'smooth' });
    }
  }, []);

  return (
    <div style={css('position: relative;')}>
      <TopNav />
      <Hero onSeeHowItWorks={scrollToPipeline} />
      <ProblemsSection />
      <StatsSection />
      <PipelineSection ref={pipelineRef} />
      <CapabilitiesSection />
      <ChannelsSection />
      <EvidenceSection />
      <SiteFooter />
    </div>
  );
};

export default LandingPage;
