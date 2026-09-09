import { Navigate, Route, Routes } from 'react-router-dom';

import ScreenOverlays from './components/common/ScreenOverlays';
import { AppShellProvider } from './context/AppShellContext';
import usePointerTracking from './hooks/usePointer';
import ConsolePage from './routes/ConsolePage';
import LandingPage from './routes/LandingPage';
import LoginPage from './routes/LoginPage';
import SignupPage from './routes/SignupPage';
import { css } from './utils/css';

const App = () => {
  // One global listener feeds the parallax in every canvas scene.
  usePointerTracking();

  return (
    <AppShellProvider>
      <div style={css('position: relative; min-height: 100vh; background: #E9E3D6; overflow-x: hidden;')}>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />
          <Route path="/console" element={<ConsolePage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        <ScreenOverlays />
      </div>
    </AppShellProvider>
  );
};

export default App;
