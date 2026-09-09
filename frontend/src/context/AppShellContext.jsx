import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const AppShellContext = createContext(null);

/**
 * Chrome that sits above every route: the DIM/LIGHT screen wash and the radar
 * wipe played while entering the console.
 */
export const AppShellProvider = ({ children }) => {
  const navigate = useNavigate();
  const [dim, setDim] = useState(false);
  const [transitioning, setTransitioning] = useState(false);
  const timerRef = useRef(null);

  const toggleDim = useCallback(() => setDim((d) => !d), []);

  const enterConsole = useCallback(
    (event) => {
      event?.preventDefault?.();
      setTransitioning(true);
      clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        setTransitioning(false);
        navigate('/console');
        window.scrollTo(0, 0);
      }, 950);
    },
    [navigate],
  );

  const value = useMemo(
    () => ({ dim, toggleDim, dimLabel: dim ? 'DIM' : 'LIGHT', transitioning, enterConsole }),
    [dim, toggleDim, transitioning, enterConsole],
  );

  return <AppShellContext.Provider value={value}>{children}</AppShellContext.Provider>;
};

export const useAppShell = () => {
  const ctx = useContext(AppShellContext);
  if (!ctx) throw new Error('useAppShell must be used inside <AppShellProvider>');
  return ctx;
};

export default AppShellContext;
