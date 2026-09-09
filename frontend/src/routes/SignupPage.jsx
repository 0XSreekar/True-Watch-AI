import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import AuthAltLink from '../components/auth/AuthAltLink';
import AuthCard from '../components/auth/AuthCard';
import AuthLayout from '../components/auth/AuthLayout';
import SignupForm from '../components/auth/SignupForm';
import StepDots from '../components/auth/StepDots';
import SubmitRow from '../components/auth/SubmitRow';
import { FALLBACK_ROLES, FALLBACK_UNITS } from '../data/fallbacks';
import { useAppShell } from '../context/AppShellContext';
import useAuthForm from '../hooks/useAuthForm';
import { authApi, withFallback } from '../services';

const TITLES = ['Who is signing on', 'Set your credentials', 'Role and undertaking'];

export const SignupPage = () => {
  const navigate = useNavigate();
  const { enterConsole } = useAppShell();
  const auth = useAuthForm({ mode: 'signup', onAuthenticated: () => enterConsole() });

  const [units, setUnits] = useState(FALLBACK_UNITS);
  const [roles, setRoles] = useState(FALLBACK_ROLES);

  useEffect(() => {
    let alive = true;
    Promise.all([
      withFallback(authApi.fetchUnits(), FALLBACK_UNITS, 'units'),
      withFallback(authApi.fetchRoles(), FALLBACK_ROLES, 'roles'),
    ]).then(([u, r]) => {
      if (!alive) return;
      setUnits(u);
      setRoles(r);
    });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <AuthLayout>
      <AuthCard
        shake={auth.shake}
        steps={<StepDots step={auth.step} />}
        title={TITLES[auth.step]}
        subtitle="Accounts are provisioned per post and reviewed by your sector supervisor."
        footer={
          <AuthAltLink
            prompt="Already provisioned?"
            linkLabel="Sign in"
            onClick={() => navigate('/login')}
          />
        }
      >
        <SignupForm {...auth} units={units} roles={roles} />
        <SubmitRow
          label={
            auth.loading ? 'VERIFYING TERMINAL…' : auth.step < 2 ? 'CONTINUE' : 'CREATE ACCOUNT'
          }
          loading={auth.loading}
          onSubmit={auth.submit}
          showBack={auth.step > 0}
          onBack={auth.back}
        />
      </AuthCard>
    </AuthLayout>
  );
};

export default SignupPage;
