import { useNavigate } from 'react-router-dom';

import AuthAltLink from '../components/auth/AuthAltLink';
import AuthCard from '../components/auth/AuthCard';
import AuthLayout from '../components/auth/AuthLayout';
import LoginForm from '../components/auth/LoginForm';
import SubmitRow from '../components/auth/SubmitRow';
import { useAppShell } from '../context/AppShellContext';
import useAuthForm from '../hooks/useAuthForm';

export const LoginPage = () => {
  const navigate = useNavigate();
  const { enterConsole } = useAppShell();

  const auth = useAuthForm({ mode: 'login', onAuthenticated: () => enterConsole() });

  return (
    <AuthLayout>
      <AuthCard
        shake={auth.shake}
        title="Terminal sign-in"
        subtitle="Authorised SSB personnel only. This terminal is logged."
        footer={
          <AuthAltLink
            prompt="No account for this post?"
            linkLabel="Request one"
            onClick={() => navigate('/signup')}
          />
        }
      >
        <LoginForm {...auth} />
        <SubmitRow
          label={auth.loading ? 'VERIFYING TERMINAL…' : 'SIGN IN TO CONSOLE'}
          loading={auth.loading}
          onSubmit={auth.submit}
          showBack={false}
          onBack={auth.back}
        />
      </AuthCard>
    </AuthLayout>
  );
};

export default LoginPage;
