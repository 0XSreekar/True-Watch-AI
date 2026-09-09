import { useCallback, useState } from 'react';

import { authApi, withFallback } from '../services';

const EMPTY = {
  id: '',
  pw: '',
  name: '',
  unit: '',
  email: '',
  np: '',
  cp: '',
  role: '',
  terms: false,
};

/** 0–5 strength score driving the signup password meter. */
export const passwordScore = (p) => {
  let s = 0;
  if (p.length >= 8) s += 1;
  if (p.length >= 12) s += 1;
  if (/[A-Z]/.test(p) && /[a-z]/.test(p)) s += 1;
  if (/\d/.test(p)) s += 1;
  if (/[^A-Za-z0-9]/.test(p)) s += 1;
  return Math.min(5, s);
};

/**
 * Form state, per-step validation and submission for both auth screens.
 * Validation stays client-side; the API re-checks and issues the session.
 */
export const useAuthForm = ({ mode, onAuthenticated }) => {
  const [form, setForm] = useState(EMPTY);
  const [errors, setErrors] = useState({});
  const [ok, setOk] = useState({});
  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [shake, setShake] = useState(0);
  const [showPw, setShowPw] = useState(false);
  const [remember, setRemember] = useState(true);

  const setField = useCallback((key, value) => {
    setForm((f) => ({ ...f, [key]: value }));
  }, []);

  const validate = useCallback(() => {
    const e = {};
    const good = {};

    if (mode === 'login') {
      if (!form.id) e.id = 'Official ID required';
      else good.id = 1;
      if (form.pw.length < 4) e.pw = 'Password too short';
      else good.pw = 1;
    } else if (step === 0) {
      if (form.name.trim().length < 3) e.name = 'Enter your full name';
      else good.name = 1;
      if (!/^[A-Za-z]{2,4}\/?\d{3,7}$/.test(form.id)) e.id = 'Format: SSB/104382';
      else good.id = 1;
      if (!form.unit) e.unit = 'Select your posting';
      else good.unit = 1;
    } else if (step === 1) {
      if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(form.email)) e.email = 'Enter a valid official email';
      else good.email = 1;
      if (passwordScore(form.np) < 3) e.np = 'Use 10+ characters with a number and a symbol';
      else good.np = 1;
      if (form.cp !== form.np || !form.cp) e.cp = 'Passwords do not match';
      else good.cp = 1;
    } else {
      if (!form.role) e.role = 'Choose a role';
      else good.role = 1;
      if (!form.terms) e.terms = 'You must accept the terms';
      else good.terms = 1;
    }

    setErrors(e);
    setOk(good);
    if (Object.keys(e).length) {
      setShake((n) => n + 1);
      return false;
    }
    return true;
  }, [form, mode, step]);

  const submit = useCallback(async () => {
    if (!validate()) return;

    if (mode === 'signup' && step < 2) {
      setStep((s) => s + 1);
      setErrors({});
      setOk({});
      return;
    }

    setLoading(true);
    const payload =
      mode === 'login'
        ? { officialId: form.id, password: form.pw, remember }
        : {
            name: form.name,
            officialId: form.id,
            unit: form.unit,
            email: form.email,
            password: form.np,
            role: form.role,
          };

    // The console must still open if the API is down — this is a demo build.
    const session = await withFallback(
      mode === 'login' ? authApi.login(payload) : authApi.register(payload),
      null,
      `${mode} request`,
    );

    setLoading(false);
    onAuthenticated?.(session, form);
  }, [form, mode, onAuthenticated, remember, step, validate]);

  const back = useCallback(() => {
    setStep((s) => Math.max(0, s - 1));
    setErrors({});
    setOk({});
  }, []);

  const reset = useCallback(() => {
    setErrors({});
    setOk({});
    setStep(0);
  }, []);

  return {
    form,
    setField,
    errors,
    ok,
    step,
    loading,
    shake,
    showPw,
    toggleShowPw: () => setShowPw((v) => !v),
    remember,
    toggleRemember: () => setRemember((v) => !v),
    validate,
    submit,
    back,
    reset,
    pwScore: passwordScore(form.np),
  };
};

export default useAuthForm;
