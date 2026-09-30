/**
 * Decides which existing form field shows a server-side auth error, and (for signup) which step
 * that field lives on. The server answers with one message; it is shown verbatim through the
 * field's own error line, so no new UI is involved.
 *
 * @returns {{ field: string, step: number }}
 */
export const routeAuthError = (mode, message, currentStep = 0) => {
  const text = String(message || '');
  const rules =
    mode === 'login'
      ? [
          [/official\s?id/i, 'id', 0],
          [/./, 'pw', 0],
        ]
      : [
          [/name/i, 'name', 0],
          [/official\s?id/i, 'id', 0],
          [/unit/i, 'unit', 0],
          [/email/i, 'email', 1],
          [/password/i, 'np', 1],
          [/role/i, 'role', 2],
          [/missing/i, 'name', 0],
        ];
  const hit = rules.find(([re]) => re.test(text));
  if (hit) return { field: hit[1], step: hit[2] };
  // Unrecognised message: show it on the last field of the step the person is looking at.
  const fallback = ['id', 'np', 'role'][currentStep] || 'id';
  return { field: mode === 'login' ? 'pw' : fallback, step: currentStep };
};

export default routeAuthError;
