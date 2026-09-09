import { OPERATOR, ROLES, UNITS } from '../data/mockData.js';
import { hex } from '../utils/random.js';

/**
 * Placeholder authentication.
 * Real system: verify against the SSB directory, issue a signed JWT, log the terminal.
 * NOTE: nothing here is secure — it accepts any well-formed credentials by design.
 */
const session = (payload) => ({
  token: `demo.${hex(24)}`,
  expiresIn: 43200,
  user: {
    id: payload.officialId,
    name: payload.name || OPERATOR.name,
    role: payload.role || OPERATOR.defaultRole,
    unit: payload.unit || OPERATOR.unit,
    initials: OPERATOR.initials,
  },
});

export const login = ({ officialId, password }) => {
  if (!officialId) return { ok: false, error: 'Official ID required' };
  if (!password || password.length < 4) return { ok: false, error: 'Password too short' };
  return { ok: true, ...session({ officialId }) };
};

export const register = (payload) => {
  const required = ['name', 'officialId', 'unit', 'email', 'password', 'role'];
  const missing = required.filter((k) => !payload?.[k]);
  if (missing.length) return { ok: false, error: `Missing: ${missing.join(', ')}` };
  return { ok: true, pendingApproval: true, ...session(payload) };
};

export const listUnits = () => UNITS;
export const listRoles = () => ROLES;
