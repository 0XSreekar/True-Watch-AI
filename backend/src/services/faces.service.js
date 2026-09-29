import { badRequest, notFound } from '../utils/httpError.js';
import { obj, str, validate } from '../utils/validate.js';

const unsealSchema = obj({ reason: str({ min: 5, max: 500 }) });

/**
 * Face unsealing (slide 4: "unsealing is access-controlled"). Face crops never leave the edge, so
 * this records the supervisor's authorisation, with the stated reason, in the audit log and
 * returns the grant. Releasing the crop itself is the edge's decision on presenting that grant.
 */
export const createFacesService = ({ repos, audit }) => ({
  unseal(alertId, body, actor) {
    const check = validate(unsealSchema, body);
    if (!check.ok) throw badRequest('A reason of at least 5 characters is required', { details: check.errors });
    const alert = repos.alerts.findById(alertId);
    if (!alert || alert.stage === 'gone') throw notFound('Alert not found');

    const entry = audit.record(actor, 'face.unmask', alertId, {
      reason: body.reason,
      camera: alert.camera.id,
    });
    return { alert_id: alertId, unsealed: true, audit_id: entry.id, granted_at: entry.at };
  },
});

export default createFacesService;
