/**
 * Records privileged actions in the hash-chained audit_log (threat 4). `actor` is the
 * authenticated user, or null when the console runs unauthenticated (AUTH_REQUIRED=false),
 * in which case the row still exists and says so.
 */
export const createAuditService = ({ repos }) => ({
  record(actor, action, subject, detail = null) {
    return repos.audit.append({
      actorId: actor?.id ?? null,
      action,
      subject,
      detail: actor ? detail : { ...(detail || {}), unauthenticated: true },
    });
  },
  recent: (limit = 100) => repos.audit.recent(limit),
  verify: () => repos.audit.verify(),
});

export default createAuditService;
