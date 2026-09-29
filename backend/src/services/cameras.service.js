import { badRequest, notFound } from '../utils/httpError.js';
import { arr, num, obj, str, validate } from '../utils/validate.js';

const point = arr(num({ min: 0, max: 1 }), { min: 2, max: 2 });
const fenceSchema = obj({ polygon: arr(point, { min: 3, max: 64 }) });
const HHMM = /^([01]\d|2[0-3]):[0-5]\d$/;
const hoursSchema = obj({ start: str({ pattern: HHMM }), end: str({ pattern: HHMM }) });

/** Camera / post registry, read from the database. rtsp_url never leaves the repository. */
export const createCamerasService = ({ repos, audit }) => {
  const requireCamera = (id) => {
    if (!repos.cameras.get(id)) throw notFound('Camera not found');
  };

  return {
    listCameras: ({ sector } = {}) => repos.cameras.list({ sector: !sector || sector === 'all' ? undefined : sector }),
    getCamera: (id) => repos.cameras.get(id),
    listPosts: () => repos.posts.list(),
    listSectors: () => repos.cameras.sectors(),

    getSettings(id) {
      requireCamera(id);
      return repos.cameraSettings.get(id);
    },

    /** Supervisor: replace the virtual fence polygon (normalised 0..1 points). Audited. */
    updateFence(id, body, actor) {
      requireCamera(id);
      const check = validate(fenceSchema, body);
      if (!check.ok) throw badRequest('Invalid fence', { details: check.errors });
      const settings = repos.cameraSettings.update(id, { fence: body.polygon }, actor?.id);
      audit.record(actor, 'camera.fence.edit', id, { points: body.polygon.length });
      return settings;
    },

    /** Supervisor: replace the sanctioned-hours window. Audited. */
    updateHours(id, body, actor) {
      requireCamera(id);
      const check = validate(hoursSchema, body);
      if (!check.ok) throw badRequest('Invalid hours', { details: check.errors });
      const settings = repos.cameraSettings.update(id, { hours: { start: body.start, end: body.end } }, actor?.id);
      audit.record(actor, 'camera.hours.edit', id, { start: body.start, end: body.end });
      return settings;
    },
  };
};

export default createCamerasService;
