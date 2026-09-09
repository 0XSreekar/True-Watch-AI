import { Router } from 'express';
import * as controller from '../controllers/alerts.controller.js';

const router = Router();

router.get('/', controller.index);
router.get('/budget', controller.budget);
router.post('/simulate', controller.simulate);
router.post('/:id/act', controller.act);
router.post('/:id/dismiss', controller.dismiss);

export default router;
