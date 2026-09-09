import { Router } from 'express';
import * as controller from '../controllers/analytics.controller.js';

const router = Router();

router.get('/', controller.index);
router.get('/traffic', controller.traffic);

export default router;
