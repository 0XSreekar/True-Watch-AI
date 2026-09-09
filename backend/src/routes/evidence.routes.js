import { Router } from 'express';
import * as controller from '../controllers/evidence.controller.js';

const router = Router();

router.get('/chain', controller.chain);
router.post('/verify', controller.verify);

export default router;
