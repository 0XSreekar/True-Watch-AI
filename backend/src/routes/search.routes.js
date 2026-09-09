import { Router } from 'express';
import * as controller from '../controllers/search.controller.js';

const router = Router();

router.get('/placeholders', controller.placeholders);
router.post('/', controller.run);

export default router;
