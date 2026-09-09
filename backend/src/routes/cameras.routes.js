import { Router } from 'express';
import * as controller from '../controllers/cameras.controller.js';

const router = Router();

router.get('/', controller.index);
router.get('/sectors', controller.sectors);
router.get('/posts', controller.posts);
router.get('/:id', controller.show);

export default router;
