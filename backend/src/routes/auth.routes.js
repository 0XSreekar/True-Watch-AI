import { Router } from 'express';
import * as controller from '../controllers/auth.controller.js';

const router = Router();

router.post('/login', controller.login);
router.post('/register', controller.register);
router.get('/units', controller.units);
router.get('/roles', controller.roles);

export default router;
