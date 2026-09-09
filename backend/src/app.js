import cors from 'cors';
import express from 'express';
import morgan from 'morgan';

import config from './config/index.js';
import errorHandler from './middleware/errorHandler.js';
import notFound from './middleware/notFound.js';
import routes from './routes/index.js';

export const createApp = () => {
  const app = express();

  app.use(cors({ origin: config.corsOrigins, credentials: true }));
  app.use(express.json());
  if (config.env !== 'test') app.use(morgan('dev'));

  app.use('/api', routes);

  app.use(notFound);
  app.use(errorHandler);

  return app;
};

export default createApp;
