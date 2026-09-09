import config from './config/index.js';
import createApp from './app.js';

const app = createApp();

app.listen(config.port, () => {
  console.log(`TRUE WATCH API listening on http://localhost:${config.port}/api`);
  console.log(`CORS origins: ${config.corsOrigins.join(', ')}`);
});
