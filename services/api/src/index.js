import { createApp } from './app.js';
import { config } from './config.js';
import * as store from './store.js';

await store.init();

const app = createApp();
const server = app.listen(config.port, () => {
  console.log(`[api] listening on :${config.port}`);
  console.log(`[api] data dir  ${config.dataDir}`);
  console.log(`[api] model     ${config.modelUrl}`);
});

for (const signal of ['SIGTERM', 'SIGINT']) {
  process.on(signal, () => {
    console.log(`[api] ${signal} received, draining`);
    server.close(() => process.exit(0));
    // Kubernetes sends SIGKILL after the grace period; do not outlive it.
    setTimeout(() => process.exit(0), 10_000).unref();
  });
}
