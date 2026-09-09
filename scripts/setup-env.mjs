/** Copies each .env.example to .env when one is not already present. */
import { copyFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

for (const pkg of ['frontend', 'backend']) {
  const example = join(root, pkg, '.env.example');
  const target = join(root, pkg, '.env');
  if (existsSync(example) && !existsSync(target)) {
    copyFileSync(example, target);
    console.log(`created ${pkg}/.env`);
  } else {
    console.log(`${pkg}/.env already present — left untouched`);
  }
}
