import fs from 'node:fs';
import path from 'node:path';

import Database from 'better-sqlite3';

import { MIGRATIONS } from './migrations.js';

/** Applies every migration not yet recorded. Safe to call on every boot. */
export const migrate = (db) => {
  db.exec(`CREATE TABLE IF NOT EXISTS schema_migrations (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
  )`);
  const applied = new Set(db.prepare('SELECT id FROM schema_migrations').all().map((r) => r.id));
  const record = db.prepare('INSERT INTO schema_migrations (id, name, applied_at) VALUES (?, ?, ?)');

  for (const migration of MIGRATIONS) {
    if (applied.has(migration.id)) continue;
    db.transaction(() => {
      db.exec(migration.sql);
      record.run(migration.id, migration.name, new Date().toISOString());
    })();
  }
};

/**
 * Opens (creating the directory if needed) and migrates the SQLite file.
 * WAL and foreign keys are on; ':memory:' is accepted for throwaway instances.
 */
export const openDatabase = (file) => {
  if (file !== ':memory:') fs.mkdirSync(path.dirname(file), { recursive: true });
  const db = new Database(file);
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');
  db.pragma('busy_timeout = 5000');
  migrate(db);
  return db;
};

export default openDatabase;
