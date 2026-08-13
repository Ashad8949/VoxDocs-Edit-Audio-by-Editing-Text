/**
 * Project persistence.
 *
 * Projects live as a directory per project on disk: the original upload, the
 * canonical render master, a streamable preview, and one JSON document holding
 * the transcript and edit state. A database would buy little here — the media
 * files dominate, they are already on a filesystem, and everything the API
 * needs is a point read by id.
 */

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { config } from './config.js';

/** Project ids appear in filesystem paths, so keep them unambiguous. */
const ID_PATTERN = /^[A-Za-z0-9_-]{6,64}$/;

export class NotFoundError extends Error {
  constructor(message = 'not found') {
    super(message);
    this.name = 'NotFoundError';
    this.status = 404;
  }
}

export function newId() {
  return crypto.randomBytes(12).toString('base64url');
}

/**
 * Resolve a project directory, refusing anything that could escape the data
 * directory. Ids come from URLs, so this is a security boundary, not a nicety.
 * @param {string} id
 */
export function projectDir(id) {
  if (typeof id !== 'string' || !ID_PATTERN.test(id)) {
    throw new NotFoundError('invalid project id');
  }
  const dir = path.join(config.dataDir, 'projects', id);
  const root = path.join(config.dataDir, 'projects');
  if (path.relative(root, dir).startsWith('..')) throw new NotFoundError('invalid project id');
  return dir;
}

/** @param {string} id */
export function projectPaths(id) {
  const dir = projectDir(id);
  return {
    dir,
    meta: path.join(dir, 'project.json'),
    master: path.join(dir, 'master.wav'),
    preview: path.join(dir, 'preview.m4a'),
    renders: path.join(dir, 'renders'),
    sourceFor: (extension) => path.join(dir, `source${extension}`),
  };
}

export async function init() {
  await fs.mkdir(path.join(config.dataDir, 'projects'), { recursive: true });
  await fs.mkdir(path.join(config.dataDir, 'tmp'), { recursive: true });
}

/**
 * Write the project document atomically, so a crash mid-write cannot leave a
 * half-serialised transcript behind.
 * @param {any} project
 */
export async function save(project) {
  const paths = projectPaths(project.id);
  await fs.mkdir(paths.dir, { recursive: true });
  const temporary = `${paths.meta}.${process.pid}.tmp`;
  project.updatedAt = new Date().toISOString();
  await fs.writeFile(temporary, JSON.stringify(project, null, 2), 'utf8');
  await fs.rename(temporary, paths.meta);
  return project;
}

/** @param {string} id */
export async function load(id) {
  const paths = projectPaths(id);
  try {
    return JSON.parse(await fs.readFile(paths.meta, 'utf8'));
  } catch (error) {
    if (/** @type {any} */ (error).code === 'ENOENT') throw new NotFoundError('project not found');
    throw error;
  }
}

export async function list() {
  const root = path.join(config.dataDir, 'projects');
  let entries;
  try {
    entries = await fs.readdir(root, { withFileTypes: true });
  } catch {
    return [];
  }

  const projects = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    try {
      const project = await load(entry.name);
      projects.push(summarize(project));
    } catch {
      // A directory without a readable document is a partial upload; skip it.
    }
  }
  projects.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
  return projects;
}

/** @param {string} id */
export async function remove(id) {
  const paths = projectPaths(id);
  await fs.rm(paths.dir, { recursive: true, force: true });
}

/** The listing view: everything except the bulky per-word arrays. */
export function summarize(project) {
  return {
    id: project.id,
    name: project.name,
    createdAt: project.createdAt,
    updatedAt: project.updatedAt,
    status: project.status,
    duration: project.duration,
    hasVideo: project.hasVideo,
    wordCount: project.transcript?.words?.length ?? 0,
    language: project.transcript?.language ?? null,
    error: project.error ?? null,
  };
}
