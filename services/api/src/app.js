/**
 * HTTP surface.
 *
 * Ingest runs in the background because transcribing an hour of audio takes
 * minutes and no browser upload should be held open for that. The client polls
 * the project until its status leaves "transcribing".
 */

import express from 'express';
import multer from 'multer';
import fs from 'node:fs/promises';
import path from 'node:path';
import { AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, config } from './config.js';
import * as ffmpeg from './ffmpeg.js';
import * as model from './modelClient.js';
import { ValidationError, ingest, loadEnvelope, planEdit, render, resolveTokens } from './pipeline.js';
import * as store from './store.js';

const upload = multer({
  dest: path.join(config.dataDir, 'tmp'),
  limits: { fileSize: config.maxUploadBytes, files: 1 },
});

/** @param {(req: any, res: any, next: any) => Promise<any>} handler */
const wrap = (handler) => (req, res, next) => handler(req, res, next).catch(next);

export function createApp() {
  const app = express();
  app.use(express.json({ limit: '32mb' }));
  app.disable('x-powered-by');

  app.use((req, res, next) => {
    res.setHeader('access-control-allow-origin', config.corsOrigin);
    res.setHeader('access-control-allow-headers', 'content-type');
    res.setHeader('access-control-allow-methods', 'GET,POST,DELETE,OPTIONS');
    if (req.method === 'OPTIONS') return res.sendStatus(204);
    next();
  });

  // ------------------------------------------------------------- health

  app.get('/api/health', wrap(async (_req, res) => {
    let modelHealth = null;
    let modelError = null;
    try {
      modelHealth = await model.health();
    } catch (error) {
      modelError = error.message;
    }
    res.json({
      status: modelHealth ? 'ok' : 'degraded',
      model: modelHealth,
      modelError,
      renderSampleRate: config.renderSampleRate,
    });
  }));

  app.get('/api/ready', (_req, res) => res.json({ ready: true }));

  // ----------------------------------------------------------- projects

  app.get('/api/projects', wrap(async (_req, res) => {
    res.json({ projects: await store.list() });
  }));

  app.post('/api/projects', upload.single('file'), wrap(async (req, res) => {
    if (!req.file) throw new ValidationError('expected a "file" upload');

    const extension = path.extname(req.file.originalname || '').toLowerCase();
    const known = AUDIO_EXTENSIONS.has(extension) || VIDEO_EXTENSIONS.has(extension);
    if (extension && !known) {
      await fs.rm(req.file.path, { force: true });
      throw new ValidationError(`unsupported file type "${extension}"`);
    }

    const id = store.newId();
    const paths = store.projectPaths(id);
    await fs.mkdir(paths.dir, { recursive: true });

    const sourceName = `source${extension || '.bin'}`;
    await fs.rename(req.file.path, path.join(paths.dir, sourceName));

    const project = {
      id,
      name: req.body?.name || req.file.originalname || 'Untitled',
      createdAt: new Date().toISOString(),
      status: 'queued',
      sourceFile: sourceName,
      duration: 0,
      hasVideo: false,
      transcript: { words: [], segments: [], language: null, backend: null },
      renders: [],
    };
    await store.save(project);

    // Fire and forget: the client polls for completion.
    const language = req.body?.language || null;
    ingest(project, path.join(paths.dir, sourceName), { language }).catch(async (error) => {
      project.status = 'failed';
      project.error = error.message || String(error);
      await store.save(project).catch(() => {});
    });

    res.status(202).json({ project: store.summarize(project) });
  }));

  app.get('/api/projects/:id', wrap(async (req, res) => {
    const project = await store.load(req.params.id);
    res.json({ project });
  }));

  app.delete('/api/projects/:id', wrap(async (req, res) => {
    await store.load(req.params.id); // 404 rather than silently succeeding
    await store.remove(req.params.id);
    res.json({ deleted: true });
  }));

  app.get('/api/projects/:id/envelope', wrap(async (req, res) => {
    const project = await store.load(req.params.id);
    const envelope = await loadEnvelope(project);
    if (!envelope) throw new store.NotFoundError('no envelope for this project');

    // The editor draws a few thousand pixels at most; downsample by peak so
    // transients survive rather than being averaged away.
    const points = Math.max(0, Math.min(20000, Number(req.query.points) || 0));
    if (!points || points >= envelope.rms.length) return res.json(envelope);

    const bucket = envelope.rms.length / points;
    const peaks = new Array(points);
    for (let i = 0; i < points; i++) {
      const start = Math.floor(i * bucket);
      const end = Math.min(envelope.rms.length, Math.floor((i + 1) * bucket) + 1);
      let max = 0;
      for (let k = start; k < end; k++) if (envelope.rms[k] > max) max = envelope.rms[k];
      peaks[i] = Number(max.toFixed(5));
    }
    res.json({ fps: envelope.fps * (points / envelope.rms.length), rms: peaks, downsampled: true });
  }));

  // -------------------------------------------------------------- media

  app.get('/api/projects/:id/media', wrap(async (req, res) => {
    const project = await store.load(req.params.id);
    const paths = store.projectPaths(project.id);

    // The preview is a small AAC copy; the original is served when the client
    // needs the video track or the preview never got made.
    const wantOriginal = req.query.original === '1' || project.hasVideo;
    let filePath = paths.preview;
    if (wantOriginal || project.previewFailed) {
      filePath = path.join(paths.dir, project.sourceFile ?? 'source.bin');
    }
    try {
      await fs.access(filePath);
    } catch {
      filePath = path.join(paths.dir, project.sourceFile ?? 'source.bin');
    }
    await sendFile(req, res, filePath);
  }));

  // ------------------------------------------------------------ editing

  app.post('/api/projects/:id/plan', wrap(async (req, res) => {
    const project = await store.load(req.params.id);
    requireReady(project);
    const tokens = resolveTokens(project, req.body ?? {});
    const { segments, stats } = await planEdit(project, tokens);
    res.json({
      stats,
      segments: req.body?.includeSegments ? segments : undefined,
      tokens: req.body?.includeTokens ? tokens : undefined,
    });
  }));

  app.post('/api/projects/:id/render', wrap(async (req, res) => {
    const project = await store.load(req.params.id);
    requireReady(project);
    const tokens = resolveTokens(project, req.body ?? {});
    const record = await render(project, tokens, {
      format: req.body?.format ?? 'wav',
      video: req.body?.video ?? false,
    });
    res.json({ render: record, downloadUrl: `/api/projects/${project.id}/renders/${record.id}` });
  }));

  app.get('/api/projects/:id/renders/:renderId', wrap(async (req, res) => {
    const project = await store.load(req.params.id);
    const record = (project.renders ?? []).find((r) => r.id === req.params.renderId);
    if (!record) throw new store.NotFoundError('render not found');

    const paths = store.projectPaths(project.id);
    const filePath = path.join(paths.renders, record.id, record.file);
    const base = sanitizeName(project.name);
    res.setHeader('content-disposition',
      `attachment; filename="${base}-edited.${record.format}"`);
    await sendFile(req, res, filePath);
  }));

  // ------------------------------------------------------------- errors

  app.use((_req, res) => res.status(404).json({ error: 'not_found' }));

  app.use((error, _req, res, _next) => {
    const status = error.status ?? (error instanceof ffmpeg.FfmpegError ? 500 : 500);
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({ error: 'file_too_large', limitBytes: config.maxUploadBytes });
    }
    if (status >= 500) console.error('[api]', error);
    res.status(status).json({
      error: error.code ?? error.name ?? 'error',
      message: error.message ?? 'unexpected error',
    });
  });

  return app;
}

/** @param {any} project */
function requireReady(project) {
  if (project.status === 'failed') {
    throw new ValidationError(`this project failed to import: ${project.error ?? 'unknown error'}`);
  }
  if (project.status !== 'ready') {
    const error = new ValidationError('the transcript is still being prepared');
    error.status = 409;
    throw error;
  }
}

/** @param {string} name */
function sanitizeName(name) {
  return (String(name || 'voxdocs').replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9._-]+/g, '-') || 'voxdocs')
    .slice(0, 60);
}

const MIME_TYPES = {
  '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.m4a': 'audio/mp4', '.aac': 'audio/aac',
  '.flac': 'audio/flac', '.ogg': 'audio/ogg', '.opus': 'audio/opus',
  '.mp4': 'video/mp4', '.mov': 'video/quicktime', '.webm': 'video/webm', '.mkv': 'video/x-matroska',
};

/**
 * Serve a file with range support, which the browser's media element requires
 * in order to seek.
 */
async function sendFile(req, res, filePath) {
  const { createReadStream } = await import('node:fs');
  let stat;
  try {
    stat = await fs.stat(filePath);
  } catch {
    throw new store.NotFoundError('media not found');
  }

  const type = MIME_TYPES[path.extname(filePath).toLowerCase()] ?? 'application/octet-stream';
  res.setHeader('content-type', type);
  res.setHeader('accept-ranges', 'bytes');

  const range = req.headers.range;
  if (range) {
    const match = /bytes=(\d*)-(\d*)/.exec(range);
    if (match) {
      const start = match[1] ? Number(match[1]) : 0;
      const end = match[2] ? Number(match[2]) : stat.size - 1;
      if (start >= stat.size || start > end) {
        res.setHeader('content-range', `bytes */${stat.size}`);
        return res.status(416).end();
      }
      const last = Math.min(end, stat.size - 1);
      res.status(206);
      res.setHeader('content-range', `bytes ${start}-${last}/${stat.size}`);
      res.setHeader('content-length', last - start + 1);
      return createReadStream(filePath, { start, end: last }).pipe(res);
    }
  }

  res.setHeader('content-length', stat.size);
  return createReadStream(filePath).pipe(res);
}
