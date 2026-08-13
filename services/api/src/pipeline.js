/**
 * The two long-running operations: ingest and render.
 *
 * Kept out of the HTTP layer so they can be tested directly and, later, moved
 * behind a queue without touching the routes.
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import { buildEdl, diffTranscript, identityTokens } from '@voxdocs/edl';
import { makePreviewAudio, normalizeToMaster, probe } from './ffmpeg.js';
import * as model from './modelClient.js';
import { encodeAudio, expandSegments, opDurations, renderOps, renderVideo } from './render.js';
import { newId, projectPaths, save } from './store.js';

const AUDIO_FORMATS = new Set(['wav', 'mp3', 'm4a']);

export class ValidationError extends Error {
  /** @param {string} message */
  constructor(message) {
    super(message);
    this.name = 'ValidationError';
    this.status = 400;
  }
}

/**
 * Decode, measure, transcribe. Ingest is the only place the media is inspected;
 * everything downstream works from the master and the transcript.
 *
 * @param {any} project
 * @param {string} sourcePath
 * @param {{ language?: string | null }} [options]
 */
export async function ingest(project, sourcePath, options = {}) {
  const paths = projectPaths(project.id);

  const info = await probe(sourcePath);
  if (!info.hasAudio) {
    throw new ValidationError('this file has no audio track to transcribe');
  }

  project.duration = info.duration;
  project.hasVideo = info.hasVideo;
  project.video = info.hasVideo
    ? { width: info.width, height: info.height, fps: Number(info.fps.toFixed(3)) }
    : null;
  project.status = 'transcribing';
  await save(project);

  // One canonical master drives every later cut.
  await normalizeToMaster(sourcePath, paths.master);
  try {
    await makePreviewAudio(sourcePath, paths.preview);
  } catch {
    // A missing preview only costs the editor its scrub audio; not fatal.
    project.previewFailed = true;
  }

  const result = await model.transcribe(paths.master, project.id, options.language ?? null);

  project.transcript = {
    words: result.words ?? [],
    segments: result.segments ?? [],
    language: result.language ?? 'en',
    backend: result.backend ?? 'unknown',
  };
  project.voice = result.voice ?? null;
  project.duration = result.duration || project.duration;
  project.status = 'ready';
  project.error = null;

  // The envelope is large and only the editor wants it; keep it out of the
  // project document so every other read stays cheap.
  await fs.writeFile(
    path.join(paths.dir, 'envelope.json'),
    JSON.stringify(result.envelope ?? { fps: 100, rms: [] }),
    'utf8'
  );

  await save(project);
  return project;
}

/** @param {any} project */
export async function loadEnvelope(project) {
  const paths = projectPaths(project.id);
  try {
    return JSON.parse(await fs.readFile(path.join(paths.dir, 'envelope.json'), 'utf8'));
  } catch {
    return null;
  }
}

/**
 * Turn a request body into an edit-token list.
 *
 * Two shapes are accepted: the structured list the editor maintains (exact, no
 * guessing), and plain text (aligned against the original). The editor uses the
 * first; pastes, scripts and API clients use the second.
 *
 * @param {any} project
 * @param {any} body
 */
export function resolveTokens(project, body) {
  const words = project.transcript?.words ?? [];

  if (Array.isArray(body?.tokens)) {
    const tokens = [];
    for (const token of body.tokens) {
      if (token && typeof token.ref === 'string') tokens.push({ ref: token.ref });
      else if (token && typeof token.insert === 'string') tokens.push({ insert: token.insert });
      else throw new ValidationError('each token must be {ref:string} or {insert:string}');
    }
    return tokens;
  }

  if (typeof body?.text === 'string') {
    return diffTranscript(words, body.text);
  }

  return identityTokens(words);
}

/**
 * Build the EDL without rendering, for live duration and cost feedback.
 * @param {any} project
 * @param {Array<{ ref: string } | { insert: string }>} tokens
 */
export async function planEdit(project, tokens) {
  const envelope = await loadEnvelope(project);
  return buildEdl(project.transcript?.words ?? [], tokens, {
    duration: project.duration,
    envelope,
  });
}

/**
 * Render an edit to a finished file.
 *
 * @param {any} project
 * @param {Array<{ ref: string } | { insert: string }>} tokens
 * @param {{ format?: string, video?: boolean }} [options]
 */
export async function render(project, tokens, options = {}) {
  const format = String(options.format ?? 'wav').toLowerCase();
  const wantVideo = Boolean(options.video) && Boolean(project.hasVideo);
  if (!wantVideo && !AUDIO_FORMATS.has(format)) {
    throw new ValidationError(`unsupported format "${format}" (use wav, mp3 or m4a)`);
  }

  const paths = projectPaths(project.id);
  const { segments, stats } = await planEdit(project, tokens);

  // Ask the model server for every insertion at once.
  const synthIndices = [];
  const items = [];
  segments.forEach((segment, index) => {
    if (segment.kind !== 'synth') return;
    synthIndices.push(index);
    items.push({
      text: segment.text,
      context_before: segment.contextBefore,
      context_after: segment.contextAfter,
      lead_gap: segment.leadGap,
      trail_gap: segment.trailGap,
    });
  });

  const synthesis = new Map();
  if (items.length > 0) {
    const reseed = async () => {
      await model.putVoiceProfile(
        project.id,
        project.transcript.words,
        project.duration,
        paths.master
      );
    };
    const response = await model.synthesizeBatch(project.id, items, reseed);
    (response.results ?? []).forEach((result, i) => synthesis.set(synthIndices[i], result));
  }

  await fs.mkdir(paths.renders, { recursive: true });
  const renderId = newId();
  const workDir = path.join(paths.renders, renderId);
  await fs.mkdir(workDir, { recursive: true });

  try {
    const { ops, warnings } = await expandSegments(segments, synthesis, workDir);

    const masterOut = path.join(workDir, 'render.wav');
    const rendered = await renderOps(paths.master, ops, masterOut);

    // Video needs to know how long each insertion actually turned out to be.
    const synthDurations = new Map();
    if (wantVideo) {
      const durations = await opDurations(ops);
      let cursor = 0;
      for (let i = 0; i < segments.length; i++) {
        if (segments[i].kind === 'copy') {
          cursor += 1;
          continue;
        }
        const result = synthesis.get(i);
        const count = (result?.units ?? []).filter((u) => u.type !== 'silence' || u.duration > 0.001)
          .length;
        let total = 0;
        for (let k = 0; k < count && cursor < durations.length; k++, cursor++) {
          total += durations[cursor];
        }
        synthDurations.set(i, total);
      }
    }

    let outputPath;
    let outputFormat;
    if (wantVideo) {
      outputPath = path.join(workDir, 'output.mp4');
      outputFormat = 'mp4';
      const sourcePath = await findSource(project);
      await renderVideo(sourcePath, masterOut, segments, synthDurations, outputPath);
    } else {
      outputFormat = format;
      outputPath = path.join(workDir, `output.${format}`);
      await encodeAudio(masterOut, outputPath, /** @type {any} */ (format));
    }

    // The intermediate master and any inline TTS clips have served their purpose.
    await fs.rm(masterOut, { force: true });
    for (const entry of await fs.readdir(workDir)) {
      if (entry.startsWith('tts-')) await fs.rm(path.join(workDir, entry), { force: true });
    }

    const { size } = await fs.stat(outputPath);
    const record = {
      id: renderId,
      createdAt: new Date().toISOString(),
      format: outputFormat,
      file: path.basename(outputPath),
      bytes: size,
      duration: rendered.duration,
      pieces: rendered.pieces,
      stats,
      warnings,
      synthesis: summariseSynthesis(synthesis),
    };

    project.renders = [record, ...(project.renders ?? [])].slice(0, 25);
    await save(project);
    return record;
  } catch (error) {
    await fs.rm(workDir, { recursive: true, force: true });
    throw error;
  }
}

/** @param {Map<number, any>} synthesis */
function summariseSynthesis(synthesis) {
  const covered = [];
  const generated = [];
  const missing = [];
  const backends = new Set();
  for (const result of synthesis.values()) {
    covered.push(...(result.covered ?? []));
    generated.push(...(result.generated ?? []));
    missing.push(...(result.missing ?? []));
    for (const backend of result.backends ?? []) backends.add(backend);
  }
  const total = covered.length + generated.length + missing.length;
  return {
    words: total,
    fromVoiceBank: covered.length,
    fromTts: generated.length,
    missing,
    backends: [...backends],
    coverage: total ? Number((covered.length / total).toFixed(4)) : 1,
  };
}

/** Locate the original upload, whose extension varies. */
async function findSource(project) {
  const paths = projectPaths(project.id);
  if (project.sourceFile) {
    const candidate = path.join(paths.dir, project.sourceFile);
    try {
      await fs.access(candidate);
      return candidate;
    } catch {
      /* fall through to a scan */
    }
  }
  for (const entry of await fs.readdir(paths.dir)) {
    if (entry.startsWith('source.')) return path.join(paths.dir, entry);
  }
  throw new ValidationError('the original media file is no longer available');
}
