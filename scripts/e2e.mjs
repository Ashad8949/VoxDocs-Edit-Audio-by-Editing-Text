#!/usr/bin/env node
/**
 * End-to-end smoke test against a running stack.
 *
 * Unlike the unit suites, this one asserts on the *audio*: it edits a real
 * recording through the real HTTP API and then transcribes the rendered result
 * to check that the words that came out are the words that were asked for. A
 * pipeline can be green on every unit test and still emit silence.
 *
 * Usage:
 *   node scripts/e2e.mjs [--api http://localhost:3000] [--file sample.wav]
 *
 * Requires the API and model servers to be running. See README.md.
 */

import { execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 && args[index + 1] ? args[index + 1] : fallback;
};

const API = flag('api', process.env.VOXDOCS_API_URL ?? 'http://localhost:3000').replace(/\/+$/, '');
let sampleFile = flag('file', '');

let failures = 0;
const check = (label, ok, detail = '') => {
  console.log(`${ok ? '  ok  ' : ' FAIL '} ${label}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures += 1;
};

const api = async (pathname, init) => {
  const response = await fetch(`${API}${pathname}`, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(`${pathname} → ${response.status} ${body.message ?? body.error ?? ''}`);
  }
  return body;
};

const json = (body) => ({
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
});

const workDir = await fs.mkdtemp(path.join(os.tmpdir(), 'voxdocs-e2e-'));

/** Words spoken in the sample, used to build the edits below. */
const SENTENCE =
  'Four score and seven years ago our fathers brought forth on this continent a new nation, '
  + 'conceived in liberty and dedicated to the proposition that all men are created equal.';

try {
  console.log(`VoxDocs end-to-end check against ${API}\n`);

  // ---------------------------------------------------------------- health
  const health = await api('/api/health');
  check('api is reachable', true);
  check('model server is reachable', health.status === 'ok', health.modelError ?? '');
  if (health.status !== 'ok') throw new Error('the model server is not available');

  // ------------------------------------------------------------ the sample
  if (!sampleFile) {
    sampleFile = path.join(workDir, 'sample.wav');
    try {
      await execFileAsync('espeak-ng', ['-w', sampleFile, SENTENCE]);
    } catch {
      throw new Error('no --file given and espeak-ng is not installed to generate one');
    }
  }

  // --------------------------------------------------------------- ingest
  const form = new FormData();
  form.append('file', new Blob([await fs.readFile(sampleFile)]), path.basename(sampleFile));
  form.append('name', 'e2e');

  const created = await api('/api/projects', { method: 'POST', body: form });
  const id = created.project.id;
  check('upload accepted', Boolean(id), id);

  let project;
  const deadline = Date.now() + 10 * 60 * 1000;
  for (;;) {
    ({ project } = await api(`/api/projects/${id}`));
    if (project.status === 'ready' || project.status === 'failed') break;
    if (Date.now() > deadline) throw new Error('timed out waiting for transcription');
    await new Promise((r) => setTimeout(r, 1500));
  }
  check('transcription finished', project.status === 'ready', project.error ?? '');
  if (project.status !== 'ready') throw new Error(project.error);

  const words = project.transcript.words;
  check('transcript has word-level timings', words.length > 5, `${words.length} words`);
  check(
    'word timings are monotonic and inside the media',
    words.every((w, i) => w.start < w.end && (i === 0 || w.start >= words[i - 1].start))
      && words[words.length - 1].end <= project.duration + 0.05
  );

  const spoken = words.map((w) => w.text.toLowerCase().replace(/[^a-z0-9]/g, '')).join(' ');
  check('transcript resembles the sample', spoken.includes('liberty'), spoken.slice(0, 60));

  // ------------------------------------------------------------- deletion
  // Drop everything up to "years", the way the demo removes the 1863 opener.
  const yearsIndex = words.findIndex((w) => /years/i.test(w.text));
  check('found the word to cut at', yearsIndex > 0, `index ${yearsIndex}`);

  const kept = words.slice(yearsIndex).map((w) => ({ ref: w.id }));
  const plan = await api(`/api/projects/${id}/plan`, json({ tokens: kept }));
  check('plan reports the deletion', plan.stats.deletedWords === yearsIndex,
    `${plan.stats.deletedWords} deleted`);
  check('plan predicts a shorter result',
    plan.stats.estimatedDuration < plan.stats.sourceDuration,
    `${plan.stats.estimatedDuration.toFixed(2)}s of ${plan.stats.sourceDuration.toFixed(2)}s`);

  const cut = await api(`/api/projects/${id}/render`, json({ tokens: kept, format: 'wav' }));
  check('render succeeded', Boolean(cut.render.id));
  check('rendered file is shorter than the source',
    cut.render.duration < project.duration - 0.3,
    `${cut.render.duration.toFixed(2)}s vs ${project.duration.toFixed(2)}s`);
  check('no synthesis was needed for a pure deletion',
    cut.render.synthesis.words === 0 && cut.render.warnings.length === 0);

  const cutPath = path.join(workDir, 'cut.wav');
  await downloadTo(`${API}${cut.downloadUrl}`, cutPath);
  const cutHeard = await transcribe(cutPath);
  check('the cut words are gone from the audio',
    !/four score/i.test(cutHeard),
    cutHeard.slice(0, 70));
  check('the kept words survive in the audio', /liberty/i.test(cutHeard));

  // ------------------------------------------------------------ insertion
  // The demo's "246 years ago": one word that was never spoken.
  const withInsert = [{ insert: '246' }, ...kept];
  const inserted = await api(`/api/projects/${id}/render`, json({ tokens: withInsert, format: 'wav' }));
  check('insertion rendered', Boolean(inserted.render.id));
  check('the inserted word was synthesised, not dropped',
    inserted.render.synthesis.words === 1 && inserted.render.synthesis.missing.length === 0,
    JSON.stringify(inserted.render.synthesis));

  const insertPath = path.join(workDir, 'insert.wav');
  await downloadTo(`${API}${inserted.downloadUrl}`, insertPath);
  const insertHeard = await transcribe(insertPath);
  check('the inserted word is audible in the result',
    /246|two hundred|two forty/i.test(insertHeard),
    insertHeard.slice(0, 70));

  // ------------------------------------------- insertion from the own voice
  // Repeating words the speaker already said must come from their own audio.
  const tail = words.slice(-5);
  const echoed = [
    ...words.map((w) => ({ ref: w.id })),
    { insert: tail.map((w) => w.text.replace(/[^A-Za-z0-9']/g, '')).join(' ') },
  ];
  const echo = await api(`/api/projects/${id}/render`, json({ tokens: echoed, format: 'wav' }));
  check('repeated words are lifted from the speaker’s own recording',
    echo.render.synthesis.fromVoiceBank === tail.length,
    `${echo.render.synthesis.fromVoiceBank}/${echo.render.synthesis.words} from the voice bank`);
  check('the echo makes the result longer', echo.render.duration > project.duration,
    `${echo.render.duration.toFixed(2)}s`);

  // ---------------------------------------------------------------- extras
  const mp3 = await api(`/api/projects/${id}/render`, json({ tokens: kept, format: 'mp3' }));
  check('mp3 export works', mp3.render.format === 'mp3' && mp3.render.bytes > 0);

  const empty = await api(`/api/projects/${id}/render`, json({ tokens: [], format: 'wav' }));
  check('an empty edit renders without error', empty.render.duration < 0.2);

  await api(`/api/projects/${id}`, { method: 'DELETE' });
  check('project deleted', true);
} catch (error) {
  console.error(`\nAborted: ${error.message}`);
  failures += 1;
} finally {
  await fs.rm(workDir, { recursive: true, force: true });
}

console.log(failures === 0 ? '\nAll end-to-end checks passed.' : `\n${failures} check(s) failed.`);
process.exit(failures === 0 ? 0 : 1);

/** @param {string} url @param {string} destination */
async function downloadTo(url, destination) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`download failed: ${response.status}`);
  await fs.writeFile(destination, Buffer.from(await response.arrayBuffer()));
}

/**
 * Read back what the rendered file actually says, using the model server that
 * is already running rather than a second copy of the model.
 */
async function transcribe(filePath) {
  const form = new FormData();
  form.append('file', new Blob([await fs.readFile(filePath)]), path.basename(filePath));
  const created = await api('/api/projects', { method: 'POST', body: form });
  const id = created.project.id;

  const deadline = Date.now() + 5 * 60 * 1000;
  for (;;) {
    const { project } = await api(`/api/projects/${id}`);
    if (project.status === 'ready') {
      await api(`/api/projects/${id}`, { method: 'DELETE' });
      return project.transcript.words.map((w) => w.text).join(' ');
    }
    if (project.status === 'failed') throw new Error(`verification transcribe failed: ${project.error}`);
    if (Date.now() > deadline) throw new Error('verification transcribe timed out');
    await new Promise((r) => setTimeout(r, 1200));
  }
}
