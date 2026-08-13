/**
 * Store, routing and HTTP contract tests.
 *
 * The model server is stubbed with a local HTTP server so the whole ingest and
 * render path can be exercised deterministically, without loading an ASR model.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

const workDir = await fs.mkdtemp(path.join(os.tmpdir(), 'voxdocs-api-'));
process.env.VOXDOCS_DATA_DIR = workDir;

// ---------------------------------------------------------------- stub model

/** Words the fake ASR "hears", matching the tone master built below. */
const FAKE_WORDS = [
  { id: 'w0', text: 'alpha', start: 0.05, end: 0.9, confidence: 0.9 },
  { id: 'w1', text: 'bravo', start: 1.05, end: 1.9, confidence: 0.9 },
  { id: 'w2', text: 'charlie', start: 2.05, end: 2.9, confidence: 0.9 },
  { id: 'w3', text: 'delta', start: 3.05, end: 3.9, confidence: 0.9 },
];

const modelCalls = { transcribe: 0, synthesize: 0, reseed: 0 };
let failNextSynthesisWith409 = false;

const modelServer = http.createServer((req, res) => {
  const chunks = [];
  req.on('data', (c) => chunks.push(c));
  req.on('end', () => {
    const send = (status, body) => {
      res.writeHead(status, { 'content-type': 'application/json' });
      res.end(JSON.stringify(body));
    };

    if (req.url === '/health') return send(200, { status: 'ok' });

    if (req.url === '/transcribe') {
      modelCalls.transcribe += 1;
      return send(200, {
        words: FAKE_WORDS,
        segments: [{ start: 0, end: 3.9, text: 'alpha bravo charlie delta', first_word: 0, last_word: 3 }],
        language: 'en',
        backend: 'stub',
        duration: 4.0,
        envelope: { fps: 10, rms: new Array(40).fill(0.3) },
        voice: { median_f0: 120, speech_rms: 0.1, peak: 0.8, sample_rate: 16000 },
      });
    }

    if (req.url === '/voice-profile') {
      modelCalls.reseed += 1;
      return send(200, { project_id: 'x', words: FAKE_WORDS.length });
    }

    if (req.url === '/synthesize/batch') {
      if (failNextSynthesisWith409) {
        failNextSynthesisWith409 = false;
        return send(409, { error: 'voice_profile_missing' });
      }
      modelCalls.synthesize += 1;
      const body = JSON.parse(Buffer.concat(chunks).toString() || '{}');
      const results = (body.items ?? []).map((item) => ({
        // Pretend every insertion is covered by the bank, lifting second 1.
        units: [{ type: 'source', start: 1.0, end: 1.5, gain: 1, word: item.text }],
        backends: ['voice-bank'],
        covered: item.text.split(' '),
        generated: [],
        missing: [],
        coverage: 1,
      }));
      return send(200, { results });
    }

    send(404, { error: 'not_found' });
  });
});

await new Promise((resolve) => modelServer.listen(0, '127.0.0.1', resolve));
process.env.VOXDOCS_MODEL_URL = `http://127.0.0.1:${modelServer.address().port}`;

// Import only after the environment is set, since config reads it at load time.
const store = await import('../src/store.js');
const { createApp } = await import('../src/app.js');
const { resolveTokens, ValidationError } = await import('../src/pipeline.js');

await store.init();
const app = createApp();
const server = app.listen(0, '127.0.0.1');
await new Promise((resolve) => server.once('listening', resolve));
const base = `http://127.0.0.1:${server.address().port}`;

test.after(async () => {
  server.close();
  modelServer.close();
  await fs.rm(workDir, { recursive: true, force: true });
});

/** Four one-second tones, matching FAKE_WORDS. */
const mediaPath = path.join(workDir, 'input.wav');
await execFileAsync('ffmpeg', [
  '-nostdin', '-v', 'error', '-y', '-f', 'lavfi',
  '-i', 'sine=frequency=440:duration=4:sample_rate=44100',
  '-ac', '1', '-c:a', 'pcm_s16le', mediaPath,
]);

/** @param {string} pathname @param {RequestInit} [init] */
const api = (pathname, init) => fetch(`${base}${pathname}`, init);

// -------------------------------------------------------------------- store

test('project ids that could escape the data directory are rejected', () => {
  for (const bad of ['../etc', 'a/../../b', '..', '/etc/passwd', 'a/b', 'short', '', 'a'.repeat(200)]) {
    assert.throws(() => store.projectDir(bad), /invalid project id/, `accepted ${JSON.stringify(bad)}`);
  }
});

test('well-formed project ids resolve inside the data directory', () => {
  const dir = store.projectDir('abcdef123456');
  assert.ok(dir.startsWith(path.join(workDir, 'projects')));
});

test('newId produces ids the validator accepts', () => {
  for (let i = 0; i < 50; i++) assert.doesNotThrow(() => store.projectDir(store.newId()));
});

test('saving is atomic and leaves no partial document behind', async () => {
  const project = { id: store.newId(), name: 'x', createdAt: new Date().toISOString(), status: 'ready' };
  await store.save(project);
  const loaded = await store.load(project.id);
  assert.equal(loaded.name, 'x');
  assert.ok(loaded.updatedAt);
  const entries = await fs.readdir(store.projectDir(project.id));
  assert.ok(!entries.some((e) => e.endsWith('.tmp')));
});

test('loading a missing project raises NotFound', async () => {
  await assert.rejects(() => store.load('doesnotexist1'), /project not found/);
});

// ------------------------------------------------------------------- health

test('health reports the model server status', async () => {
  const body = await (await api('/api/health')).json();
  assert.equal(body.status, 'ok');
  assert.equal(body.renderSampleRate, 48000);
});

test('unknown routes return a json 404', async () => {
  const response = await api('/api/nope');
  assert.equal(response.status, 404);
  assert.equal((await response.json()).error, 'not_found');
});

// ------------------------------------------------------------------ ingest

let projectId;

test('uploading a file creates a project and transcribes it', async () => {
  const form = new FormData();
  form.append('file', new Blob([await fs.readFile(mediaPath)]), 'input.wav');
  form.append('name', 'Tone Test');

  const response = await api('/api/projects', { method: 'POST', body: form });
  assert.equal(response.status, 202, 'ingest is accepted, not awaited');
  const { project } = await response.json();
  projectId = project.id;
  assert.equal(project.status, 'queued');

  // Poll until the background ingest finishes.
  let current;
  for (let i = 0; i < 100; i++) {
    current = (await (await api(`/api/projects/${projectId}`)).json()).project;
    if (current.status === 'ready' || current.status === 'failed') break;
    await new Promise((r) => setTimeout(r, 100));
  }
  assert.equal(current.status, 'ready', `ingest failed: ${current.error}`);
  assert.equal(current.transcript.words.length, 4);
  assert.equal(modelCalls.transcribe, 1);
  assert.ok(Math.abs(current.duration - 4.0) < 0.1);
});

test('the project appears in the listing without its bulky word array', async () => {
  const { projects } = await (await api('/api/projects')).json();
  const found = projects.find((p) => p.id === projectId);
  assert.ok(found);
  assert.equal(found.wordCount, 4);
  assert.equal(found.transcript, undefined);
});

test('rejecting an unsupported file extension', async () => {
  const form = new FormData();
  form.append('file', new Blob([Buffer.from('hello')]), 'notes.txt');
  const response = await api('/api/projects', { method: 'POST', body: form });
  assert.equal(response.status, 400);
  assert.match((await response.json()).message, /unsupported file type/);
});

test('uploading with no file at all is a 400', async () => {
  const response = await api('/api/projects', { method: 'POST', body: new FormData() });
  assert.equal(response.status, 400);
});

test('a file with no audio track fails the project with a readable reason', async () => {
  const silentVideo = path.join(workDir, 'novideo.mp4');
  await execFileAsync('ffmpeg', [
    '-nostdin', '-v', 'error', '-y', '-f', 'lavfi',
    '-i', 'color=c=black:s=64x64:d=1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', silentVideo,
  ]);

  const form = new FormData();
  form.append('file', new Blob([await fs.readFile(silentVideo)]), 'novideo.mp4');
  const { project } = await (await api('/api/projects', { method: 'POST', body: form })).json();

  let current;
  for (let i = 0; i < 100; i++) {
    current = (await (await api(`/api/projects/${project.id}`)).json()).project;
    if (current.status === 'failed' || current.status === 'ready') break;
    await new Promise((r) => setTimeout(r, 100));
  }
  assert.equal(current.status, 'failed');
  assert.match(current.error, /no audio track/);
});

// ------------------------------------------------------------------ editing

test('planning an unchanged transcript reports no edits', async () => {
  const response = await api(`/api/projects/${projectId}/plan`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({}),
  });
  const { stats } = await response.json();
  assert.equal(stats.keptWords, 4);
  assert.equal(stats.deletedWords, 0);
  assert.equal(stats.cuts, 0);
});

test('planning a deletion reports the shorter result before rendering', async () => {
  const response = await api(`/api/projects/${projectId}/plan`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text: 'alpha delta' }),
  });
  const { stats } = await response.json();
  assert.equal(stats.deletedWords, 2);
  assert.equal(stats.cuts, 1);
  assert.ok(stats.estimatedDuration < stats.sourceDuration);
});

test('planning accepts an explicit token list from the editor', async () => {
  const response = await api(`/api/projects/${projectId}/plan`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ tokens: [{ ref: 'w0' }, { insert: 'new words' }, { ref: 'w3' }] }),
  });
  const { stats } = await response.json();
  assert.equal(stats.keptWords, 2);
  assert.equal(stats.insertedWords, 2);
});

test('a malformed token list is rejected', async () => {
  const response = await api(`/api/projects/${projectId}/plan`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ tokens: [{ bogus: true }] }),
  });
  assert.equal(response.status, 400);
});

test('resolveTokens falls back to the identity edit', () => {
  const project = { transcript: { words: FAKE_WORDS } };
  assert.equal(resolveTokens(project, {}).length, 4);
  assert.equal(resolveTokens(project, { text: 'alpha' }).length, 1);
  assert.throws(() => resolveTokens(project, { tokens: [42] }), ValidationError);
});

// ----------------------------------------------------------------- rendering

test('rendering a deletion produces a shorter file that downloads', async () => {
  const response = await api(`/api/projects/${projectId}/render`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text: 'alpha delta', format: 'wav' }),
  });
  const { render, downloadUrl } = await response.json();

  assert.equal(render.stats.deletedWords, 2);
  assert.equal(render.pieces, 2);
  assert.ok(render.duration < 3.0, `expected under 3 s, got ${render.duration}`);
  assert.ok(render.bytes > 1000);

  const download = await api(downloadUrl);
  assert.equal(download.status, 200);
  assert.match(download.headers.get('content-disposition'), /attachment; filename="Tone-Test-edited\.wav"/);
  assert.ok((await download.arrayBuffer()).byteLength > 1000);
});

test('rendering an insertion consults the model server', async () => {
  const before = modelCalls.synthesize;
  const response = await api(`/api/projects/${projectId}/render`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ tokens: [{ ref: 'w0' }, { insert: 'echo' }, { ref: 'w3' }] }),
  });
  const { render } = await response.json();
  assert.equal(modelCalls.synthesize, before + 1);
  assert.equal(render.synthesis.fromVoiceBank, 1);
  assert.equal(render.synthesis.coverage, 1);
  assert.deepEqual(render.warnings, []);
});

test('an evicted voice profile is re-seeded transparently', async () => {
  failNextSynthesisWith409 = true;
  const before = modelCalls.reseed;
  const response = await api(`/api/projects/${projectId}/render`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ tokens: [{ ref: 'w0' }, { insert: 'echo' }] }),
  });
  assert.equal(response.status, 200);
  assert.equal(modelCalls.reseed, before + 1, 'the profile should have been re-seeded once');
});

test('rendering rejects an unsupported output format', async () => {
  const response = await api(`/api/projects/${projectId}/render`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ format: 'ogg' }),
  });
  assert.equal(response.status, 400);
  assert.match((await response.json()).message, /unsupported format/);
});

test('rendering an audio-only project to mp3 works', async () => {
  const response = await api(`/api/projects/${projectId}/render`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text: 'alpha bravo charlie delta', format: 'mp3' }),
  });
  const { render } = await response.json();
  assert.equal(render.format, 'mp3');
  assert.ok(render.bytes > 500);
});

test('renders are listed on the project, newest first', async () => {
  const { project } = await (await api(`/api/projects/${projectId}`)).json();
  assert.ok(project.renders.length >= 2);
  const times = project.renders.map((r) => r.createdAt);
  assert.deepEqual(times, [...times].sort().reverse());
});

test('downloading an unknown render is a 404', async () => {
  const response = await api(`/api/projects/${projectId}/renders/nosuchrender`);
  assert.equal(response.status, 404);
});

// -------------------------------------------------------------------- media

test('media is served with range support so the player can seek', async () => {
  const full = await api(`/api/projects/${projectId}/media`);
  assert.equal(full.status, 200);
  assert.equal(full.headers.get('accept-ranges'), 'bytes');
  const total = Number(full.headers.get('content-length'));
  assert.ok(total > 0);

  const partial = await api(`/api/projects/${projectId}/media`, { headers: { range: 'bytes=0-99' } });
  assert.equal(partial.status, 206);
  assert.equal(partial.headers.get('content-length'), '100');
  assert.match(partial.headers.get('content-range'), new RegExp(`^bytes 0-99/${total}$`));
});

test('an unsatisfiable range is refused with 416', async () => {
  const response = await api(`/api/projects/${projectId}/media`, {
    headers: { range: 'bytes=99999999-' },
  });
  assert.equal(response.status, 416);
});

test('the envelope downsamples to the requested resolution', async () => {
  const full = await (await api(`/api/projects/${projectId}/envelope`)).json();
  assert.equal(full.rms.length, 40);

  const small = await (await api(`/api/projects/${projectId}/envelope?points=10`)).json();
  assert.equal(small.rms.length, 10);
  assert.equal(small.downsampled, true);
});

// ------------------------------------------------------------------ lifecycle

test('editing a project that is still importing is refused with 409', async () => {
  const project = {
    id: store.newId(), name: 'pending', createdAt: new Date().toISOString(),
    status: 'transcribing', transcript: { words: [] },
  };
  await store.save(project);
  const response = await api(`/api/projects/${project.id}/plan`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({}),
  });
  assert.equal(response.status, 409);
});

test('deleting a project removes it and its files', async () => {
  const { project } = await (await api(`/api/projects/${projectId}`)).json();
  const dir = store.projectDir(project.id);
  assert.ok((await fs.stat(dir)).isDirectory());

  assert.equal((await api(`/api/projects/${projectId}`, { method: 'DELETE' })).status, 200);
  assert.equal((await api(`/api/projects/${projectId}`)).status, 404);
  await assert.rejects(() => fs.stat(dir));
});

test('deleting a project twice is a 404, not a silent success', async () => {
  const response = await api(`/api/projects/${projectId}`, { method: 'DELETE' });
  assert.equal(response.status, 404);
});

test('requests for a traversal-shaped project id are refused', async () => {
  for (const bad of ['..%2f..%2fetc', '..', 'a%2Fb']) {
    const response = await api(`/api/projects/${bad}`);
    assert.ok(response.status === 404, `expected 404 for ${bad}, got ${response.status}`);
  }
});
