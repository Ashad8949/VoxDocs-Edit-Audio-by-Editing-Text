/**
 * Render tests. These run real ffmpeg against real generated audio — the whole
 * point is to check that samples come out where the EDL says they should.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

const workDir = await fs.mkdtemp(path.join(os.tmpdir(), 'voxdocs-render-'));
process.env.VOXDOCS_DATA_DIR = workDir;

const { expandSegments, opDurations, renderOps, encodeAudio } = await import('../src/render.js');
const { durationOf, probe } = await import('../src/ffmpeg.js');

test.after(() => fs.rm(workDir, { recursive: true, force: true }));

/**
 * Build a master file made of distinct 1-second tones, so it is possible to
 * tell from the output exactly which parts of the source survived.
 * Second 0 = 200 Hz, second 1 = 400 Hz, second 2 = 800 Hz, second 3 = 1600 Hz.
 */
const TONES = [200, 400, 800, 1600];
let masterPath;

test('build a tone master', async () => {
  masterPath = path.join(workDir, 'master.wav');
  const filters = TONES.map(
    (hz, i) => `sine=frequency=${hz}:duration=1:sample_rate=48000[t${i}]`
  );
  const labels = TONES.map((_, i) => `[t${i}]`).join('');
  const script = `${filters.join(';')};${labels}concat=n=${TONES.length}:v=0:a=1[out]`;
  await execFileAsync('ffmpeg', [
    '-nostdin', '-v', 'error', '-y',
    '-filter_complex', script, '-map', '[out]',
    '-ac', '1', '-ar', '48000', '-c:a', 'pcm_f32le',
    masterPath,
  ]);
  assert.ok((await durationOf(masterPath)) > 3.9);
});

/** Dominant frequency of a slice of a wav file, via ffmpeg's showfreqs-free path. */
async function dominantFrequency(filePath, start, duration) {
  const { stdout } = await execFileAsync('ffprobe', [
    '-v', 'error', '-f', 'lavfi',
    '-i', `amovie=${filePath.replace(/[\\:']/g, (c) => `\\${c}`)},atrim=start=${start}:duration=${duration},astats=metadata=1:reset=0`,
    '-show_entries', 'frame_tags=lavfi.astats.1.Zero_crossings_rate',
    '-of', 'csv=p=0',
  ]);
  const rates = stdout.trim().split('\n').map(Number).filter((n) => Number.isFinite(n) && n > 0);
  if (rates.length === 0) return 0;
  const rate = rates[rates.length - 1];
  // Zero crossings per sample -> Hz for a sine: two crossings per cycle.
  return (rate * 48000) / 2;
}

test('a single copy op reproduces exactly that slice of the source', async () => {
  const out = path.join(workDir, 'copy.wav');
  const result = await renderOps(masterPath, [{ kind: 'source', start: 2.0, end: 3.0 }], out);

  assert.ok(Math.abs(result.duration - 1.0) < 0.02, `duration ${result.duration}`);
  assert.equal(result.pieces, 1);
  const hz = await dominantFrequency(out, 0.1, 0.6);
  assert.ok(Math.abs(hz - 800) < 60, `expected the 800 Hz tone, measured ${hz}`);
});

test('deleting the middle joins the outer pieces and drops the inner audio', async () => {
  const out = path.join(workDir, 'cut.wav');
  const result = await renderOps(
    masterPath,
    [
      { kind: 'source', start: 0.0, end: 1.0 },
      { kind: 'source', start: 3.0, end: 4.0 },
    ],
    out
  );

  assert.ok(Math.abs(result.duration - 2.0) < 0.03, `duration ${result.duration}`);
  assert.equal(result.pieces, 2);

  const first = await dominantFrequency(out, 0.2, 0.5);
  const second = await dominantFrequency(out, 1.2, 0.5);
  assert.ok(Math.abs(first - 200) < 40, `first piece measured ${first}`);
  assert.ok(Math.abs(second - 1600) < 120, `second piece measured ${second}`);
});

test('silence ops contribute their exact duration', async () => {
  const out = path.join(workDir, 'silence.wav');
  const result = await renderOps(
    masterPath,
    [
      { kind: 'source', start: 0.0, end: 0.5 },
      { kind: 'silence', duration: 0.25 },
      { kind: 'source', start: 0.0, end: 0.5 },
    ],
    out
  );
  assert.ok(Math.abs(result.duration - 1.25) < 0.03, `duration ${result.duration}`);
});

test('an empty edit renders a valid, essentially empty file rather than failing', async () => {
  const out = path.join(workDir, 'empty.wav');
  const result = await renderOps(masterPath, [], out);
  assert.equal(result.pieces, 0);
  const info = await probe(out);
  assert.ok(info.hasAudio);
  assert.ok(info.duration < 0.2);
});

test('zero-length and sub-millisecond ops are dropped before they reach ffmpeg', async () => {
  const { ops } = await expandSegments(
    [
      { kind: 'copy', start: 1.0, end: 1.0, wordIds: ['a'], firstWordIndex: 0, lastWordIndex: 0 },
      { kind: 'copy', start: 2.0, end: 3.0, wordIds: ['b'], firstWordIndex: 1, lastWordIndex: 1 },
    ],
    new Map(),
    path.join(workDir, 'inline-a')
  );
  assert.equal(ops.length, 1);
  assert.equal(ops[0].start, 2.0);
});

test('seam fades keep the join continuous instead of clicking', async () => {
  // Concatenating the tail of a tone onto the head of a different tone steps the
  // waveform; the fade must pull the samples through zero at the join.
  const out = path.join(workDir, 'seam.wav');
  await renderOps(
    masterPath,
    [
      { kind: 'source', start: 0.4, end: 0.9 },
      { kind: 'source', start: 3.1, end: 3.6 },
    ],
    out,
    { fade: 0.01 }
  );

  // Extract raw samples around the seam at 0.5 s and check the peak excursion
  // between neighbouring samples stays small.
  const { stdout } = await execFileAsync('ffmpeg', [
    '-nostdin', '-v', 'error', '-i', out,
    '-af', 'atrim=start=0.495:end=0.505',
    '-f', 'f32le', '-ac', '1', '-ar', '48000', '-',
  ], { encoding: 'buffer', maxBuffer: 1 << 24 });

  const samples = new Float32Array(
    stdout.buffer.slice(stdout.byteOffset, stdout.byteOffset + stdout.length)
  );
  assert.ok(samples.length > 100, 'got samples around the seam');
  let maxStep = 0;
  for (let i = 1; i < samples.length; i++) {
    maxStep = Math.max(maxStep, Math.abs(samples[i] - samples[i - 1]));
  }
  // A 1600 Hz sine at 48 kHz steps ~0.21 per sample at its steepest; a hard
  // splice discontinuity would be far larger than that.
  assert.ok(maxStep < 0.35, `discontinuity at the seam: ${maxStep}`);
});

test('opDurations reports source and silence lengths without probing', async () => {
  const durations = await opDurations([
    { kind: 'source', start: 1.0, end: 2.5 },
    { kind: 'silence', duration: 0.2 },
  ]);
  assert.deepEqual(durations, [1.5, 0.2]);
});

test('expandSegments turns voice-bank units into ordinary source ops', async () => {
  const segments = [
    { kind: 'copy', start: 0, end: 1, wordIds: ['w0'], firstWordIndex: 0, lastWordIndex: 0 },
    { kind: 'synth', text: 'hello', contextBefore: null, contextAfter: null, leadGap: 0, trailGap: 0 },
  ];
  const synthesis = new Map([
    [1, { units: [{ type: 'source', start: 2.0, end: 2.4, gain: 1, word: 'hello' }], missing: [] }],
  ]);
  const { ops, warnings } = await expandSegments(segments, synthesis, path.join(workDir, 'inline-b'));

  assert.equal(ops.length, 2);
  assert.equal(ops[1].kind, 'source', 'a bank hit is lifted from the master, not shipped as audio');
  assert.equal(ops[1].start, 2.0);
  assert.deepEqual(warnings, []);
});

test('expandSegments materialises inline TTS audio to disk', async () => {
  const wavPath = path.join(workDir, 'tone.wav');
  await execFileAsync('ffmpeg', [
    '-nostdin', '-v', 'error', '-y', '-f', 'lavfi',
    '-i', 'sine=frequency=440:duration=0.3:sample_rate=48000',
    '-ac', '1', '-c:a', 'pcm_s16le', wavPath,
  ]);
  const data = (await fs.readFile(wavPath)).toString('base64');

  const segments = [
    { kind: 'synth', text: 'zebra', contextBefore: null, contextAfter: null, leadGap: 0, trailGap: 0 },
  ];
  const synthesis = new Map([
    [0, { units: [{ type: 'audio', data, sample_rate: 48000, word: 'zebra' }], missing: [] }],
  ]);
  const inlineDir = path.join(workDir, 'inline-c');
  const { ops } = await expandSegments(segments, synthesis, inlineDir);

  assert.equal(ops.length, 1);
  assert.equal(ops[0].kind, 'file');
  assert.ok((await fs.stat(ops[0].file)).size > 0);

  const out = path.join(workDir, 'inline-render.wav');
  const result = await renderOps(masterPath, ops, out);
  assert.ok(Math.abs(result.duration - 0.3) < 0.05, `duration ${result.duration}`);
});

test('words nothing could synthesise are surfaced as warnings, not swallowed', async () => {
  const segments = [
    { kind: 'synth', text: 'zebra', contextBefore: null, contextAfter: null, leadGap: 0, trailGap: 0 },
  ];
  const synthesis = new Map([[0, { units: [], missing: ['zebra'] }]]);
  const { ops, warnings } = await expandSegments(segments, synthesis, path.join(workDir, 'inline-d'));

  assert.equal(ops.length, 0);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /zebra/);
});

test('a segment with no synthesis result warns rather than rendering silence', async () => {
  const segments = [
    { kind: 'synth', text: 'ghost', contextBefore: null, contextAfter: null, leadGap: 0, trailGap: 0 },
  ];
  const { ops, warnings } = await expandSegments(segments, new Map(), path.join(workDir, 'inline-e'));
  assert.equal(ops.length, 0);
  assert.match(warnings[0], /ghost/);
});

test('batched rendering matches a single-pass render of the same ops', async () => {
  const ops = Array.from({ length: 12 }, (_, i) => ({
    kind: 'source',
    start: (i % 4) * 1.0,
    end: (i % 4) * 1.0 + 0.25,
  }));

  const single = path.join(workDir, 'single.wav');
  await renderOps(masterPath, ops, single);

  // Force the batching path by shrinking the per-pass limit.
  const { config } = await import('../src/config.js');
  const original = config.maxSegmentsPerPass;
  config.maxSegmentsPerPass = 5;
  const batched = path.join(workDir, 'batched.wav');
  const result = await renderOps(masterPath, ops, batched);
  config.maxSegmentsPerPass = original;

  assert.equal(result.pieces, 12);
  const a = await durationOf(single);
  const b = await durationOf(batched);
  assert.ok(Math.abs(a - b) < 0.05, `single ${a} vs batched ${b}`);
  assert.ok(Math.abs(b - 3.0) < 0.1, `expected ~3 s, got ${b}`);
});

test('encodeAudio produces each delivery format', async () => {
  const source = path.join(workDir, 'copy.wav');
  for (const [format, codec] of [['wav', 'wav'], ['mp3', 'mp3'], ['m4a', 'mp4']]) {
    const out = path.join(workDir, `enc.${format}`);
    await encodeAudio(source, out, format);
    const info = await probe(out);
    assert.ok(info.hasAudio, `${format} has audio`);
    assert.match(info.format, new RegExp(codec));
  }
});

// ------------------------------------------------------------------- video

/** A 4-second 25 fps clip to apply the same EDL to. */
let videoPath;

test('build a test video', async () => {
  videoPath = path.join(workDir, 'clip.mp4');
  await execFileAsync('ffmpeg', [
    '-nostdin', '-v', 'error', '-y',
    '-f', 'lavfi', '-i', 'testsrc=size=160x120:rate=25:duration=4',
    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=4:sample_rate=48000',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest',
    videoPath,
  ]);
  const info = await probe(videoPath);
  assert.ok(info.hasVideo && info.hasAudio);
});

/** Duration of a single stream, which is what sync actually depends on. */
async function streamDuration(filePath, kind) {
  const { stdout } = await execFileAsync('ffprobe', [
    '-v', 'error', '-select_streams', kind === 'video' ? 'v' : 'a',
    '-show_entries', 'stream=duration', '-of', 'csv=p=0', filePath,
  ]);
  return Number(stdout.trim().split('\n')[0]);
}

test('video cuts follow the same EDL as the audio', async () => {
  const { renderVideo } = await import('../src/render.js');
  const segments = [
    { kind: 'copy', start: 0.0, end: 1.0, wordIds: ['w0'], firstWordIndex: 0, lastWordIndex: 0 },
    { kind: 'copy', start: 3.0, end: 4.0, wordIds: ['w3'], firstWordIndex: 3, lastWordIndex: 3 },
  ];
  const audio = path.join(workDir, 'cut.wav'); // rendered earlier: 2 s
  const out = path.join(workDir, 'cut.mp4');
  await renderVideo(videoPath, audio, segments, new Map(), out);

  const info = await probe(out);
  assert.ok(info.hasVideo && info.hasAudio);
  assert.ok(Math.abs(info.duration - 2.0) < 0.1, `expected ~2 s, got ${info.duration}`);
});

test('an insertion that opens the timeline pre-rolls a freeze frame', async () => {
  // Regression: a zero-length leading shot yielded no frames at all, so tpad had
  // nothing to clone and the picture came out shorter than the sound.
  const { renderVideo } = await import('../src/render.js');
  const segments = [
    { kind: 'synth', text: 'intro', contextBefore: null, contextAfter: null, leadGap: 0, trailGap: 0 },
    { kind: 'copy', start: 2.0, end: 4.0, wordIds: ['w2'], firstWordIndex: 2, lastWordIndex: 3 },
  ];

  // 0.5 s of inserted speech in front of 2 s of copied audio.
  const audio = path.join(workDir, 'lead-audio.wav');
  await renderOps(
    masterPath,
    [{ kind: 'silence', duration: 0.5 }, { kind: 'source', start: 2.0, end: 4.0 }],
    audio
  );

  const out = path.join(workDir, 'lead.mp4');
  await renderVideo(videoPath, audio, segments, new Map([[0, 0.5]]), out);

  const videoSeconds = await streamDuration(out, 'video');
  const audioSeconds = await streamDuration(out, 'audio');
  assert.ok(videoSeconds > 2.3, `the freeze frame is missing: video is only ${videoSeconds}s`);
  // Within one frame at 25 fps; no audio may be truncated to match the picture.
  assert.ok(
    Math.abs(videoSeconds - audioSeconds) <= 0.045,
    `picture and sound disagree: video ${videoSeconds}s vs audio ${audioSeconds}s`
  );
});

test('an insertion in the middle freezes the shot that precedes it', async () => {
  const { renderVideo } = await import('../src/render.js');
  const segments = [
    { kind: 'copy', start: 0.0, end: 1.0, wordIds: ['w0'], firstWordIndex: 0, lastWordIndex: 0 },
    { kind: 'synth', text: 'middle', contextBefore: 'a', contextAfter: 'b', leadGap: 0, trailGap: 0 },
    { kind: 'copy', start: 3.0, end: 4.0, wordIds: ['w3'], firstWordIndex: 3, lastWordIndex: 3 },
  ];
  const audio = path.join(workDir, 'mid-audio.wav');
  await renderOps(
    masterPath,
    [
      { kind: 'source', start: 0.0, end: 1.0 },
      { kind: 'silence', duration: 0.4 },
      { kind: 'source', start: 3.0, end: 4.0 },
    ],
    audio
  );

  const out = path.join(workDir, 'mid.mp4');
  await renderVideo(videoPath, audio, segments, new Map([[1, 0.4]]), out);

  const videoSeconds = await streamDuration(out, 'video');
  assert.ok(Math.abs(videoSeconds - 2.4) < 0.08, `expected ~2.4 s of picture, got ${videoSeconds}`);
});

test('an edit that removes every frame is refused with a clear message', async () => {
  const { renderVideo } = await import('../src/render.js');
  await assert.rejects(
    () => renderVideo(videoPath, path.join(workDir, 'cut.wav'), [], new Map(),
      path.join(workDir, 'none.mp4')),
    /removed every frame/
  );
});

test('encodeAudio rejects an unknown format', async () => {
  await assert.rejects(
    () => encodeAudio(path.join(workDir, 'copy.wav'), path.join(workDir, 'x.ogg'), 'ogg'),
    /unsupported audio format/
  );
});
