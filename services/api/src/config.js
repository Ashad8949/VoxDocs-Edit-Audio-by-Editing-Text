import path from 'node:path';
import process from 'node:process';

const env = process.env;

/** @param {string} name @param {number} fallback */
function num(name, fallback) {
  const raw = env[name];
  if (raw === undefined || raw === '') return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export const config = {
  port: num('PORT', 3000),
  dataDir: path.resolve(env.VOXDOCS_DATA_DIR ?? './data'),
  modelUrl: (env.VOXDOCS_MODEL_URL ?? 'http://localhost:8000').replace(/\/+$/, ''),
  /** Transcription of a long file is slow; this ceiling is generous on purpose. */
  modelTimeoutMs: num('VOXDOCS_MODEL_TIMEOUT_MS', 30 * 60 * 1000),
  maxUploadBytes: num('VOXDOCS_MAX_UPLOAD_MB', 1024) * 1024 * 1024,

  /** Everything is rendered at this rate so segments concatenate sample-exactly. */
  renderSampleRate: num('VOXDOCS_RENDER_RATE', 48000),
  /** Short fade at every seam. Long enough to kill clicks, short enough to be inaudible. */
  seamFadeSeconds: num('VOXDOCS_SEAM_FADE', 0.008),
  /** Beyond this many pieces, the render is done in batches. */
  maxSegmentsPerPass: num('VOXDOCS_MAX_SEGMENTS_PER_PASS', 400),

  ffmpeg: env.VOXDOCS_FFMPEG ?? 'ffmpeg',
  ffprobe: env.VOXDOCS_FFPROBE ?? 'ffprobe',

  corsOrigin: env.VOXDOCS_CORS_ORIGIN ?? '*',
};

export const AUDIO_EXTENSIONS = new Set([
  '.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.oga', '.opus', '.wma', '.aiff', '.aif',
]);

export const VIDEO_EXTENSIONS = new Set([
  '.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.mpg', '.mpeg', '.wmv',
]);
