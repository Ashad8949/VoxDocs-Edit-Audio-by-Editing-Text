/**
 * Thin promise wrapper around the ffmpeg/ffprobe binaries.
 *
 * Arguments are always passed as an array — never interpolated into a shell
 * string — so a filename containing a quote or a semicolon is data, not code.
 */

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fs from 'node:fs/promises';
import { config } from './config.js';

const execFileAsync = promisify(execFile);

export class FfmpegError extends Error {
  /** @param {string} message @param {string} [stderr] */
  constructor(message, stderr = '') {
    super(message);
    this.name = 'FfmpegError';
    this.stderr = stderr;
  }
}

/**
 * Run ffmpeg.
 *
 * When a `filterScript` is supplied it is written to a file and passed via
 * -filter_complex_script rather than on the command line: a heavily edited
 * transcript produces a filter graph far longer than the OS argument limit.
 *
 * @param {string[]} args every argument except the filter graph
 * @param {{ filterScript?: string, timeoutMs?: number }} [options]
 */
export async function runFfmpeg(args, options = {}) {
  let scriptPath;
  try {
    let finalArgs = args;
    if (options.filterScript) {
      scriptPath = `${await tempName('filter')}.txt`;
      await fs.writeFile(scriptPath, options.filterScript, 'utf8');
      // Filter options must precede the output file, which callers put last.
      finalArgs = [
        ...args.slice(0, -1),
        '-filter_complex_script', scriptPath,
        args[args.length - 1],
      ];
    }

    const { stderr } = await execFileAsync(config.ffmpeg, ['-nostdin', '-v', 'error', ...finalArgs], {
      maxBuffer: 32 * 1024 * 1024,
      timeout: options.timeoutMs ?? 0,
    });
    return stderr ?? '';
  } catch (error) {
    const stderr = /** @type {any} */ (error)?.stderr ?? '';
    const detail = String(stderr).trim().split('\n').slice(-3).join(' ');
    throw new FfmpegError(`ffmpeg failed: ${detail || error.message}`, String(stderr));
  } finally {
    if (scriptPath) await fs.rm(scriptPath, { force: true });
  }
}

/**
 * Probe a media file.
 * @param {string} filePath
 * @returns {Promise<{ duration: number, hasVideo: boolean, hasAudio: boolean,
 *                     width: number, height: number, fps: number, format: string }>}
 */
export async function probe(filePath) {
  const args = [
    '-v', 'error',
    '-show_entries', 'format=duration,format_name',
    '-show_entries', 'stream=codec_type,width,height,avg_frame_rate',
    '-of', 'json',
    filePath,
  ];
  let stdout;
  try {
    ({ stdout } = await execFileAsync(config.ffprobe, args, { maxBuffer: 8 * 1024 * 1024 }));
  } catch (error) {
    throw new FfmpegError('could not read media file', /** @type {any} */ (error)?.stderr ?? '');
  }

  const parsed = JSON.parse(stdout || '{}');
  const streams = parsed.streams ?? [];
  const video = streams.find((s) => s.codec_type === 'video');
  const audio = streams.find((s) => s.codec_type === 'audio');

  let fps = 0;
  if (video?.avg_frame_rate && video.avg_frame_rate !== '0/0') {
    const [numerator, denominator] = String(video.avg_frame_rate).split('/').map(Number);
    if (denominator) fps = numerator / denominator;
  }

  return {
    duration: Number(parsed.format?.duration ?? 0) || 0,
    format: parsed.format?.format_name ?? '',
    hasVideo: Boolean(video),
    hasAudio: Boolean(audio),
    width: Number(video?.width ?? 0) || 0,
    height: Number(video?.height ?? 0) || 0,
    fps,
  };
}

/** Duration of a file in seconds, 0 when unknown. */
export async function durationOf(filePath) {
  try {
    return (await probe(filePath)).duration;
  } catch {
    return 0;
  }
}

let counter = 0;
/** @param {string} prefix */
export async function tempName(prefix) {
  const dir = `${config.dataDir}/tmp`;
  await fs.mkdir(dir, { recursive: true });
  counter += 1;
  return `${dir}/${prefix}-${process.pid}-${Date.now()}-${counter}`;
}

/**
 * Decode any input to the canonical render format: mono, float32, fixed rate.
 * Working in one format throughout means concatenation is sample-exact and no
 * hidden resampling creeps in between segments.
 * @param {string} input
 * @param {string} output
 */
export async function normalizeToMaster(input, output) {
  await runFfmpeg([
    '-y', '-i', input,
    '-map', 'a:0',
    '-ac', '1',
    '-ar', String(config.renderSampleRate),
    '-c:a', 'pcm_f32le',
    output,
  ]);
  return output;
}

/**
 * Produce a small AAC copy for the browser to stream while editing.
 * @param {string} input
 * @param {string} output
 */
export async function makePreviewAudio(input, output) {
  await runFfmpeg([
    '-y', '-i', input,
    '-map', 'a:0',
    '-ac', '1',
    '-ar', '44100',
    '-c:a', 'aac',
    '-b:a', '96k',
    '-movflags', '+faststart',
    output,
  ]);
  return output;
}
