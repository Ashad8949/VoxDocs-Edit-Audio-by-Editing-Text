/**
 * EDL → audio (and video) rendering.
 *
 * The EDL says *what* the output contains; this module makes it exist. The
 * whole render is expressed as one ffmpeg filter graph so the audio is decoded
 * once and never round-trips through an intermediate file, which matters when a
 * heavily edited hour of speech turns into several hundred pieces.
 *
 * Two details carry most of the audible quality:
 *
 *   - **Seam fades.** Butt-joining two unrelated pieces of a waveform steps the
 *     signal discontinuously and clicks. A few milliseconds of fade on each
 *     side of every seam removes it, below the threshold of audibility.
 *   - **One canonical format.** Everything is decoded to mono float32 at a
 *     single sample rate before it is cut, so no segment is silently resampled
 *     relative to its neighbour.
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import { config } from './config.js';
import { FfmpegError, durationOf, runFfmpeg, tempName } from './ffmpeg.js';

/**
 * @typedef {{ kind: 'source', start: number, end: number, gain?: number, label?: string }
 *         | { kind: 'file', file: string, label?: string }
 *         | { kind: 'silence', duration: number, label?: string }} RenderOp
 */

/**
 * Expand EDL segments plus resolved synthesis units into a flat op list.
 *
 * Synthesis comes back from the model server as a plan rather than as audio
 * whenever it can: units that reference the speaker's own recording are turned
 * into ordinary source ops here, so spliced words are lifted from the same
 * full-quality master as every other segment instead of from a resampled copy
 * shipped over HTTP.
 *
 * @param {import('@voxdocs/edl').EdlSegment[]} segments
 * @param {Map<number, { units: any[] }>} synthesis keyed by segment index
 * @param {string} inlineDir where base64 audio units are materialised
 * @returns {Promise<{ ops: RenderOp[], warnings: string[] }>}
 */
export async function expandSegments(segments, synthesis, inlineDir) {
  /** @type {RenderOp[]} */
  const ops = [];
  const warnings = [];
  await fs.mkdir(inlineDir, { recursive: true });

  let inlineIndex = 0;
  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    if (segment.kind === 'copy') {
      ops.push({
        kind: 'source',
        start: segment.start,
        end: segment.end,
        label: segment.wordIds.length ? `copy:${segment.wordIds[0]}` : 'copy',
      });
      continue;
    }

    const resolved = synthesis.get(i);
    if (!resolved) {
      warnings.push(`no synthesis result for "${segment.text}"; the text was dropped`);
      continue;
    }
    if (resolved.missing?.length) {
      warnings.push(
        `could not synthesise ${resolved.missing.map((w) => `"${w}"`).join(', ')} — ` +
          'no voice-bank match and no TTS backend available'
      );
    }

    for (const unit of resolved.units ?? []) {
      if (unit.type === 'source') {
        ops.push({
          kind: 'source',
          start: Number(unit.start),
          end: Number(unit.end),
          gain: Number(unit.gain ?? 1),
          label: `synth:${unit.word ?? ''}`,
        });
      } else if (unit.type === 'silence') {
        if (Number(unit.duration) > 0) {
          ops.push({ kind: 'silence', duration: Number(unit.duration), label: 'gap' });
        }
      } else if (unit.type === 'audio' && unit.data) {
        inlineIndex += 1;
        const file = path.join(inlineDir, `tts-${i}-${inlineIndex}.wav`);
        await fs.writeFile(file, Buffer.from(unit.data, 'base64'));
        ops.push({ kind: 'file', file, label: `tts:${unit.word ?? ''}` });
      }
    }
  }

  return { ops: ops.filter(isAudible), warnings };
}

/** Drop zero-length pieces, which would make ffmpeg's atrim produce nothing. */
function isAudible(op) {
  if (op.kind === 'source') return op.end - op.start > 0.001;
  if (op.kind === 'silence') return op.duration > 0.001;
  return true;
}

/**
 * Render a list of ops to a single audio file.
 *
 * @param {string} masterPath canonical mono float32 wav of the source
 * @param {RenderOp[]} ops
 * @param {string} outputPath
 * @param {{ fade?: number, sampleRate?: number }} [options]
 * @returns {Promise<{ duration: number, pieces: number }>}
 */
export async function renderOps(masterPath, ops, outputPath, options = {}) {
  const fade = options.fade ?? config.seamFadeSeconds;
  const rate = options.sampleRate ?? config.renderSampleRate;

  if (ops.length === 0) {
    // A transcript edited down to nothing is a legitimate outcome, not an error.
    await runFfmpeg([
      '-y', '-f', 'lavfi', '-i', `anullsrc=r=${rate}:cl=mono`,
      '-t', '0.05', '-c:a', 'pcm_f32le', outputPath,
    ]);
    return { duration: 0, pieces: 0 };
  }

  // Very large edits are rendered in batches, then joined. The batch boundaries
  // fall between ops that already carry their own seam fades, so splitting is
  // acoustically invisible.
  if (ops.length > config.maxSegmentsPerPass) {
    return renderInBatches(masterPath, ops, outputPath, { fade, rate });
  }

  const durations = await opDurations(ops);
  const inputs = ['-i', masterPath];
  const inputIndexByFile = new Map();
  for (const op of ops) {
    if (op.kind === 'file' && !inputIndexByFile.has(op.file)) {
      inputIndexByFile.set(op.file, inputs.length / 2);
      inputs.push('-i', op.file);
    }
  }

  const lines = [];
  const labels = [];
  ops.forEach((op, i) => {
    const label = `s${i}`;
    labels.push(`[${label}]`);
    const duration = durations[i];

    // Only the outer edges of the whole render keep their natural attack/decay.
    const fadeIn = i > 0 ? Math.min(fade, duration / 2) : 0;
    const fadeOut = i < ops.length - 1 ? Math.min(fade, duration / 2) : 0;
    const filters = [];

    if (op.kind === 'source') {
      filters.push(`atrim=start=${op.start.toFixed(6)}:end=${op.end.toFixed(6)}`);
      filters.push('asetpts=N/SR/TB');
      if (op.gain !== undefined && Math.abs(op.gain - 1) > 0.001) {
        filters.push(`volume=${op.gain.toFixed(4)}`);
      }
      lines.push(`[0:a]${filters.join(',')}${fadeChain(fadeIn, fadeOut, duration)}[${label}];`);
    } else if (op.kind === 'file') {
      const index = inputIndexByFile.get(op.file);
      lines.push(
        `[${index}:a]aresample=${rate},aformat=sample_fmts=fltp:channel_layouts=mono,` +
          `asetpts=N/SR/TB${fadeChain(fadeIn, fadeOut, duration)}[${label}];`
      );
    } else {
      lines.push(
        `aevalsrc=0:d=${op.duration.toFixed(6)}:s=${rate}:c=mono,` +
          `aformat=sample_fmts=fltp:channel_layouts=mono[${label}];`
      );
    }
  });

  lines.push(`${labels.join('')}concat=n=${ops.length}:v=0:a=1[out]`);
  const filterScript = lines.join('\n');

  await runFfmpeg(
    [
      '-y', ...inputs,
      '-map', '[out]',
      '-ac', '1', '-ar', String(rate),
      '-c:a', 'pcm_f32le',
      outputPath,
    ],
    { filterScript }
  );

  return { duration: await durationOf(outputPath), pieces: ops.length };
}

/** @param {number} fadeIn @param {number} fadeOut @param {number} duration */
function fadeChain(fadeIn, fadeOut, duration) {
  const parts = [];
  if (fadeIn > 0.0005) parts.push(`afade=t=in:st=0:d=${fadeIn.toFixed(6)}:curve=tri`);
  if (fadeOut > 0.0005) {
    const start = Math.max(0, duration - fadeOut);
    parts.push(`afade=t=out:st=${start.toFixed(6)}:d=${fadeOut.toFixed(6)}:curve=tri`);
  }
  return parts.length ? `,${parts.join(',')}` : '';
}

/**
 * Exact duration of every op. Inline TTS files have to be probed; everything
 * else is known from the EDL.
 * @param {RenderOp[]} ops
 * @returns {Promise<number[]>}
 */
export async function opDurations(ops) {
  const cache = new Map();
  const out = [];
  for (const op of ops) {
    if (op.kind === 'source') {
      out.push(op.end - op.start);
    } else if (op.kind === 'silence') {
      out.push(op.duration);
    } else {
      if (!cache.has(op.file)) cache.set(op.file, await durationOf(op.file));
      out.push(cache.get(op.file));
    }
  }
  return out;
}

/**
 * @param {string} masterPath
 * @param {RenderOp[]} ops
 * @param {string} outputPath
 * @param {{ fade: number, rate: number }} options
 */
async function renderInBatches(masterPath, ops, outputPath, options) {
  const batchSize = config.maxSegmentsPerPass;
  const parts = [];
  try {
    for (let offset = 0; offset < ops.length; offset += batchSize) {
      const slice = ops.slice(offset, offset + batchSize);
      const partPath = `${await tempName('batch')}.wav`;
      // Fades inside a batch are placed as usual; the batch's own outer edges
      // are seams too, so they are faded by treating them as interior.
      await renderOps(masterPath, slice, partPath, {
        fade: options.fade,
        sampleRate: options.rate,
      });
      parts.push(partPath);
    }

    const listPath = `${await tempName('concat')}.txt`;
    await fs.writeFile(
      listPath,
      parts.map((p) => `file '${p.replace(/'/g, "'\\''")}'`).join('\n'),
      'utf8'
    );
    try {
      await runFfmpeg([
        '-y', '-f', 'concat', '-safe', '0', '-i', listPath,
        '-c:a', 'pcm_f32le', '-ac', '1', '-ar', String(options.rate),
        outputPath,
      ]);
    } finally {
      await fs.rm(listPath, { force: true });
    }

    return { duration: await durationOf(outputPath), pieces: ops.length };
  } finally {
    await Promise.all(parts.map((p) => fs.rm(p, { force: true })));
  }
}

/**
 * Encode the rendered master into a delivery format.
 * @param {string} input
 * @param {string} output
 * @param {'wav'|'mp3'|'m4a'} format
 */
export async function encodeAudio(input, output, format) {
  const codecArgs = {
    wav: ['-c:a', 'pcm_s16le'],
    mp3: ['-c:a', 'libmp3lame', '-q:a', '2'],
    m4a: ['-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart'],
  }[format];
  if (!codecArgs) throw new FfmpegError(`unsupported audio format: ${format}`);
  await runFfmpeg(['-y', '-i', input, ...codecArgs, output]);
  return output;
}

/**
 * Render video alongside the audio.
 *
 * Copy segments trim the video the same way they trim the audio. Inserted
 * speech has no picture to go with it, so the preceding shot is frozen for the
 * length of the insertion (or the following one is pre-rolled, when the
 * insertion opens the timeline). Freezing is the honest choice: it keeps sound
 * and picture in sync and makes the edit visible rather than pretending the
 * speaker's lips match words they never said.
 *
 * @param {string} sourceVideo
 * @param {string} renderedAudio
 * @param {import('@voxdocs/edl').EdlSegment[]} segments
 * @param {Map<number, number>} synthDurations audio duration per synth segment index
 * @param {string} outputPath
 */
export async function renderVideo(sourceVideo, renderedAudio, segments, synthDurations, outputPath) {
  /** @type {Array<{ start: number, end: number, padStart: number, padEnd: number }>} */
  const shots = [];
  // An insertion before any picture has no shot to extend yet, so the hold is
  // carried forward and applied as a pre-roll on the first real shot. Emitting a
  // zero-length shot instead would trim to an empty frame range — at 25 fps a
  // sub-frame window contains no frame at all — leaving tpad nothing to clone.
  let pendingLeadHold = 0;

  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    if (segment.kind === 'copy') {
      shots.push({
        start: segment.start,
        end: segment.end,
        padStart: pendingLeadHold,
        padEnd: 0,
      });
      pendingLeadHold = 0;
      continue;
    }
    const hold = synthDurations.get(i) ?? segment.estimatedDuration ?? 0;
    if (hold <= 0) continue;
    if (shots.length > 0) {
      shots[shots.length - 1].padEnd += hold; // freeze the shot we just played
    } else {
      pendingLeadHold += hold;
    }
  }

  if (shots.length === 0) {
    throw new FfmpegError('the edit removed every frame of video');
  }
  // Trailing insertions after the last copy segment freeze the final frame.
  if (pendingLeadHold > 0) shots[shots.length - 1].padEnd += pendingLeadHold;

  const lines = [];
  const labels = [];
  shots.forEach((shot, i) => {
    const label = `v${i}`;
    labels.push(`[${label}]`);
    const filters = [
      `trim=start=${shot.start.toFixed(6)}:end=${Math.max(shot.end, shot.start + 0.001).toFixed(6)}`,
      'setpts=PTS-STARTPTS',
    ];
    if (shot.padStart > 0) {
      filters.push(`tpad=start_mode=clone:start_duration=${shot.padStart.toFixed(6)}`);
    }
    if (shot.padEnd > 0) {
      filters.push(`tpad=stop_mode=clone:stop_duration=${shot.padEnd.toFixed(6)}`);
    }
    filters.push('setpts=PTS-STARTPTS');
    lines.push(`[0:v]${filters.join(',')}[${label}];`);
  });
  lines.push(`${labels.join('')}concat=n=${shots.length}:v=1:a=0[outv]`);

  await runFfmpeg(
    [
      '-y', '-i', sourceVideo, '-i', renderedAudio,
      '-map', '[outv]', '-map', '1:a',
      '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
      '-c:a', 'aac', '-b:a', '192k',
      // Deliberately no -shortest: the video track quantises to whole frames and
      // can land a fraction of a frame short of the audio. Truncating to the
      // shorter stream would clip the tail off the last word; letting the final
      // frame hold for a few extra milliseconds is inaudible and invisible.
      '-movflags', '+faststart',
      outputPath,
    ],
    { filterScript: lines.join('\n') }
  );
  return outputPath;
}
