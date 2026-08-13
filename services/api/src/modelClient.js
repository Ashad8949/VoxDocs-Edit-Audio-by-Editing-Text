/**
 * Client for the model server.
 *
 * The API server owns the durable transcript; the model server only caches a
 * voice profile. That asymmetry is deliberate — it means a model pod can be
 * restarted or scaled out at any moment and the worst case is one re-seed,
 * which this client performs transparently on a 409.
 */

import fs from 'node:fs';
import { config } from './config.js';

export class ModelError extends Error {
  /** @param {string} message @param {number} status @param {string} [code] */
  constructor(message, status = 502, code = 'model_error') {
    super(message);
    this.name = 'ModelError';
    this.status = status;
    this.code = code;
  }
}

/** @param {Response} response */
async function readError(response) {
  let body;
  try {
    body = await response.json();
  } catch {
    body = {};
  }
  return new ModelError(
    body.message || body.error || `model server returned ${response.status}`,
    response.status === 503 ? 503 : 502,
    body.error || 'model_error'
  );
}

/** @param {string} pathname @param {RequestInit} init */
async function request(pathname, init = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.modelTimeoutMs);
  try {
    return await fetch(`${config.modelUrl}${pathname}`, { ...init, signal: controller.signal });
  } catch (error) {
    if (/** @type {any} */ (error)?.name === 'AbortError') {
      throw new ModelError('model server timed out', 504, 'model_timeout');
    }
    throw new ModelError(`cannot reach model server at ${config.modelUrl}`, 503, 'model_unreachable');
  } finally {
    clearTimeout(timer);
  }
}

export async function health() {
  const response = await request('/health');
  if (!response.ok) throw await readError(response);
  return response.json();
}

/**
 * Transcribe a media file, seeding the voice profile as a side effect.
 * @param {string} filePath
 * @param {string} projectId
 * @param {string | null} [language]
 */
export async function transcribe(filePath, projectId, language = null) {
  const form = new FormData();
  const bytes = await fs.promises.readFile(filePath);
  form.append('audio', new Blob([bytes]), filePath.split('/').pop() ?? 'audio');
  form.append('project_id', projectId);
  if (language) form.append('language', language);

  const response = await request('/transcribe', { method: 'POST', body: form });
  if (!response.ok) throw await readError(response);
  return response.json();
}

/**
 * Re-seed a voice profile the model server has evicted.
 * @param {string} projectId
 * @param {Array<{ text: string, start: number, end: number, confidence?: number }>} words
 * @param {number} duration
 * @param {string | null} [audioPath] supply only when a neural backend needs it
 */
export async function putVoiceProfile(projectId, words, duration, audioPath = null) {
  const form = new FormData();
  form.append('project_id', projectId);
  form.append('words', JSON.stringify(words));
  form.append('duration', String(duration));
  if (audioPath) {
    const bytes = await fs.promises.readFile(audioPath);
    form.append('audio', new Blob([bytes]), 'audio.wav');
  }

  const response = await request('/voice-profile', { method: 'POST', body: form });
  if (!response.ok) throw await readError(response);
  return response.json();
}

/**
 * Resolve every insertion in one round trip, re-seeding the profile once if the
 * model server has forgotten this project.
 *
 * @param {string} projectId
 * @param {Array<{ text: string, context_before?: string|null, context_after?: string|null,
 *                 lead_gap?: number, trail_gap?: number }>} items
 * @param {() => Promise<void>} reseed
 */
export async function synthesizeBatch(projectId, items, reseed) {
  if (items.length === 0) return { results: [] };

  const send = async () =>
    request('/synthesize/batch', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ project_id: projectId, items }),
    });

  let response = await send();
  if (response.status === 409) {
    await reseed();
    response = await send();
  }
  if (!response.ok) throw await readError(response);
  return response.json();
}
