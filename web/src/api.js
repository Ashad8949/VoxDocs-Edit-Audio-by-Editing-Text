/** Thin API client. Every call returns parsed JSON or throws an ApiError. */

export class ApiError extends Error {
  constructor(message, status, code) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new ApiError('Cannot reach the VoxDocs server.', 0, 'offline');
  }
  if (response.status === 204) return null;

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new ApiError(
      body.message || body.error || `Request failed (${response.status})`,
      response.status,
      body.error
    );
  }
  return body;
}

export const listProjects = () => request('/api/projects').then((b) => b.projects);

export const getProject = (id) => request(`/api/projects/${id}`).then((b) => b.project);

export const deleteProject = (id) => request(`/api/projects/${id}`, { method: 'DELETE' });

export const health = () => request('/api/health');

/**
 * Upload with progress. XHR rather than fetch, because fetch still cannot
 * report upload progress and these files are large.
 */
export function uploadProject(file, { name, onProgress } = {}) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append('file', file);
    if (name) form.append('name', name);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/projects');
    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
    });
    xhr.addEventListener('load', () => {
      let body = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* fall through to the status check */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body.project);
      else reject(new ApiError(body.message || `Upload failed (${xhr.status})`, xhr.status, body.error));
    });
    xhr.addEventListener('error', () => reject(new ApiError('Upload failed.', 0, 'network')));
    xhr.addEventListener('abort', () => reject(new ApiError('Upload cancelled.', 0, 'aborted')));
    xhr.send(form);
  });
}

export const getEnvelope = (id, points = 2000) =>
  request(`/api/projects/${id}/envelope?points=${points}`);

const json = (body) => ({
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
});

/** Cost/duration preview for an edit, without rendering it. */
export const planEdit = (id, tokens) => request(`/api/projects/${id}/plan`, json({ tokens }));

/**
 * Queue a render. The backend answers 202 immediately and does the work on a
 * Celery worker, so this returns the pending render plus where to poll it.
 */
export const queueRender = (id, tokens, { format = 'wav', video = false, quality = 'standard' } = {}) =>
  request(`/api/projects/${id}/render`, json({ tokens, format, video, quality }));

export const getRender = (projectId, renderId) =>
  request(`/api/projects/${projectId}/renders/${renderId}/status`).then((b) => b.render);

/**
 * Queue a render and wait for the worker to finish it.
 *
 * Rendering an hour of audio takes minutes, which is exactly why it is not done
 * inside the request. Polling keeps the client honest about that instead of
 * holding a connection open and hoping no proxy times it out.
 *
 * @param {(render: any) => void} [onProgress] called on each status change
 * @param {{ signal?: AbortSignal, intervalMs?: number }} [options]
 */
export async function renderEdit(projectId, tokens, options = {}, onProgress) {
  const { format, video, quality, signal, intervalMs = 1000 } = options;
  const queued = await queueRender(projectId, tokens, { format, video, quality });
  let render = queued.render;
  onProgress?.(render);

  while (render.status === 'pending' || render.status === 'rendering') {
    if (signal?.aborted) throw new ApiError('Render cancelled.', 0, 'aborted');
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
    const next = await getRender(projectId, render.id);
    if (next.status !== render.status) onProgress?.(next);
    render = next;
  }

  if (render.status === 'failed') {
    throw new ApiError(render.error || 'The render failed.', 500, 'render_failed');
  }
  return render;
}

/**
 * Voice-quality tiers (subscription levels). `value` is the string the API
 * expects; `note` explains the trade-off. Pro/Studio need a trained voice
 * model (not built yet) — until then they transparently behave like Standard.
 */
export const QUALITY_TIERS = [
  { value: 'free', label: 'Free — robotic voice', note: 'Fast, generic formant voice (eSpeak).' },
  { value: 'standard', label: 'Standard — voice clone', note: "Clones the speaker's voice (XTTS)." },
  { value: 'pro', label: 'Pro — enhanced clone', note: 'Closer timbre match (RVC). Coming soon.' },
  { value: 'studio', label: 'Studio — trained voice', note: 'Best match, per-speaker model. Coming soon.' },
];

export const mediaUrl = (id) => `/api/projects/${id}/media`;

export const downloadUrl = (projectId, renderId) =>
  `/api/projects/${projectId}/renders/${renderId}`;
