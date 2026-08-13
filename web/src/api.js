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

export const renderEdit = (id, tokens, { format = 'wav', video = false } = {}) =>
  request(`/api/projects/${id}/render`, json({ tokens, format, video }));

export const mediaUrl = (id) => `/api/projects/${id}/media`;

export const downloadUrl = (projectId, renderId) =>
  `/api/projects/${projectId}/renders/${renderId}`;
