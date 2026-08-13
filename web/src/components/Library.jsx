import { useCallback, useEffect, useRef, useState } from 'react';
import * as api from '../api.js';

/** Project list plus the drop target that creates new ones. */
export default function Library({ onOpen }) {
  const [projects, setProjects] = useState([]);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState(null);
  const inputRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      setProjects(await api.listProjects());
    } catch (listError) {
      setError(listError.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    api.health().then(setStatus).catch(() => {});
    // Anything still importing will change status without us asking.
    const timer = setInterval(refresh, 4000);
    return () => clearInterval(timer);
  }, [refresh]);

  const upload = async (file) => {
    if (!file) return;
    setError(null);
    setProgress(0);
    try {
      const project = await api.uploadProject(file, {
        name: file.name,
        onProgress: setProgress,
      });
      setProgress(null);
      onOpen(project.id);
    } catch (uploadError) {
      setProgress(null);
      setError(uploadError.message);
    }
  };

  const asrReady = status?.model?.asr?.available !== false;

  return (
    <div className="screen">
      <header className="hero">
        <h1>VoxDocs</h1>
        <p>
          Edit audio by editing text. Delete a word from the transcript and it disappears
          from the recording; type a new one and it is spoken in the same voice.
        </p>
      </header>

      {status && !asrReady && (
        <div className="alert">
          <strong>The transcription model is not available.</strong>
          <p>{status.model?.asr?.error ?? status.modelError}</p>
        </div>
      )}

      <div
        className={`dropzone ${dragOver ? 'is-over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          upload(e.dataTransfer.files?.[0]);
        }}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          hidden
          accept="audio/*,video/*"
          onChange={(e) => upload(e.target.files?.[0])}
        />
        {progress === null ? (
          <>
            <strong>Drop an audio or video file here</strong>
            <span className="muted">or click to choose — mp3, wav, m4a, mp4, mov and more</span>
          </>
        ) : (
          <>
            <strong>Uploading… {Math.round(progress * 100)}%</strong>
            <div className="progress"><div style={{ width: `${progress * 100}%` }} /></div>
          </>
        )}
      </div>

      {error && <div className="alert">{error}</div>}

      <h2>Your projects</h2>
      {projects.length === 0 && <p className="muted">Nothing here yet.</p>}

      <ul className="projects">
        {projects.map((project) => (
          <li key={project.id} className="project">
            <button className="project-open" onClick={() => onOpen(project.id)}>
              <strong>{project.name}</strong>
              <span className="muted small">
                {statusLabel(project)}
                {project.wordCount > 0 && ` · ${project.wordCount} words`}
                {project.hasVideo && ' · video'}
              </span>
            </button>
            <button
              className="danger"
              title="Delete project"
              onClick={async () => {
                await api.deleteProject(project.id).catch((e) => setError(e.message));
                refresh();
              }}
            >
              ✕
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function statusLabel(project) {
  if (project.status === 'failed') return `failed — ${project.error ?? 'unknown error'}`;
  if (project.status !== 'ready') return 'transcribing…';
  const minutes = Math.floor((project.duration ?? 0) / 60);
  const seconds = Math.floor((project.duration ?? 0) % 60);
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}
