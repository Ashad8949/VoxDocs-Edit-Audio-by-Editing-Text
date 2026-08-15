import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as api from '../api.js';
import {
  deleteRange,
  deletedSpans,
  docFromWords,
  docText,
  insertAt,
  localStats,
  replaceRange,
  restoreRange,
  toTokens,
  updateInsert,
  wordIndexAtTime,
} from '../lib/doc.js';
import DubStudio from './DubStudio.jsx';
import Transcript from './Transcript.jsx';
import Waveform from './Waveform.jsx';

const POLL_MS = 2000;

export default function Editor({ projectId, onBack }) {
  const [project, setProject] = useState(null);
  const [envelope, setEnvelope] = useState(null);
  const [error, setError] = useState(null);
  const [view, setView] = useState('edit'); // 'edit' | 'dub'

  const [doc, setDoc] = useState([]);
  const [past, setPast] = useState([]);
  const [future, setFuture] = useState([]);

  const [selection, setSelection] = useState(null);
  const [caret, setCaret] = useState(null);
  const [draft, setDraft] = useState(null);
  const [showDeleted, setShowDeleted] = useState(true);

  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);

  const [plan, setPlan] = useState(null);
  const [rendering, setRendering] = useState(false);
  const [renderStage, setRenderStage] = useState(null);
  const [result, setResult] = useState(null);
  const [format, setFormat] = useState('wav');

  // Renders completed this session, so the preview can play the edited
  // result directly instead of only offering it as a download; 'original'
  // means the source media, otherwise it's a render id.
  const [renders, setRenders] = useState([]);
  const [previewSource, setPreviewSource] = useState('original');

  const mediaRef = useRef(null);
  const editedAudioRef = useRef(null);
  const surfaceRef = useRef(null);

  // ------------------------------------------------------------ loading

  useEffect(() => {
    let cancelled = false;
    let timer;

    const poll = async () => {
      try {
        const loaded = await api.getProject(projectId);
        if (cancelled) return;
        setProject(loaded);
        if (loaded.status === 'ready') {
          setDoc(docFromWords(loaded.transcript.words, loaded.transcript.segments));
          api.getEnvelope(projectId, 2400).then((e) => !cancelled && setEnvelope(e)).catch(() => {});
        } else if (loaded.status !== 'failed') {
          timer = setTimeout(poll, POLL_MS);
        }
      } catch (loadError) {
        if (!cancelled) setError(loadError.message);
      }
    };

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [projectId]);

  // ------------------------------------------------------------- history

  const apply = useCallback(
    (next) => {
      if (next === doc) return;
      setPast((p) => [...p.slice(-99), doc]);
      setFuture([]);
      setDoc(next);
      setResult(null);
    },
    [doc]
  );

  const undo = useCallback(() => {
    setPast((p) => {
      if (p.length === 0) return p;
      const previous = p[p.length - 1];
      setFuture((f) => [doc, ...f]);
      setDoc(previous);
      setResult(null);
      return p.slice(0, -1);
    });
  }, [doc]);

  const redo = useCallback(() => {
    setFuture((f) => {
      if (f.length === 0) return f;
      setPast((p) => [...p, doc]);
      setDoc(f[0]);
      setResult(null);
      return f.slice(1);
    });
  }, [doc]);

  // -------------------------------------------------------------- edits

  const selectionRange = selection
    ? [Math.min(selection.anchor, selection.focus), Math.max(selection.anchor, selection.focus)]
    : null;

  const deleteSelection = useCallback(() => {
    if (!selectionRange) return;
    apply(deleteRange(doc, selectionRange[0], selectionRange[1]));
  }, [apply, doc, selectionRange]);

  const restoreSelection = useCallback(() => {
    if (!selectionRange) return;
    apply(restoreRange(doc, selectionRange[0], selectionRange[1]));
  }, [apply, doc, selectionRange]);

  const commitDraft = useCallback(
    (accept) => {
      const pending = draft;
      setDraft(null);
      // The inline input held focus while it was open; hand the keyboard back
      // to the surface so shortcuts like undo keep working straight afterwards.
      requestAnimationFrame(() => surfaceRef.current?.focus());
      if (!pending || !accept || !pending.text.trim()) return;
      if (pending.replacing) {
        apply(replaceRange(doc, pending.replacing[0], pending.replacing[1], pending.text));
        setSelection(null);
      } else {
        apply(insertAt(doc, pending.position, pending.text));
      }
      setCaret(null);
    },
    [apply, doc, draft]
  );

  // Keyboard: the editor behaves like a document, so the usual keys must work.
  useEffect(() => {
    const surface = surfaceRef.current;
    if (!surface) return;

    const onKeyDown = (event) => {
      if (draft) return; // the inline input owns the keyboard while it is open

      const meta = event.metaKey || event.ctrlKey;
      if (meta && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        if (event.shiftKey) redo();
        else undo();
        return;
      }
      if (meta) return;

      if (event.key === 'Backspace' || event.key === 'Delete') {
        event.preventDefault();
        deleteSelection();
        return;
      }
      if (event.key === 'Escape') {
        setSelection(null);
        setCaret(null);
        return;
      }
      if (event.key === ' ' && !selectionRange) {
        event.preventDefault();
        togglePlay();
        return;
      }
      if (event.key === 'Enter') {
        event.preventDefault();
        const position = caret ?? (selectionRange ? selectionRange[1] + 1 : doc.length);
        setDraft({ position, text: '', replacing: selectionRange });
        return;
      }

      // A printable character starts an insertion, replacing any selection —
      // the same thing typing does in any text editor.
      if (event.key.length === 1 && !event.altKey) {
        event.preventDefault();
        const position = caret ?? (selectionRange ? selectionRange[0] : doc.length);
        setDraft({ position, text: event.key, replacing: selectionRange });
      }
    };

    surface.addEventListener('keydown', onKeyDown);
    return () => surface.removeEventListener('keydown', onKeyDown);
  });

  // ----------------------------------------------------------- playback

  // When previewing an edited render that wasn't exported as a video (no
  // re-muxed picture to go with it), the original video keeps playing muted
  // for a picture while a second, hidden <audio> carries the edited sound —
  // that way hearing an edit never has to wait on a video re-encode.
  const activeRender = renders.find((r) => r.id === previewSource) || null;
  const previewingEdited = previewSource !== 'original' && activeRender !== null;
  const dualTrack = Boolean(project?.hasVideo && previewingEdited && activeRender.format !== 'mp4');

  const togglePlay = useCallback(() => {
    const media = mediaRef.current;
    if (!media) return;
    const audio = dualTrack ? editedAudioRef.current : null;
    if (media.paused) {
      media.play().catch(() => {});
      audio?.play().catch(() => {});
    } else {
      media.pause();
      audio?.pause();
    }
  }, [dualTrack]);

  // Original and edited versions rarely share a timeline once words are cut
  // or inserted, so switching between them starts over rather than trying to
  // preserve a position that would land somewhere unrelated.
  const switchPreview = useCallback((source) => {
    mediaRef.current?.pause();
    editedAudioRef.current?.pause();
    setPreviewSource(source);
    setCurrentTime(0);
  }, []);

  const seek = useCallback((time) => {
    const clamped = Math.max(0, time);
    const media = mediaRef.current;
    if (media) media.currentTime = clamped;
    if (dualTrack && editedAudioRef.current) editedAudioRef.current.currentTime = clamped;
  }, [dualTrack]);

  const activeIndex = useMemo(() => wordIndexAtTime(doc, currentTime), [doc, currentTime]);
  const cuts = useMemo(() => deletedSpans(doc), [doc]);

  // ---------------------------------------------------------- the plan

  const tokens = useMemo(() => toTokens(doc), [doc]);
  const stats = useMemo(() => localStats(doc), [doc]);

  useEffect(() => {
    if (!project || project.status !== 'ready') return;
    let cancelled = false;
    // Debounced: the estimate follows typing, it does not race it.
    const timer = setTimeout(() => {
      api
        .planEdit(projectId, tokens)
        .then((response) => !cancelled && setPlan(response.stats))
        .catch(() => {});
    }, 350);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [projectId, tokens, project]);

  const doRender = async () => {
    setRendering(true);
    setRenderStage('queued');
    setError(null);
    try {
      const render = await api.renderEdit(
        projectId,
        tokens,
        { format, video: Boolean(project.hasVideo) && format === 'video' },
        (update) => setRenderStage(update.status)
      );
      setResult(render);
      setRenders((prev) => [render, ...prev]);
      setPreviewSource(render.id);
      setProject(await api.getProject(projectId));
    } catch (renderError) {
      setError(renderError.message);
    } finally {
      setRendering(false);
      setRenderStage(null);
    }
  };

  // ------------------------------------------------------------- render

  if (error && !project) {
    return (
      <div className="screen">
        <button className="link" onClick={onBack}>← All projects</button>
        <div className="alert">{error}</div>
      </div>
    );
  }

  if (!project) return <div className="screen"><div className="muted">Loading…</div></div>;

  if (project.status === 'failed') {
    return (
      <div className="screen">
        <button className="link" onClick={onBack}>← All projects</button>
        <h1>{project.name}</h1>
        <div className="alert">
          <strong>This file could not be imported.</strong>
          <p>{project.error}</p>
        </div>
      </div>
    );
  }

  if (project.status !== 'ready') {
    return (
      <div className="screen">
        <button className="link" onClick={onBack}>← All projects</button>
        <h1>{project.name}</h1>
        <div className="working">
          <div className="spinner" />
          <div>
            <strong>Transcribing…</strong>
            <p className="muted">
              Finding every word and exactly when it was said. Longer recordings take longer;
              this page updates itself.
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (view === 'dub') {
    return (
      <DubStudio
        projectId={projectId}
        project={project}
        onBack={() => setView('edit')}
      />
    );
  }

  const duration = previewingEdited ? activeRender.duration : (project.duration ?? 0);
  const estimated = plan?.estimatedDuration ?? stats.keptSeconds;

  // The video element shows the edited render directly only when that
  // render is itself an mp4; otherwise it shows the original picture (muted,
  // when dualTrack) while the separate <audio> below carries the real sound.
  const mediaSrc = previewingEdited && !dualTrack
    ? api.downloadUrl(projectId, activeRender.id)
    : api.mediaUrl(projectId);

  return (
    <div className="editor">
      <header className="bar">
        <button className="link" onClick={onBack}>← All projects</button>
        <h1 title={project.name}>{project.name}</h1>
        <div className="grow" />
        {project.hasVideo && (
          <button className="link" onClick={() => setView('dub')}>Translate &amp; Dub →</button>
        )}
        <span className="muted small">
          {project.transcript.backend} · {project.transcript.language}
        </span>
      </header>

      <section className="stage">
        {project.hasVideo ? (
          <video
            ref={mediaRef}
            className="video"
            src={mediaSrc}
            muted={dualTrack}
            // In dualTrack mode the hidden <audio> below is the real clock —
            // it carries the edited timeline, which runs a different length
            // than the original video once words are cut or inserted.
            onTimeUpdate={dualTrack ? undefined : (e) => setCurrentTime(e.currentTarget.currentTime)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            controls={false}
          />
        ) : (
          <audio
            ref={mediaRef}
            src={mediaSrc}
            onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
          />
        )}

        {dualTrack && (
          <audio
            ref={editedAudioRef}
            src={api.downloadUrl(projectId, activeRender.id)}
            onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
            style={{ display: 'none' }}
          />
        )}

        <div className="transport">
          <button className="play" onClick={togglePlay} aria-label={playing ? 'Pause' : 'Play'}>
            {playing ? '❚❚' : '▶'}
          </button>
          <Waveform
            envelope={envelope}
            duration={duration}
            currentTime={currentTime}
            cuts={cuts}
            onSeek={seek}
          />
          <span className="time">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        </div>

        {renders.length > 0 && (
          <div className="preview-source">
            <button
              className={previewSource === 'original' ? 'active' : ''}
              onClick={() => switchPreview('original')}
            >
              Original
            </button>
            <button
              className={previewSource !== 'original' ? 'active' : ''}
              onClick={() => switchPreview(renders[0].id)}
            >
              Edited
            </button>
          </div>
        )}
      </section>

      <section className="toolbar">
        <button onClick={deleteSelection} disabled={!selectionRange}>
          Cut {selectionRange ? `${selectionRange[1] - selectionRange[0] + 1} word(s)` : 'selection'}
        </button>
        <button onClick={restoreSelection} disabled={!selectionRange}>Restore</button>
        <button onClick={undo} disabled={past.length === 0}>Undo</button>
        <button onClick={redo} disabled={future.length === 0}>Redo</button>
        <label className="toggle">
          <input
            type="checkbox"
            checked={showDeleted}
            onChange={(e) => setShowDeleted(e.target.checked)}
          />
          Show cuts
        </label>

        <div className="grow" />

        <div className="stat">
          <b>{stats.kept}</b> kept
        </div>
        <div className="stat">
          <b>{stats.deleted}</b> cut
        </div>
        <div className="stat">
          <b>{stats.inserted}</b> added
        </div>
        <div className="stat">
          <b>{formatTime(estimated)}</b> result
          {duration > 0 && (
            <span className="muted small"> ({signed(estimated - duration)})</span>
          )}
        </div>
      </section>

      <section
        className="surface"
        ref={surfaceRef}
        tabIndex={0}
        onMouseDown={() => surfaceRef.current?.focus()}
      >
        <Transcript
          doc={doc}
          selection={selection}
          caret={caret}
          activeIndex={activeIndex}
          showDeleted={showDeleted}
          draft={draft}
          onSelect={setSelection}
          onCaret={(position) => {
            setCaret(position);
            // Placing a caret between words dismisses a word selection, but
            // clearing the caret must not clear the selection that just
            // replaced it — selecting a word does both, in that order.
            if (position !== null) setSelection(null);
          }}
          onSeek={seek}
          onDraftChange={(text) => setDraft((d) => (d ? { ...d, text } : d))}
          onCommitInsert={commitDraft}
          onUpdateInsert={(id, text) => apply(updateInsert(doc, id, text))}
        />
      </section>

      <footer className="export">
        <div className="hint">
          Click a word to select it, drag or shift-click for a phrase, then press
          <kbd>Delete</kbd>. Click between words and type to add new speech.
          <kbd>Space</kbd> plays, double-click a word to jump there.
        </div>
        <div className="grow" />

        <select value={format} onChange={(e) => setFormat(e.target.value)}>
          <option value="wav">WAV</option>
          <option value="mp3">MP3</option>
          <option value="m4a">M4A</option>
          {project.hasVideo && <option value="video">MP4 (video)</option>}
        </select>
        <button className="primary" onClick={doRender} disabled={rendering}>
          {rendering ? renderLabel(renderStage) : 'Export'}
        </button>
      </footer>

      {error && <div className="alert floating" onClick={() => setError(null)}>{error}</div>}

      {result && (
        <div className="result">
          <div>
            <strong>Ready — {formatTime(result.duration)}</strong>
            <span className="muted small">
              {' '}from {result.pieces} piece{result.pieces === 1 ? '' : 's'}
              {result.synthesis.words > 0 && (
                <>
                  {' · '}
                  {result.synthesis.fromVoiceBank}/{result.synthesis.words} added words taken
                  from your own voice
                </>
              )}
            </span>
            {result.warnings?.length > 0 && (
              <ul className="warnings">
                {result.warnings.map((warning, i) => <li key={i}>{warning}</li>)}
              </ul>
            )}
          </div>
          <div className="grow" />
          <a className="primary button" href={api.downloadUrl(projectId, result.id)} download>
            Download
          </a>
        </div>
      )}
    </div>
  );
}

/** What the export button says while a worker is busy. */
function renderLabel(stage) {
  if (stage === 'pending') return 'Queued…';
  if (stage === 'rendering') return 'Rendering…';
  return 'Working…';
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  const whole = Math.floor(seconds);
  const minutes = Math.floor(whole / 60);
  return `${minutes}:${String(whole % 60).padStart(2, '0')}`;
}

function signed(delta) {
  const sign = delta >= 0 ? '+' : '−';
  return `${sign}${formatTime(Math.abs(delta))}`;
}
