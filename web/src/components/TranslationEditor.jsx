import { useEffect, useMemo, useRef, useState } from 'react';
import * as api from '../api.js';
import Waveform from './Waveform.jsx';

/**
 * TranslationEditor: click a sentence in the translated transcript to jump
 * the video there, edit its wording inline, delete it outright, or keep the
 * original-language audio for it instead. Mirrors the click-to-navigate,
 * edit-in-place feel of the main Editor, but at segment granularity — a
 * translation doesn't have real per-word timestamps the way a transcript of
 * actual speech does, so editing a specific word just means editing the text.
 */
export default function TranslationEditor({ projectId, project, translation, onEditsApplied }) {
  const [segments, setSegments] = useState([]);
  const [edits, setEdits] = useState([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [loadError, setLoadError] = useState(null);

  const [envelope, setEnvelope] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [editingIndex, setEditingIndex] = useState(null);

  const mediaRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/projects/${projectId}/translations/${translation.id}`)
      .then((r) => r.json())
      .then((data) => {
        if (cancelled) return;
        setSegments(data.translation.segments || []);
        setEdits(data.translation.edits || []);
      })
      .catch(() => !cancelled && setLoadError('Failed to load translation'));
    return () => {
      cancelled = true;
    };
  }, [projectId, translation.id]);

  useEffect(() => {
    let cancelled = false;
    api
      .getEnvelope(projectId, 2400)
      .then((e) => !cancelled && setEnvelope(e))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  // ------------------------------------------------------------ edit state

  const editByIndex = useMemo(() => {
    const map = new Map();
    for (const e of edits) if (e.type === 'segment') map.set(e.index, e);
    return map;
  }, [edits]);

  const replaceEdit = (index, patch) => {
    setEdits((prev) => {
      const next = prev.filter((e) => !(e.type === 'segment' && e.index === index));
      if (patch) next.push({ type: 'segment', index, ...patch });
      return next;
    });
  };

  const isDeleted = (segment) => Boolean(editByIndex.get(segment.index)?.deleted);
  const isKeepOriginal = (segment) => Boolean(editByIndex.get(segment.index)?.keepOriginal);
  const isEdited = (segment) => {
    const edit = editByIndex.get(segment.index);
    return Boolean(
      edit && !edit.deleted && !edit.keepOriginal && edit.text !== undefined && edit.text !== segment.translatedText
    );
  };

  const displayText = (segment) => {
    const edit = editByIndex.get(segment.index);
    if (!edit || edit.deleted) return segment.translatedText;
    return edit.text ?? segment.translatedText;
  };

  const commitEdit = (segment, rawText) => {
    const text = rawText.trim();
    if (!text || text === segment.translatedText) {
      replaceEdit(segment.index, null); // no real change; clear any prior edit
    } else {
      replaceEdit(segment.index, { text });
    }
    setEditingIndex(null);
  };

  const toggleDelete = (segment) => {
    replaceEdit(segment.index, isDeleted(segment) ? null : { deleted: true });
  };

  const toggleKeepOriginal = (segment) => {
    replaceEdit(segment.index, isKeepOriginal(segment) ? null : { keepOriginal: true, text: segment.originalText });
  };

  // ------------------------------------------------------------- playback

  const togglePlay = () => {
    const media = mediaRef.current;
    if (!media) return;
    if (media.paused) media.play().catch(() => {});
    else media.pause();
  };

  const seek = (time) => {
    const media = mediaRef.current;
    if (media) media.currentTime = Math.max(0, time);
  };

  const activeIndex = useMemo(() => {
    const hit = segments.find((s) => currentTime >= s.start && currentTime < s.end);
    return hit ? hit.index : null;
  }, [segments, currentTime]);

  const duration = project?.duration ?? 0;

  // -------------------------------------------------------------- saving

  const handleSaveEdits = async () => {
    setSaving(true);
    setError(null);
    try {
      const response = await fetch(`/api/projects/${projectId}/translations/${translation.id}/edit`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ edits }),
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.message || 'Failed to save edits');
      }
      onEditsApplied?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const editedCount = segments.filter((s) => isEdited(s)).length;
  const deletedCount = segments.filter((s) => isDeleted(s)).length;
  const keptOriginalCount = segments.filter((s) => isKeepOriginal(s)).length;

  return (
    <div className="translation-editor">
      <h3>Edit Translation</h3>
      <p className="subtitle">
        Click a sentence to jump there. Hover for edit, delete, or keep-original-voice actions.
      </p>

      {loadError && <div className="error">{loadError}</div>}

      {project?.hasVideo && (
        <div className="preview">
          <video
            ref={mediaRef}
            className="preview-video"
            src={api.mediaUrl(projectId)}
            onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            controls={false}
          />
          <div className="transport">
            <button type="button" className="play" onClick={togglePlay} aria-label={playing ? 'Pause' : 'Play'}>
              {playing ? '❚❚' : '▶'}
            </button>
            <Waveform envelope={envelope} duration={duration} currentTime={currentTime} cuts={[]} onSeek={seek} />
          </div>
        </div>
      )}

      {segments.length === 0 ? (
        <p className="empty">Loading transcript...</p>
      ) : (
        <div className="transcript-flow">
          {segments.map((segment) => {
            const deleted = isDeleted(segment);
            const keptOriginal = isKeepOriginal(segment);
            const edited = isEdited(segment);
            const editing = editingIndex === segment.index;
            const active = activeIndex === segment.index;

            return (
              <span
                key={segment.index}
                className={[
                  'chunk',
                  deleted && 'deleted',
                  keptOriginal && 'kept-original',
                  edited && 'edited',
                  active && 'active',
                  editing && 'editing',
                ]
                  .filter(Boolean)
                  .join(' ')}
              >
                {editing ? (
                  <input
                    autoFocus
                    className="inline-input"
                    defaultValue={displayText(segment)}
                    onBlur={(e) => commitEdit(segment, e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        e.currentTarget.blur();
                      }
                      if (e.key === 'Escape') setEditingIndex(null);
                    }}
                  />
                ) : (
                  <span className="chunk-text" onClick={() => seek(segment.start)} title={`${segment.start.toFixed(1)}s`}>
                    {deleted ? segment.translatedText : displayText(segment)}
                  </span>
                )}

                <span className="chunk-actions">
                  <button
                    type="button"
                    className="icon-btn"
                    title="Edit wording"
                    onClick={() => setEditingIndex(segment.index)}
                    disabled={deleted}
                  >
                    ✎
                  </button>
                  <button
                    type="button"
                    className="icon-btn"
                    title={deleted ? 'Restore' : 'Delete sentence'}
                    onClick={() => toggleDelete(segment)}
                  >
                    {deleted ? '↺' : '×'}
                  </button>
                  <button
                    type="button"
                    className="icon-btn"
                    title={keptOriginal ? 'Use translation' : 'Keep original voice'}
                    onClick={() => toggleKeepOriginal(segment)}
                    disabled={deleted}
                  >
                    {keptOriginal ? '🌐' : '🎙'}
                  </button>
                </span>
              </span>
            );
          })}
        </div>
      )}

      {error && <div className="error">{error}</div>}

      <div className="editor-actions">
        <div className="summary">
          {editedCount > 0 && <span>{editedCount} edited</span>}
          {deletedCount > 0 && <span>{deletedCount} deleted</span>}
          {keptOriginalCount > 0 && <span>{keptOriginalCount} kept original</span>}
        </div>
        <button onClick={handleSaveEdits} disabled={saving || segments.length === 0} className="btn-primary">
          {saving ? 'Saving...' : 'Save Edits & Render'}
        </button>
      </div>

      <style jsx>{`
        .translation-editor {
          padding: 20px;
          background: #f9f9f9;
          border-radius: 8px;
        }

        .translation-editor h3 {
          margin-top: 0;
          font-size: 1.2rem;
        }

        .subtitle {
          color: #666;
          font-size: 0.9rem;
          margin-bottom: 16px;
        }

        .preview {
          margin-bottom: 20px;
          background: #111;
          border-radius: 8px;
          overflow: hidden;
        }

        .preview-video {
          width: 100%;
          max-height: 360px;
          display: block;
          background: #000;
        }

        .transport {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 8px 12px;
          background: #1a1a1a;
        }

        .play {
          width: 32px;
          height: 32px;
          border-radius: 50%;
          border: none;
          background: #2196f3;
          color: white;
          cursor: pointer;
          flex: none;
          font-size: 0.85rem;
        }

        .transport .waveform {
          flex: 1;
          height: 40px;
          cursor: pointer;
        }

        .transcript-flow {
          background: white;
          border: 1px solid #ddd;
          border-radius: 6px;
          padding: 16px;
          margin-bottom: 20px;
          max-height: 420px;
          overflow-y: auto;
          line-height: 2.4;
        }

        .chunk {
          position: relative;
          display: inline-flex;
          align-items: center;
          gap: 2px;
          margin: 2px 4px 2px 0;
          padding: 2px 4px;
          border-radius: 4px;
          border-bottom: 2px solid transparent;
        }

        .chunk.active {
          background: #fff3cd;
        }

        .chunk.edited {
          border-bottom-color: #2196f3;
        }

        .chunk.kept-original {
          border-bottom-color: #4caf50;
          background: #f0f8ff;
        }

        .chunk.deleted .chunk-text {
          text-decoration: line-through;
          color: #aaa;
        }

        .chunk-text {
          cursor: pointer;
        }

        .chunk-text:hover {
          text-decoration: underline;
        }

        .inline-input {
          font: inherit;
          padding: 2px 4px;
          border: 1px solid #2196f3;
          border-radius: 3px;
          min-width: 120px;
        }

        .chunk-actions {
          display: inline-flex;
          gap: 2px;
          opacity: 0;
          transition: opacity 0.15s;
        }

        .chunk:hover .chunk-actions,
        .chunk.editing .chunk-actions {
          opacity: 1;
        }

        .icon-btn {
          border: none;
          background: #eee;
          border-radius: 3px;
          width: 20px;
          height: 20px;
          font-size: 0.75rem;
          line-height: 1;
          cursor: pointer;
          padding: 0;
        }

        .icon-btn:hover:not(:disabled) {
          background: #ddd;
        }

        .icon-btn:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }

        .editor-actions {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
        }

        .summary {
          display: flex;
          gap: 12px;
          font-size: 0.85rem;
          color: #666;
        }

        .btn-primary {
          padding: 10px 24px;
          background: #2196f3;
          color: white;
          border: none;
          border-radius: 4px;
          font-size: 1rem;
          cursor: pointer;
          transition: background 0.2s;
        }

        .btn-primary:hover:not(:disabled) {
          background: #1976d2;
        }

        .btn-primary:disabled {
          background: #ccc;
          cursor: not-allowed;
        }

        .error {
          padding: 12px;
          background: #ffebee;
          border: 1px solid #ef5350;
          border-radius: 4px;
          color: #c62828;
          margin-bottom: 16px;
        }

        .empty {
          text-align: center;
          color: #999;
          padding: 40px 20px;
        }
      `}</style>
    </div>
  );
}
