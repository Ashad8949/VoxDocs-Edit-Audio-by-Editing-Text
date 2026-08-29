import { useEffect, useRef, useState } from 'react';
import * as api from '../api.js';

// Status values that mean a training run is still in flight.
const BUSY = new Set(['pending', 'uploading', 'training', 'pulling', 'evaluating']);
const STAGE_LABEL = {
  pending: 'Queued…',
  uploading: 'Uploading audio…',
  training: 'Training on GPU…',
  pulling: 'Fetching model…',
  evaluating: 'Evaluating…',
};

/**
 * VoiceModelPanel: train and monitor the project's Pro-tier voice model.
 *
 * Surfaces the whole MLOps lifecycle to the user — trigger training, watch the
 * status walk its stages, and see the speaker-similarity result (Standard vs
 * Pro) once a model is ready.
 */
export default function VoiceModelPanel({ projectId }) {
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  const load = async () => {
    try {
      const m = await api.getVoiceModel(projectId);
      setModel(m);
      return m;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    return () => clearInterval(pollRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  // Poll while a run is in flight.
  useEffect(() => {
    clearInterval(pollRef.current);
    if (model && BUSY.has(model.status)) {
      pollRef.current = setInterval(load, 4000);
    }
    return () => clearInterval(pollRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.status]);

  const train = async () => {
    setError(null);
    try {
      setModel(await api.trainVoiceModel(projectId));
    } catch (err) {
      setError(err.message);
    }
  };

  const busy = model && BUSY.has(model.status);
  const metrics = model?.metrics || {};
  const std = metrics.standard_similarity;
  const pro = metrics.pro_similarity;

  return (
    <div className="voice-model-panel">
      <div className="vm-head">
        <strong>Pro voice model</strong>
        {model?.status === 'ready' && <span className="vm-badge ready">active</span>}
        {model?.status === 'rejected' && <span className="vm-badge warn">below bar</span>}
        {model?.status === 'failed' && <span className="vm-badge err">failed</span>}
      </div>

      {loading ? (
        <p className="muted small">Loading…</p>
      ) : busy ? (
        <p className="muted small">{STAGE_LABEL[model.status] || model.status} (GPU training runs on Kaggle; you can leave this page).</p>
      ) : model?.status === 'ready' ? (
        <p className="muted small">
          Trained model serving the Pro tier.
          {pro != null && (
            <> Speaker match <b>{(pro * 100).toFixed(0)}%</b>
              {std != null && <> vs <b>{(std * 100).toFixed(0)}%</b> zero-shot</>}.</>
          )}
        </p>
      ) : (
        <p className="muted small">
          Train a per-speaker model on the speaker's own voice for the closest
          possible match. Uses your Kaggle GPU; takes a few minutes.
          {model?.status === 'rejected' && ' Last run scored below the quality bar — try more/cleaner audio.'}
          {model?.status === 'failed' && model?.error && ` Last run failed: ${model.error.slice(0, 140)}`}
        </p>
      )}

      {!busy && (
        <button className="primary" onClick={train} disabled={busy}>
          {model && model.status !== 'ready' ? 'Retrain' : model ? 'Retrain' : 'Train voice model'}
        </button>
      )}

      {error && <div className="vm-error">{error}</div>}

      <style jsx>{`
        .voice-model-panel { display: flex; flex-direction: column; gap: 8px; }
        .vm-head { display: flex; align-items: center; gap: 8px; }
        .vm-badge { font-size: 11px; padding: 1px 8px; border-radius: 20px; }
        .vm-badge.ready { background: #1f7a3d; color: #d9ffe6; }
        .vm-badge.warn { background: #7a5a1f; color: #fff2d9; }
        .vm-badge.err { background: #7a1f28; color: #ffd9dd; }
        .vm-error { color: #ffb4b4; font-size: 12px; }
      `}</style>
    </div>
  );
}
