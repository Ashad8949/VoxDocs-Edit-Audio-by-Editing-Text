import { useEffect, useState } from 'react';

/**
 * DubRenderPanel: Queue and monitor dubbed video rendering.
 */
export default function DubRenderPanel({ projectId, translation }) {
  const [format, setFormat] = useState('mp4');
  const [dubRenders, setDubRenders] = useState([]);
  const [currentDub, setCurrentDub] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const formats = [
    { value: 'mp4', label: 'MP4 (H.264)' },
    { value: 'webm', label: 'WebM (VP9)' },
    { value: 'mkv', label: 'Matroska (MKV)' },
  ];

  // Load existing dub renders
  useEffect(() => {
    const loadDubRenders = async () => {
      try {
        // Note: This endpoint doesn't exist yet; would need to be added to the API
        // For now, we'll skip loading and just allow creating new ones
      } catch (err) {
        console.error('Failed to load dub renders:', err);
      }
    };

    loadDubRenders();
  }, [projectId, translation.id]);

  // Poll for dub render status
  useEffect(() => {
    if (!currentDub || currentDub.status === 'ready' || currentDub.status === 'failed') {
      return;
    }

    const timer = setInterval(async () => {
      try {
        const response = await fetch(
          `/api/projects/${projectId}/translations/${translation.id}/dubs/${currentDub.id}/status`
        );
        if (response.ok) {
          const data = await response.json();
          setCurrentDub(data.dubRender);
        }
      } catch (err) {
        console.error('Failed to poll dub render status:', err);
      }
    }, 2000);

    return () => clearInterval(timer);
  }, [currentDub, projectId, translation.id]);

  const handleQueueRender = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `/api/projects/${projectId}/translations/${translation.id}/dub`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ format }),
        }
      );

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.message || 'Failed to queue dub render');
      }

      const data = await response.json();
      const newDub = data.dubRender;
      setCurrentDub(newDub);
      setDubRenders((prev) => [...prev, newDub]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'pending':
      case 'rendering':
        return '#ff9800';
      case 'ready':
        return '#4caf50';
      case 'failed':
        return '#f44336';
      default:
        return '#999';
    }
  };

  const getStatusLabel = (status) => {
    switch (status) {
      case 'pending':
        return 'Queued';
      case 'rendering':
        return 'Rendering...';
      case 'ready':
        return 'Ready for download';
      case 'failed':
        return 'Failed';
      default:
        return status;
    }
  };

  return (
    <div className="dub-render-panel">
      <h3>Render Dubbed Video</h3>

      {!currentDub ? (
        <div className="render-setup">
          <div className="format-select">
            <label htmlFor="output-format">Output Format:</label>
            <select
              id="output-format"
              value={format}
              onChange={(e) => setFormat(e.target.value)}
              disabled={loading}
            >
              {formats.map((fmt) => (
                <option key={fmt.value} value={fmt.value}>
                  {fmt.label}
                </option>
              ))}
            </select>
          </div>

          <button onClick={handleQueueRender} disabled={loading} className="btn-primary">
            {loading ? 'Queuing...' : 'Start Render'}
          </button>

          <p className="info">
            Rendering will combine your edited audio with the original video. This may take several minutes.
          </p>
        </div>
      ) : (
        <div className="render-status">
          <div className="status-indicator" style={{ backgroundColor: getStatusColor(currentDub.status) }} />
          <div className="status-info">
            <div className="status-label">{getStatusLabel(currentDub.status)}</div>
            <div className="status-details">
              {currentDub.duration > 0 && <span>Duration: {currentDub.duration.toFixed(1)}s</span>}
              {currentDub.bytes > 0 && (
                <span>Size: {(currentDub.bytes / 1024 / 1024).toFixed(1)} MB</span>
              )}
            </div>

            {currentDub.status === 'ready' && currentDub.downloadUrl && (
              <a href={currentDub.downloadUrl} download className="btn-download">
                Download {currentDub.format.toUpperCase()}
              </a>
            )}

            {currentDub.status === 'failed' && currentDub.error && (
              <div className="error">{currentDub.error}</div>
            )}

            {(currentDub.status === 'pending' || currentDub.status === 'rendering') && (
              <button onClick={() => setCurrentDub(null)} className="btn-secondary">
                Queue Another Render
              </button>
            )}
          </div>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {dubRenders.length > 1 && (
        <div className="previous-renders">
          <h4>Previous Renders</h4>
          <ul>
            {dubRenders.map((dub) => (
              <li key={dub.id}>
                <span className="format">{dub.format.toUpperCase()}</span>
                <span className="status">{getStatusLabel(dub.status)}</span>
                {dub.status === 'ready' && dub.downloadUrl && (
                  <a href={dub.downloadUrl} download className="link">
                    Download
                  </a>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      <style jsx>{`
        .dub-render-panel {
          padding: 20px;
          background: #f9f9f9;
          border-radius: 8px;
        }

        .dub-render-panel h3 {
          margin-top: 0;
          font-size: 1.2rem;
        }

        .render-setup {
          background: white;
          padding: 16px;
          border-radius: 6px;
          border: 1px solid #ddd;
        }

        .format-select {
          margin-bottom: 16px;
        }

        .format-select label {
          display: block;
          margin-bottom: 6px;
          font-weight: 600;
          font-size: 0.9rem;
        }

        .format-select select {
          width: 100%;
          padding: 8px;
          border: 1px solid #ccc;
          border-radius: 4px;
          font-size: 1rem;
        }

        .render-status {
          background: white;
          padding: 16px;
          border-radius: 6px;
          border: 1px solid #ddd;
          display: flex;
          gap: 16px;
          align-items: center;
        }

        .status-indicator {
          width: 16px;
          height: 16px;
          border-radius: 50%;
          flex-shrink: 0;
        }

        .status-info {
          flex: 1;
        }

        .status-label {
          font-weight: 600;
          font-size: 1rem;
          margin-bottom: 4px;
        }

        .status-details {
          display: flex;
          gap: 16px;
          font-size: 0.85rem;
          color: #666;
          margin-bottom: 12px;
        }

        .btn-download {
          display: inline-block;
          padding: 8px 16px;
          background: #4caf50;
          color: white;
          text-decoration: none;
          border-radius: 4px;
          font-size: 0.9rem;
          transition: background 0.2s;
        }

        .btn-download:hover {
          background: #45a049;
        }

        .btn-secondary {
          padding: 8px 16px;
          background: #9e9e9e;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 0.9rem;
        }

        .btn-secondary:hover {
          background: #757575;
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
          width: 100%;
        }

        .btn-primary:hover:not(:disabled) {
          background: #1976d2;
        }

        .btn-primary:disabled {
          background: #ccc;
          cursor: not-allowed;
        }

        .info {
          color: #666;
          font-size: 0.85rem;
          margin-top: 12px;
        }

        .error {
          padding: 12px;
          background: #ffebee;
          border: 1px solid #ef5350;
          border-radius: 4px;
          color: #c62828;
          margin-top: 16px;
        }

        .previous-renders {
          margin-top: 20px;
          padding-top: 20px;
          border-top: 1px solid #ddd;
        }

        .previous-renders h4 {
          margin: 0 0 12px 0;
          font-size: 0.95rem;
          color: #333;
        }

        .previous-renders ul {
          list-style: none;
          padding: 0;
          margin: 0;
        }

        .previous-renders li {
          padding: 8px;
          background: white;
          border-radius: 4px;
          margin-bottom: 8px;
          display: flex;
          gap: 12px;
          align-items: center;
          font-size: 0.9rem;
        }

        .format {
          font-weight: 600;
          min-width: 50px;
        }

        .status {
          color: #666;
          flex: 1;
        }

        .link {
          color: #2196f3;
          text-decoration: none;
          font-weight: 600;
        }

        .link:hover {
          text-decoration: underline;
        }
      `}</style>
    </div>
  );
}
