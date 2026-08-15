import { useState } from 'react';
import DubRenderPanel from './DubRenderPanel.jsx';
import TranslationEditor from './TranslationEditor.jsx';
import TranslationSelector from './TranslationSelector.jsx';

/**
 * DubStudio: Complete workflow for video translation and dubbing.
 * 
 * Workflow:
 * 1. Select target language → creates translation
 * 2. Wait for translation to complete
 * 3. Edit translated transcript
 * 4. Queue dub render (original video + edited audio)
 * 5. Download dubbed video
 */
export default function DubStudio({ projectId, project, onBack }) {
  const [stage, setStage] = useState('select'); // 'select' | 'edit' | 'render'
  const [translation, setTranslation] = useState(null);

  const handleTranslationReady = (trans) => {
    setTranslation(trans);
    setStage('edit');
  };

  const handleEditsApplied = () => {
    setStage('render');
  };

  return (
    <div className="dub-studio">
      <div className="dub-header">
        <button onClick={onBack} className="btn-back">
          ← Back
        </button>
        <h2>Dubbing Studio</h2>
        <div className="project-info">
          <span className="project-name">{project?.name || 'Untitled'}</span>
          <span className="language-badge">{project?.language?.toUpperCase() || 'AUTO'}</span>
        </div>
      </div>

      <div className="dub-content">
        {stage === 'select' && (
          <TranslationSelector projectId={projectId} project={project} onTranslationReady={handleTranslationReady} />
        )}

        {stage === 'edit' && translation && (
          <TranslationEditor
            projectId={projectId}
            project={project}
            translation={translation}
            onEditsApplied={handleEditsApplied}
          />
        )}

        {stage === 'render' && translation && (
          <DubRenderPanel projectId={projectId} translation={translation} />
        )}
      </div>

      <div className="dub-sidebar">
        <div className="workflow-status">
          <h4>Workflow</h4>
          <ol className="workflow-steps">
            <li className={stage === 'select' ? 'active' : stage !== 'select' ? 'completed' : ''}>
              Select Language
            </li>
            <li className={stage === 'edit' ? 'active' : stage === 'render' ? 'completed' : ''}>
              Edit Transcript
            </li>
            <li className={stage === 'render' ? 'active' : ''}>
              Render Video
            </li>
          </ol>
        </div>

        {translation && (
          <div className="translation-info">
            <h4>Translation Info</h4>
            <dl>
              <dt>Target Language:</dt>
              <dd>{translation.targetLanguage.toUpperCase()}</dd>
              <dt>Status:</dt>
              <dd className={`status-${translation.status}`}>{translation.status}</dd>
              <dt>Created:</dt>
              <dd>{new Date(translation.createdAt).toLocaleDateString()}</dd>
            </dl>
          </div>
        )}

        <div className="tips">
          <h4>Tips</h4>
          <ul>
            <li>Edit segments to mix languages (e.g., English + Hindi)</li>
            <li>Check "Keep original voice" to skip synthesis for that segment</li>
            <li>Rendering takes time; you can close and check back later</li>
            <li>Download the dubbed video in MP4, WebM, or MKV format</li>
          </ul>
        </div>
      </div>

      <style jsx>{`
        .dub-studio {
          display: flex;
          flex-direction: column;
          height: 100vh;
          background: #fafafa;
        }

        .dub-header {
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 16px 20px;
          background: white;
          border-bottom: 1px solid #ddd;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .btn-back {
          padding: 8px 12px;
          background: #f5f5f5;
          border: 1px solid #ddd;
          border-radius: 4px;
          cursor: pointer;
          font-size: 0.9rem;
          transition: background 0.2s;
        }

        .btn-back:hover {
          background: #e0e0e0;
        }

        .dub-header h2 {
          margin: 0;
          flex: 1;
          font-size: 1.5rem;
        }

        .project-info {
          display: flex;
          gap: 12px;
          align-items: center;
        }

        .project-name {
          font-size: 0.9rem;
          color: #666;
          max-width: 200px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .language-badge {
          padding: 4px 12px;
          background: #e3f2fd;
          color: #1976d2;
          border-radius: 20px;
          font-size: 0.8rem;
          font-weight: 600;
        }

        .dub-content {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
        }

        .dub-sidebar {
          width: 280px;
          background: white;
          border-left: 1px solid #ddd;
          padding: 20px;
          overflow-y: auto;
          position: absolute;
          right: 0;
          top: 0;
          height: 100vh;
          padding-top: 120px;
        }

        .workflow-status,
        .translation-info,
        .tips {
          margin-bottom: 24px;
        }

        .workflow-status h4,
        .translation-info h4,
        .tips h4 {
          margin: 0 0 12px 0;
          font-size: 0.95rem;
          color: #333;
          font-weight: 600;
        }

        .workflow-steps {
          list-style: none;
          padding: 0;
          margin: 0;
        }

        .workflow-steps li {
          padding: 8px 0;
          padding-left: 24px;
          position: relative;
          font-size: 0.9rem;
          color: #666;
        }

        .workflow-steps li::before {
          content: '';
          position: absolute;
          left: 0;
          top: 50%;
          transform: translateY(-50%);
          width: 12px;
          height: 12px;
          border-radius: 50%;
          border: 2px solid #ddd;
          background: white;
        }

        .workflow-steps li.active::before {
          background: #2196f3;
          border-color: #2196f3;
        }

        .workflow-steps li.completed::before {
          background: #4caf50;
          border-color: #4caf50;
        }

        .translation-info dl {
          margin: 0;
          font-size: 0.9rem;
        }

        .translation-info dt {
          font-weight: 600;
          color: #333;
          margin-top: 8px;
        }

        .translation-info dd {
          margin: 2px 0 0 0;
          color: #666;
        }

        .status-pending {
          color: #ff9800;
        }

        .status-translating {
          color: #ff9800;
        }

        .status-ready {
          color: #4caf50;
        }

        .status-failed {
          color: #f44336;
        }

        .tips ul {
          list-style: none;
          padding: 0;
          margin: 0;
          font-size: 0.85rem;
          color: #666;
        }

        .tips li {
          padding: 6px 0;
          line-height: 1.4;
        }

        .tips li::before {
          content: '• ';
          color: #2196f3;
          font-weight: bold;
          margin-right: 4px;
        }

        @media (max-width: 1200px) {
          .dub-sidebar {
            position: static;
            width: auto;
            height: auto;
            padding: 20px;
            padding-top: 20px;
            border-left: none;
            border-top: 1px solid #ddd;
            margin-top: 20px;
          }

          .dub-studio {
            flex-direction: column;
          }
        }
      `}</style>
    </div>
  );
}
