import { useEffect, useState } from 'react';
import * as api from '../api.js';

/**
 * TranslationSelector: UI for selecting target language and viewing translation status.
 */
export default function TranslationSelector({ projectId, project, onTranslationReady }) {
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [translations, setTranslations] = useState([]);
  const [currentTranslation, setCurrentTranslation] = useState(null);

  const languages = [
    { code: 'en', label: 'English' },
    { code: 'hi', label: 'Hindi (हिंदी)' },
    { code: 'hinglish', label: 'Hinglish (Mix)' },
  ];

  // Load existing translations
  useEffect(() => {
    const loadTranslations = async () => {
      try {
        const response = await fetch(`/api/projects/${projectId}/translations`);
        if (response.ok) {
          const data = await response.json();
          setTranslations(data.translations || []);
        }
      } catch (err) {
        console.error('Failed to load translations:', err);
      }
    };

    loadTranslations();
  }, [projectId]);

  // Poll for translation status
  useEffect(() => {
    if (!currentTranslation || currentTranslation.status === 'ready' || currentTranslation.status === 'failed') {
      return;
    }

    const timer = setInterval(async () => {
      try {
        const response = await fetch(`/api/projects/${projectId}/translations/${currentTranslation.id}`);
        if (response.ok) {
          const data = await response.json();
          setCurrentTranslation(data.translation);
          if (data.translation.status === 'ready') {
            onTranslationReady?.(data.translation);
          }
        }
      } catch (err) {
        console.error('Failed to poll translation status:', err);
      }
    }, 2000);

    return () => clearInterval(timer);
  }, [currentTranslation, projectId, onTranslationReady]);

  const handleCreateTranslation = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/projects/${projectId}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ targetLanguage: selectedLanguage }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.message || 'Failed to create translation');
      }

      const data = await response.json();
      const newTranslation = data.translation;
      setCurrentTranslation(newTranslation);
      setTranslations((prev) => [...prev, newTranslation]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const existingTranslation = translations.find((t) => t.targetLanguage === selectedLanguage);

  return (
    <div className="translation-selector">
      <h3>Translate & Dub</h3>

      {project?.hasVideo ? (
        <div className="translation-controls">
          <div className="language-select">
            <label htmlFor="target-language">Target Language:</label>
            <select
              id="target-language"
              value={selectedLanguage}
              onChange={(e) => {
                setSelectedLanguage(e.target.value);
                const existing = translations.find((t) => t.targetLanguage === e.target.value);
                setCurrentTranslation(existing || null);
              }}
              disabled={loading}
            >
              {languages.map((lang) => (
                <option key={lang.code} value={lang.code}>
                  {lang.label}
                </option>
              ))}
            </select>
          </div>

          {existingTranslation && existingTranslation.status === 'ready' ? (
            <div className="translation-status ready">
              <span>✓ Translation ready</span>
              <button onClick={() => onTranslationReady?.(existingTranslation)} className="btn-small">
                Edit &amp; Dub
              </button>
            </div>
          ) : currentTranslation ? (
            <div className="translation-status">
              <span>Status: {currentTranslation.status}</span>
              {currentTranslation.status === 'failed' && (
                <div className="error">{currentTranslation.error}</div>
              )}
            </div>
          ) : (
            <button onClick={handleCreateTranslation} disabled={loading} className="btn-primary">
              {loading ? 'Translating...' : 'Create Translation'}
            </button>
          )}

          {error && <div className="error">{error}</div>}
        </div>
      ) : (
        <div className="info">This project has no video to dub. Upload a video file to enable dubbing.</div>
      )}
    </div>
  );
}
