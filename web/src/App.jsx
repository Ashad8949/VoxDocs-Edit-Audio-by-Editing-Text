import { useEffect, useState } from 'react';
import Editor from './components/Editor.jsx';
import Library from './components/Library.jsx';

/**
 * Routing is the hash, so a project survives a refresh and can be linked to,
 * without pulling in a router for two screens.
 */
function projectFromHash() {
  const match = /^#\/p\/([A-Za-z0-9_-]+)$/.exec(window.location.hash);
  return match ? match[1] : null;
}

export default function App() {
  const [projectId, setProjectId] = useState(projectFromHash);

  useEffect(() => {
    const onChange = () => setProjectId(projectFromHash());
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);

  const open = (id) => {
    window.location.hash = `#/p/${id}`;
    setProjectId(id);
  };

  const back = () => {
    window.location.hash = '';
    setProjectId(null);
  };

  return projectId
    ? <Editor key={projectId} projectId={projectId} onBack={back} />
    : <Library onOpen={open} />;
}
