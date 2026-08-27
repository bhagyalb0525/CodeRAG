import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import MetricsBar from './components/MetricsBar';
import RepoIndexer from './components/RepoIndexer';
import ChatInterface from './components/ChatInterface';

const API_BASE_URL = 'http://localhost:8000';

export default function App() {
  const [backendStatus, setBackendStatus] = useState(null);
  const [activeRepo, setActiveRepo] = useState('');
  const [filesCount, setFilesCount] = useState(0);
  const [chunksCount, setChunksCount] = useState(0);

  // Check API health on mount and periodically
  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setBackendStatus(data);
      } else {
        setBackendStatus({ status: 'offline' });
      }
    } catch {
      setBackendStatus({ status: 'offline' });
    }
  };

  useEffect(() => {
    checkHealth();
    const timer = setInterval(checkHealth, 10000);
    return () => clearInterval(timer);
  }, []);

  const handleIndexSuccess = (summary) => {
    setActiveRepo(summary.repo_url);
    setFilesCount(summary.files_processed);
    setChunksCount(summary.chunks_created);
  };

  return (
    <div className="app-container">
      <Header 
        backendStatus={backendStatus} 
        activeRepo={activeRepo}
        filesCount={filesCount}
        chunksCount={chunksCount}
      />

      <MetricsBar 
        filesCount={filesCount} 
        chunksCount={chunksCount} 
        activeRepo={activeRepo}
        backendStatus={backendStatus}
      />

      <RepoIndexer 
        API_BASE_URL={API_BASE_URL} 
        onIndexSuccess={handleIndexSuccess}
        currentRepo={activeRepo}
      />

      <ChatInterface 
        API_BASE_URL={API_BASE_URL} 
        activeRepo={activeRepo}
        chunksCount={chunksCount}
      />
    </div>
  );
}
