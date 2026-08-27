import React from 'react';
import { Terminal, Cpu, Database, CheckCircle2, AlertCircle } from 'lucide-react';

export default function Header({ backendStatus, activeRepo, filesCount, chunksCount }) {
  const isOnline = backendStatus?.status === 'online';

  return (
    <header className="header-bar">
      <div className="brand">
        <div className="brand-icon">
          <Terminal size={24} />
        </div>
        <div>
          <h1 className="brand-title">CodeRAG AI Engine</h1>
          <p className="brand-subtitle">Natural Language Ingestion & Codebase Q&A</p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
        {activeRepo && (
          <div className="status-badge" style={{ borderColor: 'rgba(99, 102, 241, 0.4)', background: 'rgba(99, 102, 241, 0.1)' }}>
            <span style={{ color: '#a5f3fc', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
              📁 {activeRepo}
            </span>
          </div>
        )}

        <div className="status-badge">
          <span className={`status-dot ${isOnline ? 'online' : 'offline'}`}></span>
          <span>{isOnline ? 'API Connected' : 'API Disconnected'}</span>
        </div>
      </div>
    </header>
  );
}
