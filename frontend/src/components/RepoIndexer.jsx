import React, { useState } from 'react';
import { GitBranch, Loader2, ArrowRight, CheckCircle2, AlertTriangle, Layers } from 'lucide-react';

export default function RepoIndexer({ API_BASE_URL, onIndexSuccess, currentRepo }) {
  const [repoUrl, setRepoUrl] = useState(currentRepo || '');
  const [isIndexing, setIsIndexing] = useState(false);
  const [message, setMessage] = useState(null); // { type: 'success' | 'error' | 'info', text: string }

  const handleIndex = async (e) => {
    e.preventDefault();
    const cleanUrl = repoUrl.trim();
    
    if (!cleanUrl) {
      setMessage({ type: 'error', text: 'Please enter a valid GitHub repository URL.' });
      return;
    }
    if (!cleanUrl.startsWith('https://github.com/') && !cleanUrl.startsWith('http://github.com/')) {
      setMessage({ type: 'error', text: 'URL must start with https://github.com/' });
      return;
    }

    setIsIndexing(true);
    setMessage({ type: 'info', text: 'Cloning repository, scanning source files, and computing embeddings...' });

    try {
      const res = await fetch(`${API_BASE_URL}/api/index`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: cleanUrl }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Failed to index repository');
      }

      setMessage({ type: 'success', text: data.message });
      if (onIndexSuccess) {
        onIndexSuccess({
          repo_url: data.repo_url,
          files_processed: data.files_processed,
          chunks_created: data.chunks_created
        });
      }
    } catch (err) {
      setMessage({ type: 'error', text: err.message || 'Error communicating with indexing backend.' });
    } finally {
      setIsIndexing(false);
    }
  };

  return (
    <section className="glass-card">
      <div className="card-title">
        <GitBranch size={22} style={{ color: 'var(--accent-cyan)' }} />
        <span>1. Index a GitHub Repository</span>
      </div>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1.25rem' }}>
        Provide any public GitHub repository link. CodeRAG will clone the repo, split `.py, .js, .ts, .java, .c, .cpp` files into 50-line chunks, generate vector embeddings, and store them in PostgreSQL.
      </p>

      <form onSubmit={handleIndex} className="indexer-form">
        <div className="input-group">
          <GitBranch className="input-icon" size={18} />
          <input
            type="url"
            className="text-input"
            placeholder="https://github.com/owner/repository"
            value={repoUrl}
            onChange={(e) => setRepoUrl(e.target.value)}
            disabled={isIndexing}
          />
        </div>

        <button type="submit" className="btn-primary" disabled={isIndexing}>
          {isIndexing ? (
            <>
              <Loader2 size={18} className="spinning" />
              <span>Indexing Repo...</span>
            </>
          ) : (
            <>
              <span>Index Repository</span>
              <ArrowRight size={18} />
            </>
          )}
        </button>
      </form>

      {message && (
        <div className={`alert-banner ${message.type}`}>
          {message.type === 'success' && <CheckCircle2 size={18} />}
          {message.type === 'error' && <AlertTriangle size={18} />}
          {message.type === 'info' && <Loader2 size={18} className="spinning" />}
          <span>{message.text}</span>
        </div>
      )}
    </section>
  );
}
