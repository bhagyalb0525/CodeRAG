import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Code2, Copy, Check, FileCode } from 'lucide-react';

export default function SourceInspector({ sources }) {
  const [isOpen, setIsOpen] = useState(false);
  const [copiedIdx, setCopiedIdx] = useState(null);

  if (!sources || sources.length === 0) return null;

  const handleCopy = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  return (
    <div className="sources-container">
      <button 
        type="button" 
        className="sources-toggle"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FileCode size={16} style={{ color: 'var(--accent-cyan)' }} />
          <span>Retrieved Context ({sources.length} Code Chunks)</span>
        </span>
        {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      {isOpen && (
        <div style={{ marginTop: '0.75rem' }}>
          {sources.map((src, idx) => (
            <div key={idx} className="source-item">
              <div className="source-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Code2 size={14} style={{ color: 'var(--accent-indigo)' }} />
                  <span className="source-file">{src.file_path}</span>
                  <span style={{ color: 'var(--text-muted)' }}>
                    (Lines {src.start_line} - {src.end_line})
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span className="sim-score">
                    Score: {(src.similarity * 100).toFixed(1)}%
                  </span>

                  <button
                    type="button"
                    onClick={() => handleCopy(src.chunk_text, idx)}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: 'var(--text-secondary)',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.25rem',
                      fontSize: '0.75rem'
                    }}
                    title="Copy Chunk Code"
                  >
                    {copiedIdx === idx ? (
                      <Check size={14} style={{ color: 'var(--accent-emerald)' }} />
                    ) : (
                      <Copy size={14} />
                    )}
                  </button>
                </div>
              </div>

              <pre className="source-code">
                <code>{src.chunk_text}</code>
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
