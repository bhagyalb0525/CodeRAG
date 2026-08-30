import React from 'react';
import { FileText, Database, Cpu, Sparkles, CheckCircle2 } from 'lucide-react';

export default function MetricsBar({ filesCount, chunksCount, activeRepo, backendStatus }) {
  return (
    <div className="metrics-grid">
      <div className="metric-card">
        <div className="metric-icon" style={{ color: 'var(--accent-cyan)' }}>
          <FileText size={22} />
        </div>
        <div>
          <div className="metric-val">{filesCount || 0}</div>
          <div className="metric-lbl">Files Processed</div>
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-icon" style={{ color: 'var(--accent-indigo)' }}>
          <Database size={22} />
        </div>
        <div>
          <div className="metric-val">{chunksCount || 0}</div>
          <div className="metric-lbl">Vector Chunks</div>
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-icon" style={{ color: 'var(--accent-violet)' }}>
          <Cpu size={22} />
        </div>
        <div>
          <div className="metric-val" style={{ fontSize: '1rem', color: '#a5f3fc' }}>
            gemini-embedding-001
          </div>
          <div className="metric-lbl">Embedding Engine (384d)</div>
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-icon" style={{ color: 'var(--accent-emerald)' }}>
          <Sparkles size={22} />
        </div>
        <div>
          <div className="metric-val" style={{ fontSize: '1rem', color: '#6ee7b7' }}>
            Google Gemini
          </div>
          <div className="metric-lbl">LLM RAG Synthesizer</div>
        </div>
      </div>
    </div>
  );
}
