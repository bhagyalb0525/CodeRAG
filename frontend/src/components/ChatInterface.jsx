import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { MessageSquare, Send, Loader2, Sparkles, HelpCircle, AlertCircle } from 'lucide-react';
import SourceInspector from './SourceInspector';

const PRESET_QUESTIONS = [
  "Explain the core architecture and main modules.",
  "Where are API routes or request handlers defined?",
  "How is vector similarity search implemented?",
  "What external dependencies and libraries are used?"
];

export default function ChatInterface({ API_BASE_URL, activeRepo, chunksCount }) {
  const [question, setQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [history, setHistory] = useState([]);
  const [errorMsg, setErrorMsg] = useState(null);

  const canAsk = activeRepo && chunksCount > 0;

  const handleAsk = async (e, customQ = null) => {
    if (e) e.preventDefault();
    const queryText = (customQ || question).trim();

    if (!queryText) return;
    if (!canAsk) {
      setErrorMsg('Please index a repository first before asking questions.');
      return;
    }

    setIsAsking(true);
    setErrorMsg(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_url: activeRepo,
          question: queryText,
          top_k: 5
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Failed to process question');
      }

      setHistory((prev) => [
        {
          id: Date.now(),
          question: data.question,
          answer: data.answer,
          sources: data.sources || []
        },
        ...prev
      ]);

      if (!customQ) {
        setQuestion('');
      }
    } catch (err) {
      setErrorMsg(err.message || 'Error generating answer');
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <section className="glass-card">
      <div className="card-title">
        <MessageSquare size={22} style={{ color: 'var(--accent-indigo)' }} />
        <span>2. Ask Natural Language Questions</span>
      </div>

      {!canAsk ? (
        <div className="alert-banner info" style={{ marginTop: '0.5rem' }}>
          <HelpCircle size={18} />
          <span>👈 Please index a public GitHub repository above to unlock natural language Q&A.</span>
        </div>
      ) : (
        <>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '0.75rem' }}>
            Ask questions about <code style={{ color: '#38bdf8' }}>{activeRepo}</code>. Gemini will answer using retrieved vector code chunks as strict context.
          </p>

          {/* Quick Preset Buttons */}
          <div className="sample-pills">
            {PRESET_QUESTIONS.map((q, idx) => (
              <button
                key={idx}
                type="button"
                className="pill-btn"
                onClick={() => {
                  setQuestion(q);
                  handleAsk(null, q);
                }}
                disabled={isAsking}
              >
                ✨ {q}
              </button>
            ))}
          </div>

          <form onSubmit={(e) => handleAsk(e)} style={{ display: 'flex', gap: '0.75rem', marginTop: '0.5rem' }}>
            <div className="input-group">
              <Sparkles className="input-icon" size={18} style={{ color: 'var(--accent-indigo)' }} />
              <input
                type="text"
                className="text-input"
                placeholder="e.g. How is routing handled? Where is authentication defined?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                disabled={isAsking}
              />
            </div>

            <button type="submit" className="btn-primary" disabled={isAsking || !question.trim()}>
              {isAsking ? (
                <Loader2 size={18} className="spinning" />
              ) : (
                <>
                  <span>Ask</span>
                  <Send size={16} />
                </>
              )}
            </button>
          </form>

          {errorMsg && (
            <div className="alert-banner error">
              <AlertCircle size={18} />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* History Feed */}
          {history.length > 0 && (
            <div style={{ marginTop: '2rem' }}>
              <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: '#f8fafc' }}>
                💬 Conversation History
              </h3>

              {history.map((item) => (
                <div key={item.id} className="qa-card">
                  <div className="qa-question">
                    <HelpCircle size={20} style={{ color: 'var(--accent-cyan)' }} />
                    <span>{item.question}</span>
                  </div>

                  <div className="qa-answer">
                    <ReactMarkdown>{item.answer}</ReactMarkdown>
                  </div>

                  <SourceInspector sources={item.sources} />
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}
