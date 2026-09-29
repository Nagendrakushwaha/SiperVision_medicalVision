import React, { useState, useEffect } from 'react';
import { Clock, Layers, Award, CheckCircle, XCircle, Search } from 'lucide-react';
import StatusBadge from '../components/StatusBadge';
import { listExperiments } from '../services/api';

export default function Experiments() {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedExp, setSelectedExp] = useState(null);

  useEffect(() => {
    listExperiments()
      .then(res => {
        setExperiments(res.experiments || []);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  }, []);

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Experiment Tracking & Lineage</h1>
          <p className="page-subtitle">
            Auditable log of all local training runs, hyperparameters, and checkpoint paths
          </p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: selectedExp ? '1fr 380px' : '1fr', gap: '1.5rem', alignItems: 'start' }}>
        {/* Experiments Table */}
        <div className="glass-card" style={{ padding: '0', overflow: 'hidden' }}>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Experiment ID</th>
                  <th>Model</th>
                  <th>Epochs</th>
                  <th>Batch / LR</th>
                  <th>Best Val Metric</th>
                  <th>Duration</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan="7" style={{ textAlign: 'center', padding: '3rem' }}>
                      Loading experiment history...
                    </td>
                  </tr>
                ) : experiments.length === 0 ? (
                  <tr>
                    <td colSpan="7" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                      No training experiments logged yet. Train a model in the Training Lab to record experiment runs.
                    </td>
                  </tr>
                ) : (
                  experiments.map((exp, idx) => (
                    <tr
                      key={idx}
                      onClick={() => setSelectedExp(exp)}
                      style={{ cursor: 'pointer', background: selectedExp === exp ? 'rgba(6, 182, 212, 0.08)' : undefined }}
                    >
                      <td className="mono" style={{ fontWeight: 700, color: 'var(--accent-cyan)' }}>
                        {exp.exp_id}
                      </td>
                      <td>
                        <strong style={{ color: 'var(--text-primary)' }}>{exp.model_name}</strong>
                      </td>
                      <td className="mono">
                        {exp.epochs_completed} / {exp.epochs_requested}
                      </td>
                      <td className="mono" style={{ fontSize: '0.8rem' }}>
                        B{exp.batch_size} • {exp.learning_rate}
                      </td>
                      <td className="mono" style={{ color: '#10B981', fontWeight: 600 }}>
                        {exp.best_val_loss ? exp.best_val_loss.toFixed(4) : '—'}
                      </td>
                      <td className="mono">{exp.duration}s</td>
                      <td>
                        <StatusBadge status={exp.status} />
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Experiment Detail Inspector */}
        {selectedExp && (
          <div className="glass-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 className="mono" style={{ color: 'var(--accent-cyan)' }}>{selectedExp.exp_id}</h3>
              <button
                onClick={() => setSelectedExp(null)}
                className="btn btn-secondary"
                style={{ padding: '0.2rem 0.5rem', fontSize: '0.8rem' }}
              >
                &times;
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', fontSize: '0.85rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Architecture:</span>
                <strong>{selectedExp.model_name}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Status:</span>
                <StatusBadge status={selectedExp.status} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Device:</span>
                <span className="mono">{selectedExp.device}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Requested / Completed:</span>
                <span className="mono">{selectedExp.epochs_requested} / {selectedExp.epochs_completed} epochs</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Batch Size:</span>
                <span className="mono">{selectedExp.batch_size}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Learning Rate:</span>
                <span className="mono">{selectedExp.learning_rate}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Input Resolution:</span>
                <span className="mono">{selectedExp.image_size}×{selectedExp.image_size}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Best Val Loss:</span>
                <span className="mono" style={{ color: '#10B981', fontWeight: 600 }}>{selectedExp.best_val_loss}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Training Duration:</span>
                <span className="mono">{selectedExp.duration} seconds</span>
              </div>
              <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '0.65rem' }}>
                <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '0.2rem' }}>Checkpoint Path:</span>
                <code className="mono" style={{ fontSize: '0.75rem', wordBreak: 'break-all', color: 'var(--text-secondary)' }}>
                  {selectedExp.checkpoint_path}
                </code>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
