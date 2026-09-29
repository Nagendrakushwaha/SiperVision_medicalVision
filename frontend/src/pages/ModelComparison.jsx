import React, { useState, useEffect } from 'react';
import { GitCompare, Layers, Award, Zap, Cpu, CheckCircle, XCircle } from 'lucide-react';
import PlotlyChart from '../components/PlotlyChart';
import StatusBadge from '../components/StatusBadge';
import { listModels } from '../services/api';

export default function ModelComparison() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listModels()
      .then(res => {
        setModels(res.models || []);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  }, []);

  const evaluatedModels = models.filter(m => m.status === 'Evaluated' && m.metrics);

  // Comparison Bar Chart (Accuracy & F1)
  const perfData = evaluatedModels.length > 0 ? [
    {
      x: evaluatedModels.map(m => m.name),
      y: evaluatedModels.map(m => (m.metrics.accuracy * 100).toFixed(1)),
      name: 'Accuracy (%)',
      type: 'bar',
      marker: { color: '#06B6D4' }
    },
    {
      x: evaluatedModels.map(m => m.name),
      y: evaluatedModels.map(m => (m.metrics.f1_score * 100).toFixed(1)),
      name: 'F1 Score (%)',
      type: 'bar',
      marker: { color: '#10B981' }
    },
    {
      x: evaluatedModels.map(m => m.name),
      y: evaluatedModels.map(m => (m.metrics.roc_auc * 100).toFixed(1)),
      name: 'ROC-AUC (%)',
      type: 'bar',
      marker: { color: '#6366F1' }
    }
  ] : [];

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Model Architecture Comparison</h1>
          <p className="page-subtitle">
            Side-by-side comparative evaluation of deep architectures on RSNA Pneumonia cohort
          </p>
        </div>
      </div>

      {/* Comparison Master Table */}
      <div className="glass-card" style={{ marginBottom: '2rem', padding: '0', overflow: 'hidden' }}>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Status</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>ROC-AUC</th>
                <th>Specificity</th>
                <th>Parameters</th>
                <th>Size</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const met = m.metrics;
                return (
                  <tr key={m.id}>
                    <td>
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <strong style={{ color: 'var(--text-primary)' }}>{m.name}</strong>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{m.description}</span>
                      </div>
                    </td>
                    <td>
                      <StatusBadge status={m.status} />
                    </td>
                    <td className="mono">
                      {met ? `${(met.accuracy * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="mono">
                      {met ? `${(met.precision * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="mono">
                      {met ? `${(met.recall * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="mono" style={{ color: met ? '#38BDF8' : 'inherit', fontWeight: met ? 600 : 400 }}>
                      {met ? `${(met.f1_score * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="mono" style={{ color: met ? 'var(--accent-cyan)' : 'inherit', fontWeight: met ? 600 : 400 }}>
                      {met ? met.roc_auc.toFixed(3) : '—'}
                    </td>
                    <td className="mono">
                      {met ? `${(met.specificity * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="mono" style={{ fontSize: '0.8rem' }}>
                      {m.total_parameters ? `${(m.total_parameters / 1e6).toFixed(1)}M` : '—'}
                    </td>
                    <td className="mono" style={{ fontSize: '0.8rem' }}>
                      {m.size_mb ? `${m.size_mb} MB` : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Comparison Graph */}
      {evaluatedModels.length > 0 ? (
        <div className="glass-card">
          <h3 style={{ marginBottom: '1rem' }}>Comparative Metrics Bar Chart</h3>
          <PlotlyChart
            data={perfData}
            layout={{
              barmode: 'group',
              yaxis: { title: 'Score (%)', range: [0, 100], gridcolor: 'rgba(255,255,255,0.06)' },
              height: 320,
              legend: { orientation: 'h', y: -0.2 }
            }}
          />
        </div>
      ) : (
        <div className="glass-card empty-state">
          <GitCompare size={42} className="empty-state-icon" color="var(--text-muted)" />
          <h3 className="empty-state-title">No Models Evaluated Yet</h3>
          <p className="empty-state-desc">
            Train ResNet18 or other architectures and run evaluation to see real comparative charts.
          </p>
        </div>
      )}
    </div>
  );
}
