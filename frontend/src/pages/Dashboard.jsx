import React, { useEffect, useState } from 'react';
import {
  Database,
  Layers,
  Activity,
  CheckCircle2,
  AlertTriangle,
  Clock,
  ArrowRight,
  TrendingUp,
  Cpu
} from 'lucide-react';
import { Link } from 'react-router-dom';
import MetricCard from '../components/MetricCard';
import PlotlyChart from '../components/PlotlyChart';
import StatusBadge from '../components/StatusBadge';
import { getDatasetSummary, listModels, getEvaluation, getTrainingHistory, getPredictionHistory, getSystemStatus } from '../services/api';

export default function Dashboard() {
  const [summary, setSummary] = useState(null);
  const [models, setModels] = useState([]);
  const [evaluation, setEvaluation] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [system, setSystem] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getDatasetSummary().catch(() => null),
      listModels().catch(() => ({ models: [] })),
      getEvaluation('resnet18').catch(() => null),
      getTrainingHistory('resnet18').catch(() => ({ history: [] })),
      getPredictionHistory().catch(() => ({ history: [] })),
      getSystemStatus().catch(() => null)
    ]).then(([sum, mods, ev, tr, pr, sys]) => {
      setSummary(sum);
      setModels(mods?.models || []);
      setEvaluation(ev?.data || null);
      setTrainingHistory(tr?.history || []);
      setPredictions(pr?.history || []);
      setSystem(sys);
      setLoading(false);
    });
  }, []);

  // Class Distribution Pie Chart Data
  const classDistData = summary ? [{
    values: [summary.negative_cases, summary.positive_cases],
    labels: ['Normal / No Opacity', 'Pneumonia (Positive)'],
    type: 'pie',
    hole: 0.6,
    marker: {
      colors: ['#10B981', '#EF4444']
    },
    textinfo: 'label+percent',
    hoverinfo: 'label+value+percent'
  }] : [];

  // Training History Chart Data
  const trainLossTrace = {
    x: trainingHistory.map(h => `Epoch ${h.epoch}`),
    y: trainingHistory.map(h => h.train_loss),
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Train Loss',
    line: { color: '#06B6D4', width: 2.5 }
  };
  const valLossTrace = {
    x: trainingHistory.map(h => `Epoch ${h.epoch}`),
    y: trainingHistory.map(h => h.validation_loss),
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Validation Loss',
    line: { color: '#6366F1', width: 2.5 }
  };

  return (
    <div className="page-container">
      {/* Header Banner */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Executive Research Dashboard</h1>
          <p className="page-subtitle">
            Explainable Chest X-Ray Pneumonia Detection & Localization Platform • RSNA Dataset
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <Link to="/analyze" className="btn btn-primary">
            <Activity size={16} />
            Analyze X-Ray
          </Link>
          <Link to="/training" className="btn btn-secondary">
            Training Lab
          </Link>
        </div>
      </div>

      {/* Medical Safety Banner */}
      <div className="disclaimer-banner">
        <AlertTriangle size={18} color="#F87171" style={{ minWidth: '18px' }} />
        <span>
          <strong>Medical Notice:</strong> SmartMed Vision is strictly an educational, portfolio, and research platform.
          Predictions and saliency heatmaps are for technical evaluation and MUST NOT be used for medical diagnosis.
        </span>
      </div>

      {/* Top Real Metrics Grid */}
      <div className="metrics-grid">
        <MetricCard
          title="Total RSNA Studies"
          value={summary ? summary.total_images.toLocaleString() : '—'}
          subtext="Chest radiographs (1024×1024)"
          icon={Database}
          color="cyan"
          badge="100% Real"
        />
        <MetricCard
          title="Pneumonia Positive"
          value={summary ? summary.positive_cases.toLocaleString() : '—'}
          subtext={summary ? `${summary.positive_percentage}% of cohort` : ''}
          icon={Activity}
          color="red"
          badge={`${summary?.total_bounding_boxes.toLocaleString() || '0'} BBoxes`}
        />
        <MetricCard
          title="Normal Studies"
          value={summary ? summary.negative_cases.toLocaleString() : '—'}
          subtext={summary ? `${summary.negative_percentage}% of cohort` : ''}
          icon={CheckCircle2}
          color="green"
        />
        <MetricCard
          title="Active Model Status"
          value={models.find(m => m.id === 'resnet18')?.status || 'Available'}
          subtext="ResNet-18 ImageNet Backbone"
          icon={Layers}
          color="indigo"
          badge="CPU First"
        />
      </div>

      {/* Middle Row: Charts */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        {/* Class Distribution Chart */}
        <div className="glass-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3>Cohort Class Distribution</h3>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>RSNA Training Set</span>
          </div>
          {summary ? (
            <PlotlyChart
              data={classDistData}
              layout={{
                showlegend: true,
                legend: { orientation: 'h', y: -0.15 },
                height: 290
              }}
            />
          ) : (
            <div className="empty-state">Loading dataset statistics...</div>
          )}
        </div>

        {/* Training History Progress Chart */}
        <div className="glass-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3>Latest Training Trajectory (ResNet18)</h3>
            <Link to="/training" style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)', textDecoration: 'none' }}>
              View Training &rarr;
            </Link>
          </div>
          {trainingHistory.length > 0 ? (
            <PlotlyChart
              data={[trainLossTrace, valLossTrace]}
              layout={{
                xaxis: { title: 'Epoch', gridcolor: 'rgba(255,255,255,0.06)' },
                yaxis: { title: 'Cross Entropy Loss', gridcolor: 'rgba(255,255,255,0.06)' },
                height: 290,
                legend: { orientation: 'h', y: -0.2 }
              }}
            />
          ) : (
            <div className="empty-state">
              <Clock className="empty-state-icon" size={36} />
              <div className="empty-state-title">No Training History Yet</div>
              <p className="empty-state-desc">
                Launch model training in the Training page to track real-time loss and accuracy curves.
              </p>
              <Link to="/training" className="btn btn-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}>
                Open Training Lab
              </Link>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Row: Evaluation Snapshot & Recent Predictions */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '1.5rem' }}>
        {/* Model Evaluation Summary */}
        <div className="glass-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3>Model Evaluation Snapshot</h3>
            <Link to="/evaluation" style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)', textDecoration: 'none' }}>
              Detailed Evaluation &rarr;
            </Link>
          </div>

          {evaluation ? (
            <div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem', marginBottom: '1.25rem' }}>
                <div style={{ background: 'var(--bg-elevated)', padding: '0.75rem', borderRadius: '8px', textAlign: 'center' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Accuracy</span>
                  <p className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.2rem' }}>
                    {(evaluation.metrics.accuracy * 100).toFixed(1)}%
                  </p>
                </div>
                <div style={{ background: 'var(--bg-elevated)', padding: '0.75rem', borderRadius: '8px', textAlign: 'center' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ROC-AUC</span>
                  <p className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-cyan)', marginTop: '0.2rem' }}>
                    {evaluation.metrics.roc_auc.toFixed(3)}
                  </p>
                </div>
                <div style={{ background: 'var(--bg-elevated)', padding: '0.75rem', borderRadius: '8px', textAlign: 'center' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Sensitivity</span>
                  <p className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: '#F87171', marginTop: '0.2rem' }}>
                    {(evaluation.metrics.sensitivity * 100).toFixed(1)}%
                  </p>
                </div>
                <div style={{ background: 'var(--bg-elevated)', padding: '0.75rem', borderRadius: '8px', textAlign: 'center' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Specificity</span>
                  <p className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: '#34D399', marginTop: '0.2rem' }}>
                    {(evaluation.metrics.specificity * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                Evaluated on <strong>{evaluation.evaluated_samples}</strong> hold-out patients ({evaluation.split} split) with 0 data leakage.
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-state-title">No Evaluation Results Available</div>
              <p className="empty-state-desc">
                Train ResNet18 and run test-set evaluation to inspect ROC, PR curves, and confusion matrix.
              </p>
              <Link to="/evaluation" className="btn btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}>
                Run Evaluation
              </Link>
            </div>
          )}
        </div>

        {/* Recent Predictions Audit */}
        <div className="glass-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3>Recent Predictions Audit</h3>
            <Link to="/analyze" style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)', textDecoration: 'none' }}>
              New Analysis &rarr;
            </Link>
          </div>

          {predictions.length > 0 ? (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Patient / ID</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                    <th>Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.slice(0, 5).map((p, idx) => (
                    <tr key={idx}>
                      <td className="mono" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{p.timestamp}</td>
                      <td className="mono" style={{ fontSize: '0.8rem' }}>{p.image_identifier?.substring(0, 16)}...</td>
                      <td>
                        <StatusBadge
                          status={p.prediction}
                          label={p.prediction?.includes('Pneumonia') ? 'Pneumonia' : 'Normal'}
                        />
                      </td>
                      <td className="mono" style={{ fontWeight: 600 }}>{p.probability}</td>
                      <td className="mono" style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)' }}>{p.latency_ms} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-state-title">No Recent Analyses</div>
              <p className="empty-state-desc">
                Upload a DICOM or test an RSNA study in the Analyze X-Ray module to view live predictions.
              </p>
              <Link to="/analyze" className="btn btn-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}>
                Upload X-Ray
              </Link>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
