import React, { useState, useEffect } from 'react';
import {
  LineChart as LineChartIcon,
  Play,
  RotateCcw,
  Layers,
  Award,
  Clock,
  Cpu,
  HelpCircle,
  AlertCircle
} from 'lucide-react';
import PlotlyChart from '../components/PlotlyChart';
import MetricCard from '../components/MetricCard';
import { getEvaluation, runEvaluation } from '../services/api';

export default function Evaluation() {
  const [modelName, setModelName] = useState('resnet18');
  const [evalData, setEvalData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [sampleSize, setSampleSize] = useState(100);
  const [cmMode, setCmMode] = useState('raw'); // 'raw' or 'normalized'
  const [errorMsg, setErrorMsg] = useState(null);

  const fetchEval = (model) => {
    setLoading(true);
    getEvaluation(model)
      .then(res => {
        if (res.status === 'available') {
          setEvalData(res.data);
        } else {
          setEvalData(null);
        }
        setLoading(false);
      })
      .catch(() => {
        setEvalData(null);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchEval(modelName);
  }, [modelName]);

  const handleRunEvaluation = async () => {
    setEvaluating(true);
    setErrorMsg(null);
    try {
      const res = await runEvaluation({
        model_name: modelName,
        split_name: 'test',
        max_samples: sampleSize
      });
      setEvalData(res.results);
    } catch (err) {
      setErrorMsg(err.response?.data?.detail || err.message || 'Evaluation failed. Make sure a trained checkpoint exists.');
    } finally {
      setEvaluating(false);
    }
  };

  const metrics = evalData?.metrics;
  const cmRaw = evalData?.confusion_matrix?.raw;
  const cmNorm = evalData?.confusion_matrix?.normalized;
  const specs = evalData?.model_specs;

  // ROC Curve Data
  const rocTrace = evalData?.roc_curve ? {
    x: evalData.roc_curve.map(p => p.fpr),
    y: evalData.roc_curve.map(p => p.tpr),
    text: evalData.roc_curve.map(p => `Threshold: ${p.threshold}`),
    name: `ROC Curve (AUC = ${metrics?.roc_auc.toFixed(3)})`,
    type: 'scatter',
    mode: 'lines+markers',
    line: { color: '#06B6D4', width: 2.5 }
  } : null;

  const rocDiagonal = {
    x: [0, 1],
    y: [0, 1],
    name: 'Chance Diagonal',
    type: 'scatter',
    mode: 'lines',
    line: { dash: 'dash', color: '#64748B', width: 1.5 }
  };

  // PR Curve Data
  const prTrace = evalData?.pr_curve ? {
    x: evalData.pr_curve.map(p => p.recall),
    y: evalData.pr_curve.map(p => p.precision),
    name: `PR Curve (AP = ${metrics?.pr_auc.toFixed(3)})`,
    type: 'scatter',
    mode: 'lines+markers',
    line: { color: '#10B981', width: 2.5 }
  } : null;

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Model Evaluation Dashboard</h1>
          <p className="page-subtitle">
            Rigorous Hold-Out Test Evaluation • ROC Curves, PR Curves, and Confusion Matrices
          </p>
        </div>

        {/* Model & Evaluation Trigger Bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <select
            className="form-select"
            value={modelName}
            onChange={e => setModelName(e.target.value)}
          >
            <option value="resnet18">ResNet18</option>
            <option value="mobilenet">MobileNetV3-Small</option>
            <option value="efficientnet">EfficientNet-B0</option>
          </select>

          <select
            className="form-select"
            value={sampleSize}
            onChange={e => setSampleSize(parseInt(e.target.value))}
          >
            <option value="50">50 Test Samples (Fast)</option>
            <option value="100">100 Test Samples (Balanced)</option>
            <option value="200">200 Test Samples</option>
            <option value="500">500 Test Samples (Deep)</option>
          </select>

          <button
            onClick={handleRunEvaluation}
            disabled={evaluating}
            className="btn btn-primary"
            style={{ padding: '0.65rem 1.15rem' }}
          >
            {evaluating ? (
              <>
                <RotateCcw size={16} className="animate-spin" />
                Evaluating Hold-Out Set...
              </>
            ) : (
              <>
                <Play size={16} fill="currentColor" />
                Execute Test Evaluation
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error message if checkpoint missing */}
      {errorMsg && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '8px',
          padding: '1rem',
          marginBottom: '1.5rem',
          color: '#F87171',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          <AlertCircle size={18} />
          <span>{errorMsg}</span>
        </div>
      )}

      {evalData ? (
        <>
          {/* Top Metric Cards */}
          <div className="metrics-grid">
            <MetricCard
              title="Test Accuracy"
              value={metrics ? `${(metrics.accuracy * 100).toFixed(1)}%` : '—'}
              subtext="Overall correct classifications"
              icon={Award}
              color="green"
            />
            <MetricCard
              title="ROC-AUC Score"
              value={metrics ? metrics.roc_auc.toFixed(3) : '—'}
              subtext="Discrimination capacity"
              icon={LineChartIcon}
              color="cyan"
            />
            <MetricCard
              title="Sensitivity (Recall)"
              value={metrics ? `${(metrics.sensitivity * 100).toFixed(1)}%` : '—'}
              subtext="True Positive Rate (Pneumonia)"
              icon={Award}
              color="red"
            />
            <MetricCard
              title="Specificity"
              value={metrics ? `${(metrics.specificity * 100).toFixed(1)}%` : '—'}
              subtext="True Negative Rate (Normal)"
              icon={Award}
              color="indigo"
            />
          </div>

          {/* Secondary Metric Bar: F1, Precision, Latency, Size */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '1rem',
            marginBottom: '2rem'
          }}>
            <div className="glass-card" style={{ padding: '1rem' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Precision</span>
              <p className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                {(metrics?.precision * 100).toFixed(1)}%
              </p>
            </div>
            <div className="glass-card" style={{ padding: '1rem' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>F1-Score</span>
              <p className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: '#38BDF8' }}>
                {(metrics?.f1_score * 100).toFixed(1)}%
              </p>
            </div>
            <div className="glass-card" style={{ padding: '1rem' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>PR-AUC / AP</span>
              <p className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: '#FBBF24' }}>
                {metrics?.pr_auc.toFixed(3)}
              </p>
            </div>
            <div className="glass-card" style={{ padding: '1rem' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Avg CPU Latency</span>
              <p className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                {specs?.avg_latency_ms} ms
              </p>
            </div>
            <div className="glass-card" style={{ padding: '1rem' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Model Size / Params</span>
              <p className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                {specs?.model_size_mb} MB ({((specs?.total_parameters || 0) / 1e6).toFixed(1)}M)
              </p>
            </div>
          </div>

          {/* Charts Row: Confusion Matrix & ROC Curve & PR Curve */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
            {/* Interactive Confusion Matrix */}
            <div className="glass-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
                <div>
                  <h3 style={{ fontSize: '1.05rem' }}>Confusion Matrix</h3>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    Sample Size: {evalData.evaluated_samples} test patients
                  </p>
                </div>

                <div style={{ display: 'flex', background: 'var(--bg-elevated)', borderRadius: '6px', padding: '0.2rem' }}>
                  <button
                    onClick={() => setCmMode('raw')}
                    style={{
                      background: cmMode === 'raw' ? 'var(--accent-cyan)' : 'transparent',
                      color: cmMode === 'raw' ? '#03131A' : 'var(--text-secondary)',
                      border: 'none',
                      borderRadius: '4px',
                      padding: '0.25rem 0.65rem',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      cursor: 'pointer'
                    }}
                  >
                    Raw Counts
                  </button>
                  <button
                    onClick={() => setCmMode('normalized')}
                    style={{
                      background: cmMode === 'normalized' ? 'var(--accent-cyan)' : 'transparent',
                      color: cmMode === 'normalized' ? '#03131A' : 'var(--text-secondary)',
                      border: 'none',
                      borderRadius: '4px',
                      padding: '0.25rem 0.65rem',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      cursor: 'pointer'
                    }}
                  >
                    Normalized (%)
                  </button>
                </div>
              </div>

              {/* Confusion Matrix Visual Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr 1fr', gap: '0.5rem', alignItems: 'center' }}>
                <div />
                <div style={{ textAlign: 'center', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  Predicted Normal
                </div>
                <div style={{ textAlign: 'center', fontSize: '0.75rem', fontWeight: 600, color: '#F87171' }}>
                  Predicted Pneumonia
                </div>

                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  Actual Normal
                </div>
                <div style={{
                  background: 'rgba(16, 185, 129, 0.15)',
                  border: '1px solid rgba(16, 185, 129, 0.3)',
                  padding: '1.5rem',
                  borderRadius: '8px',
                  textAlign: 'center'
                }}>
                  <span style={{ fontSize: '0.72rem', color: '#34D399', textTransform: 'uppercase' }}>True Negative (TN)</span>
                  <p className="mono" style={{ fontSize: '1.5rem', fontWeight: 700, color: '#FFFFFF', marginTop: '0.25rem' }}>
                    {cmMode === 'raw' ? cmRaw?.tn : `${((cmNorm?.tn || 0) * 100).toFixed(1)}%`}
                  </p>
                </div>
                <div style={{
                  background: 'rgba(239, 68, 68, 0.08)',
                  border: '1px solid rgba(239, 68, 68, 0.2)',
                  padding: '1.5rem',
                  borderRadius: '8px',
                  textAlign: 'center'
                }}>
                  <span style={{ fontSize: '0.72rem', color: '#F87171', textTransform: 'uppercase' }}>False Positive (FP)</span>
                  <p className="mono" style={{ fontSize: '1.5rem', fontWeight: 700, color: '#FFFFFF', marginTop: '0.25rem' }}>
                    {cmMode === 'raw' ? cmRaw?.fp : `${((cmNorm?.fp || 0) * 100).toFixed(1)}%`}
                  </p>
                </div>

                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#F87171' }}>
                  Actual Pneumonia
                </div>
                <div style={{
                  background: 'rgba(239, 68, 68, 0.08)',
                  border: '1px solid rgba(239, 68, 68, 0.2)',
                  padding: '1.5rem',
                  borderRadius: '8px',
                  textAlign: 'center'
                }}>
                  <span style={{ fontSize: '0.72rem', color: '#F87171', textTransform: 'uppercase' }}>False Negative (FN)</span>
                  <p className="mono" style={{ fontSize: '1.5rem', fontWeight: 700, color: '#FFFFFF', marginTop: '0.25rem' }}>
                    {cmMode === 'raw' ? cmRaw?.fn : `${((cmNorm?.fn || 0) * 100).toFixed(1)}%`}
                  </p>
                </div>
                <div style={{
                  background: 'rgba(16, 185, 129, 0.15)',
                  border: '1px solid rgba(16, 185, 129, 0.3)',
                  padding: '1.5rem',
                  borderRadius: '8px',
                  textAlign: 'center'
                }}>
                  <span style={{ fontSize: '0.72rem', color: '#34D399', textTransform: 'uppercase' }}>True Positive (TP)</span>
                  <p className="mono" style={{ fontSize: '1.5rem', fontWeight: 700, color: '#FFFFFF', marginTop: '0.25rem' }}>
                    {cmMode === 'raw' ? cmRaw?.tp : `${((cmNorm?.tp || 0) * 100).toFixed(1)}%`}
                  </p>
                </div>
              </div>
            </div>

            {/* ROC Curve Chart */}
            <div className="glass-card">
              <h3 style={{ fontSize: '1.05rem', marginBottom: '0.5rem' }}>Receiver Operating Characteristic (ROC)</h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                False Positive Rate vs True Positive Rate across threshold sweep
              </p>
              {rocTrace && (
                <PlotlyChart
                  data={[rocTrace, rocDiagonal]}
                  layout={{
                    xaxis: { title: 'False Positive Rate (1 - Specificity)', range: [0, 1], gridcolor: 'rgba(255,255,255,0.06)' },
                    yaxis: { title: 'True Positive Rate (Sensitivity)', range: [0, 1.05], gridcolor: 'rgba(255,255,255,0.06)' },
                    height: 270,
                    legend: { orientation: 'h', y: -0.25 }
                  }}
                />
              )}
            </div>

            {/* PR Curve Chart */}
            <div className="glass-card">
              <h3 style={{ fontSize: '1.05rem', marginBottom: '0.5rem' }}>Precision-Recall Curve</h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                Trade-off curve essential for imbalanced clinical cohorts (AP: {metrics?.pr_auc.toFixed(3)})
              </p>
              {prTrace && (
                <PlotlyChart
                  data={[prTrace]}
                  layout={{
                    xaxis: { title: 'Recall (Sensitivity)', range: [0, 1], gridcolor: 'rgba(255,255,255,0.06)' },
                    yaxis: { title: 'Precision (Positive Predictive Value)', range: [0, 1.05], gridcolor: 'rgba(255,255,255,0.06)' },
                    height: 270
                  }}
                />
              )}
            </div>
          </div>
        </>
      ) : (
        <div className="glass-card empty-state">
          <Award size={48} className="empty-state-icon" color="var(--accent-cyan)" />
          <h3 className="empty-state-title">No Evaluation Results for {modelName}</h3>
          <p className="empty-state-desc">
            This model has not been evaluated yet. Select a sample size and click "Execute Test Evaluation" to calculate actual accuracy, ROC-AUC, PR curve, and latency.
          </p>
          <button
            onClick={handleRunEvaluation}
            disabled={evaluating}
            className="btn btn-primary"
            style={{ padding: '0.65rem 1.5rem' }}
          >
            {evaluating ? 'Evaluating on CPU...' : 'Run Test Evaluation Now'}
          </button>
        </div>
      )}
    </div>
  );
}
