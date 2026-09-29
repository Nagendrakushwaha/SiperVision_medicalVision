import React, { useState, useEffect, useRef } from 'react';
import {
  Flame,
  Play,
  Square,
  RefreshCw,
  Clock,
  TrendingDown,
  TrendingUp,
  Cpu,
  Layers,
  AlertTriangle,
  CheckCircle2,
  Eye,
  Zap,
  Sparkles,
  Info
} from 'lucide-react';
import PlotlyChart from '../components/PlotlyChart';
import StatusBadge from '../components/StatusBadge';
import { startTraining, stopTraining, getTrainingStatus, getTrainingHistory, getVisualProcessing } from '../services/api';

export default function Training() {
  // Config state
  const [modelName, setModelName] = useState('resnet18');
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(8);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [earlyStopping, setEarlyStopping] = useState(true);
  const [patience, setPatience] = useState(2);
  const [maxTrainSamples, setMaxTrainSamples] = useState('');

  // Live status state
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [errorMsg, setErrorMsg] = useState(null);
  const pollTimerRef = useRef(null);

  // Visual feature processing pipeline state
  const [visualPipeline, setVisualPipeline] = useState(null);
  const [loadingPipeline, setLoadingPipeline] = useState(false);
  const [sampleFilter, setSampleFilter] = useState('pneumonia');

  // Load initial status and history
  const refreshHistory = (model) => {
    getTrainingHistory(model)
      .then(res => setHistory(res.history || []))
      .catch(() => setHistory([]));
  };

  const fetchVisualPipeline = (model, filter = sampleFilter) => {
    setLoadingPipeline(true);
    const targetVal = filter === 'pneumonia' ? 1 : filter === 'normal' ? 0 : null;
    getVisualProcessing({ model_name: model || modelName, target: targetVal })
      .then(res => {
        setVisualPipeline(res);
        setLoadingPipeline(false);
      })
      .catch(err => {
        console.error('Visual pipeline fetch failed:', err);
        setLoadingPipeline(false);
      });
  };

  useEffect(() => {
    getTrainingStatus()
      .then(data => {
        setStatus(data);
        const isRunning = data && ['training', 'preparing', 'validating', 'saving'].includes(data.status);
        if (isRunning && data.model_name) {
          setModelName(data.model_name);
          refreshHistory(data.model_name);
          fetchVisualPipeline(data.model_name, sampleFilter);
        } else {
          refreshHistory(modelName);
          fetchVisualPipeline(modelName, sampleFilter);
        }
      })
      .catch(() => {});

    // Polling loop for active training progress
    pollTimerRef.current = setInterval(() => {
      getTrainingStatus()
        .then(data => {
          setStatus(data);
          const isRunning = data && ['training', 'preparing', 'validating', 'saving'].includes(data.status);
          if (isRunning && data.history && data.history.length > 0) {
            setHistory(data.history);
          }
        })
        .catch(() => {});
    }, 1500);

    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);

  const handleEpochChange = (val) => {
    const num = parseInt(val, 10);
    if (isNaN(num)) {
      setEpochs(1);
      return;
    }
    // Limit to 1 - 100
    if (num > 100) {
      setErrorMsg('Maximum supported epochs is 100.');
      setEpochs(100);
    } else if (num < 1) {
      setErrorMsg('Epochs must be at least 1.');
      setEpochs(1);
    } else {
      setErrorMsg(null);
      setEpochs(num);
    }
  };

  const handleStartTraining = async () => {
    if (epochs < 1 || epochs > 100) {
      setErrorMsg('Epochs must be between 1 and 100.');
      return;
    }

    setErrorMsg(null);
    try {
      await startTraining({
        model_name: modelName,
        epochs: epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        image_size: 224,
        early_stopping_patience: earlyStopping ? patience : 99,
        seed: 42,
        max_train_samples: maxTrainSamples ? parseInt(maxTrainSamples) : null
      });
    } catch (err) {
      setErrorMsg(err.response?.data?.detail || err.message || 'Failed to start training.');
    }
  };

  const handleStopTraining = async () => {
    try {
      await stopTraining();
    } catch (err) {
      setErrorMsg('Failed to signal stop: ' + (err.response?.data?.detail || err.message));
    }
  };

  const isTrainingActive = status && ['training', 'preparing', 'validating', 'saving'].includes(status.status);

  // Compute progress percentage
  let progressPct = 0;
  if (status && status.total_epochs > 0) {
    if (status.status === 'completed') {
      progressPct = 100;
    } else if (isTrainingActive) {
      const epochFrac = Math.max(0, status.epoch - 1) / status.total_epochs;
      const stepFrac = status.total_steps > 0 ? (status.step / status.total_steps) / status.total_epochs : 0;
      progressPct = Math.min(100, Math.round((epochFrac + stepFrac) * 100));
    }
  }

  // Chart traces
  const lossData = [
    {
      x: history.map(h => `Epoch ${h.epoch}`),
      y: history.map(h => h.train_loss),
      name: 'Train Loss',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#06B6D4', width: 2.5 }
    },
    {
      x: history.map(h => `Epoch ${h.epoch}`),
      y: history.map(h => h.validation_loss),
      name: 'Val Loss',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#6366F1', width: 2.5 }
    }
  ];

  const accData = [
    {
      x: history.map(h => `Epoch ${h.epoch}`),
      y: history.map(h => h.train_accuracy),
      name: 'Train Acc (%)',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#10B981', width: 2.5 }
    },
    {
      x: history.map(h => `Epoch ${h.epoch}`),
      y: history.map(h => h.validation_accuracy),
      name: 'Val Acc (%)',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#F59E0B', width: 2.5 }
    }
  ];

  const durationData = [
    {
      x: history.map(h => `Epoch ${h.epoch}`),
      y: history.map(h => h.epoch_duration),
      name: 'Duration (s)',
      type: 'bar',
      marker: { color: '#38BDF8' }
    }
  ];

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Neural Network Training Lab</h1>
          <p className="page-subtitle">
            Train ResNet18, MobileNetV3, or EfficientNet-B0 with 1–100 Epochs on AMD Ryzen 5 CPU
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button
            onClick={() => refreshHistory(modelName)}
            className="btn btn-secondary"
            style={{ fontSize: '0.82rem' }}
          >
            <RefreshCw size={14} />
            Refresh Logs
          </button>
        </div>
      </div>

      {/* Error banner if any */}
      {errorMsg && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '8px',
          padding: '0.85rem 1.25rem',
          marginBottom: '1.5rem',
          color: '#F87171',
          fontSize: '0.85rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          <AlertTriangle size={16} />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Main Grid: Config Form & Live Telemetry */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 380px) 1fr', gap: '1.5rem', alignItems: 'start', marginBottom: '2rem' }}>
        {/* Left Training Configuration Card */}
        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
            <Layers size={18} color="var(--accent-cyan)" />
            <h3 style={{ fontSize: '1.1rem' }}>Configuration</h3>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
            {/* Model Architecture */}
            <div className="form-group">
              <label className="form-label">Architecture</label>
              <select
                className="form-select"
                value={modelName}
                onChange={e => {
                  const selected = e.target.value;
                  setModelName(selected);
                  refreshHistory(selected);
                  fetchVisualPipeline(selected, sampleFilter);
                }}
                disabled={isTrainingActive}
              >
                <option value="resnet18">ResNet18 (Default Standard)</option>
                <option value="mobilenet">MobileNetV3-Small (Edge/Fast)</option>
                <option value="efficientnet">EfficientNet-B0 (Compound Scaled)</option>
              </select>
            </div>

            {/* Epochs Slider (1 - 100) */}
            <div className="slider-container">
              <div className="slider-header">
                <span className="form-label">Training Epochs (1–100)</span>
                <span className="slider-val">[{epochs}]</span>
              </div>
              <input
                type="range"
                min="1"
                max="100"
                step="1"
                value={epochs}
                onChange={e => handleEpochChange(e.target.value)}
                disabled={isTrainingActive}
              />
              {/* Quick preset buttons */}
              <div style={{ display: 'flex', gap: '0.35rem', marginTop: '0.25rem', flexWrap: 'wrap' }}>
                {[1, 5, 10, 20, 50, 100].map(ep => (
                  <button
                    key={ep}
                    type="button"
                    onClick={() => handleEpochChange(ep)}
                    disabled={isTrainingActive}
                    style={{
                      background: epochs === ep ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                      color: epochs === ep ? '#03131A' : 'var(--text-secondary)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: '4px',
                      padding: '0.2rem 0.5rem',
                      fontSize: '0.72rem',
                      fontWeight: 600,
                      cursor: 'pointer'
                    }}
                  >
                    {ep}
                  </button>
                ))}
              </div>
            </div>

            {/* Batch Size & Learning Rate */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div className="form-group">
                <label className="form-label">Batch Size</label>
                <select
                  className="form-select"
                  value={batchSize}
                  onChange={e => setBatchSize(parseInt(e.target.value))}
                  disabled={isTrainingActive}
                >
                  <option value="4">4 (Light)</option>
                  <option value="8">8 (Recommended)</option>
                  <option value="16">16 (Higher RAM)</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Learning Rate</label>
                <input
                  type="number"
                  step="0.00005"
                  className="form-input"
                  value={learningRate}
                  onChange={e => setLearningRate(parseFloat(e.target.value))}
                  disabled={isTrainingActive}
                />
              </div>
            </div>

            {/* Early Stopping & Patience */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.75rem 0', borderTop: '1px solid var(--border-subtle)', borderBottom: '1px solid var(--border-subtle)' }}>
              <div>
                <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Early Stopping</span>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Patience = {patience} epochs</p>
              </div>
              <input
                type="checkbox"
                checked={earlyStopping}
                onChange={e => setEarlyStopping(e.target.checked)}
                disabled={isTrainingActive}
                style={{ width: '18px', height: '18px', cursor: 'pointer' }}
              />
            </div>

            {/* Execution Device Ticker */}
            <div style={{ background: 'var(--bg-elevated)', padding: '0.75rem', borderRadius: '8px', fontSize: '0.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)' }}>
                <span>Execution Device:</span>
                <strong style={{ color: 'var(--accent-cyan)' }}>CPU (Ryzen 5 5500U)</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                <span>Imbalance Strategy:</span>
                <strong style={{ color: '#F87171' }}>Weighted Cross-Entropy</strong>
              </div>
            </div>

            {/* Action Buttons: Start / Stop */}
            <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.5rem' }}>
              {!isTrainingActive ? (
                <button
                  onClick={handleStartTraining}
                  className="btn btn-primary"
                  style={{ flex: 1, padding: '0.85rem' }}
                >
                  <Play size={16} fill="currentColor" />
                  Start Training
                </button>
              ) : (
                <button
                  onClick={handleStopTraining}
                  className="btn btn-danger"
                  style={{ flex: 1, padding: '0.85rem' }}
                >
                  <Square size={16} fill="currentColor" />
                  Stop Training Gracefully
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Right Live Training Progress & Telemetry Card */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div className="glass-card">
            {/* Status Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
              <div>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                  Training Status
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginTop: '0.25rem' }}>
                  <StatusBadge status={status?.status || 'idle'} />
                  <span style={{ fontSize: '0.95rem', fontWeight: 600 }}>
                    {status?.message || 'Ready to initiate training.'}
                  </span>
                </div>
              </div>

              {isTrainingActive && (
                <div className="mono" style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                  Epoch {status?.epoch} / {status?.total_epochs}
                </div>
              )}
            </div>

            {/* Progress Bar */}
            <div style={{ marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '0.35rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Overall Progress</span>
                <span className="mono" style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>{progressPct}%</span>
              </div>
              <div className="progress-bar-bg">
                <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
              </div>
            </div>

            {/* Real-Time Metrics Counters */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.75rem' }}>
              <div style={{ background: 'var(--bg-elevated)', padding: '0.75rem', borderRadius: '8px' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Train Loss</span>
                <p className="mono" style={{ fontSize: '1.2rem', fontWeight: 700, color: '#06B6D4', marginTop: '0.2rem' }}>
                  {status?.train_loss ? status.train_loss.toFixed(4) : '—'}
                </p>
              </div>

              <div style={{ background: 'var(--bg-elevated)', padding: '0.75rem', borderRadius: '8px' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Train Accuracy</span>
                <p className="mono" style={{ fontSize: '1.2rem', fontWeight: 700, color: '#10B981', marginTop: '0.2rem' }}>
                  {status?.train_acc ? `${status.train_acc.toFixed(1)}%` : '—'}
                </p>
              </div>

              <div style={{ background: 'var(--bg-elevated)', padding: '0.75rem', borderRadius: '8px' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Val Loss</span>
                <p className="mono" style={{ fontSize: '1.2rem', fontWeight: 700, color: '#6366F1', marginTop: '0.2rem' }}>
                  {status?.val_loss ? status.val_loss.toFixed(4) : '—'}
                </p>
              </div>

              <div style={{ background: 'var(--bg-elevated)', padding: '0.75rem', borderRadius: '8px' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Val Accuracy</span>
                <p className="mono" style={{ fontSize: '1.2rem', fontWeight: 700, color: '#F59E0B', marginTop: '0.2rem' }}>
                  {status?.val_acc ? `${status.val_acc.toFixed(1)}%` : '—'}
                </p>
              </div>

              <div style={{ background: 'var(--bg-elevated)', padding: '0.75rem', borderRadius: '8px' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Elapsed Time</span>
                <p className="mono" style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.2rem' }}>
                  {status?.elapsed_seconds ? `${status.elapsed_seconds.toFixed(0)}s` : '0s'}
                </p>
              </div>

              <div style={{ background: 'var(--bg-elevated)', padding: '0.75rem', borderRadius: '8px' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Est. Remaining</span>
                <p className="mono" style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                  {status?.estimated_remaining_seconds ? `${status.estimated_remaining_seconds.toFixed(0)}s` : '—'}
                </p>
              </div>
            </div>
          </div>

          {/* Charts Row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>
            <div className="glass-card">
              <h4 style={{ fontSize: '0.9rem', marginBottom: '0.75rem' }}>Loss Curve (Actual Training Logs)</h4>
              {history.length > 0 ? (
                <PlotlyChart
                  data={lossData}
                  layout={{
                    xaxis: { title: 'Epoch', gridcolor: 'rgba(255,255,255,0.06)' },
                    yaxis: { title: 'Cross Entropy', gridcolor: 'rgba(255,255,255,0.06)' },
                    height: 250,
                    legend: { orientation: 'h', y: -0.25 }
                  }}
                />
              ) : (
                <div className="empty-state" style={{ padding: '2rem' }}>
                  No training history available.
                </div>
              )}
            </div>

            <div className="glass-card">
              <h4 style={{ fontSize: '0.9rem', marginBottom: '0.75rem' }}>Accuracy Trajectory</h4>
              {history.length > 0 ? (
                <PlotlyChart
                  data={accData}
                  layout={{
                    xaxis: { title: 'Epoch', gridcolor: 'rgba(255,255,255,0.06)' },
                    yaxis: { title: 'Accuracy (%)', gridcolor: 'rgba(255,255,255,0.06)' },
                    height: 250,
                    legend: { orientation: 'h', y: -0.25 }
                  }}
                />
              ) : (
                <div className="empty-state" style={{ padding: '2rem' }}>
                  No accuracy history available.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Visual Feature Recognition & Edge Processing Pipeline Section */}
      <div className="glass-card" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <Eye size={20} color="var(--accent-cyan)" />
              <h3 style={{ fontSize: '1.15rem' }}>Neural Feature Recognition & Edge Processing Pipeline</h3>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.2rem' }}>
              Anatomical edge extraction and convolutional filter decomposition for <strong>{visualPipeline?.model_display_name || modelName.toUpperCase()}</strong>
            </p>
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', background: 'var(--bg-elevated)', borderRadius: '6px', padding: '0.2rem', border: '1px solid var(--border-subtle)' }}>
              <button
                type="button"
                onClick={() => { setSampleFilter('pneumonia'); fetchVisualPipeline(modelName, 'pneumonia'); }}
                style={{
                  background: sampleFilter === 'pneumonia' ? '#EF4444' : 'transparent',
                  color: sampleFilter === 'pneumonia' ? '#FFFFFF' : 'var(--text-secondary)',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '0.3rem 0.65rem',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                Pneumonia Positive
              </button>
              <button
                type="button"
                onClick={() => { setSampleFilter('normal'); fetchVisualPipeline(modelName, 'normal'); }}
                style={{
                  background: sampleFilter === 'normal' ? '#10B981' : 'transparent',
                  color: sampleFilter === 'normal' ? '#FFFFFF' : 'var(--text-secondary)',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '0.3rem 0.65rem',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                Normal Study
              </button>
              <button
                type="button"
                onClick={() => { setSampleFilter('all'); fetchVisualPipeline(modelName, 'all'); }}
                style={{
                  background: sampleFilter === 'all' ? 'var(--accent-cyan)' : 'transparent',
                  color: sampleFilter === 'all' ? '#03131A' : 'var(--text-secondary)',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '0.3rem 0.65rem',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                Random
              </button>
            </div>

            <button
              type="button"
              onClick={() => fetchVisualPipeline(modelName, sampleFilter)}
              disabled={loadingPipeline}
              className="btn btn-secondary"
              style={{ fontSize: '0.8rem', padding: '0.4rem 0.8rem' }}
            >
              <RefreshCw size={13} className={loadingPipeline ? 'animate-spin' : ''} />
              Cycle Sample
            </button>
          </div>
        </div>

        {/* Case Metadata Banner */}
        {visualPipeline && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'var(--bg-elevated)',
            padding: '0.65rem 1rem',
            borderRadius: '8px',
            marginBottom: '1.25rem',
            fontSize: '0.82rem',
            flexWrap: 'wrap',
            gap: '0.5rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Study ID:</span>
              <span className="mono" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{visualPipeline.patient_id}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Cohort Ground-Truth:</span>
              <span style={{
                color: visualPipeline.target === 1 ? '#F87171' : '#34D399',
                fontWeight: 700,
                background: visualPipeline.target === 1 ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                padding: '0.15rem 0.5rem',
                borderRadius: '4px'
              }}>
                {visualPipeline.label}
              </span>
              {visualPipeline.boxes && visualPipeline.boxes.length > 0 && (
                <span style={{ color: '#FBBF24', fontSize: '0.75rem' }}>
                  ({visualPipeline.boxes.length} Bounding Box{visualPipeline.boxes.length > 1 ? 'es' : ''})
                </span>
              )}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>CNN Confidence:</span>
              <span className="mono" style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>
                Pneumonia: {(visualPipeline.prediction.pneumonia_probability * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        )}

        {/* 5-Card Image Processing Grid */}
        {loadingPipeline ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            <RefreshCw size={28} className="animate-spin" style={{ margin: '0 auto 0.75rem' }} />
            <p>Extracting Sobel gradients, CLAHE contrast, and {modelName} Conv1 feature activations...</p>
          </div>
        ) : visualPipeline?.images ? (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
              {/* Card 1: Original */}
              <div style={{ background: '#020408', borderRadius: '10px', padding: '0.75rem', border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-primary)' }}>1. Input Radiograph</span>
                  <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', background: 'var(--bg-elevated)', padding: '0.1rem 0.35rem', borderRadius: '3px' }}>224×224</span>
                </div>
                <div style={{ height: '210px', background: '#05070D', borderRadius: '6px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <img src={visualPipeline.images.original} alt="Input" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                </div>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.5rem', lineHeight: '1.3' }}>
                  Zero-centered ImageNet tensor normalization applied.
                </p>
              </div>

              {/* Card 2: CLAHE Contrast */}
              <div style={{ background: '#020408', borderRadius: '10px', padding: '0.75rem', border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#38BDF8' }}>2. CLAHE Density</span>
                  <span style={{ fontSize: '0.68rem', color: '#38BDF8', background: 'rgba(56, 189, 248, 0.1)', padding: '0.1rem 0.35rem', borderRadius: '3px' }}>Clip=2.8</span>
                </div>
                <div style={{ height: '210px', background: '#05070D', borderRadius: '6px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <img src={visualPipeline.images.clahe} alt="CLAHE" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                </div>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.5rem', lineHeight: '1.3' }}>
                  Enhances soft tissue contrast and parenchymal opacities.
                </p>
              </div>

              {/* Card 3: Sobel Edge Gradient */}
              <div style={{ background: '#020408', borderRadius: '10px', padding: '0.75rem', border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>3. Sobel Edges</span>
                  <span style={{ fontSize: '0.68rem', color: 'var(--accent-cyan)', background: 'rgba(6, 182, 212, 0.1)', padding: '0.1rem 0.35rem', borderRadius: '3px' }}>∇I Gradient</span>
                </div>
                <div style={{ height: '210px', background: '#05070D', borderRadius: '6px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <img src={visualPipeline.images.sobel_edges} alt="Sobel Edges" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                </div>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.5rem', lineHeight: '1.3' }}>
                  Spatial gradient magnitude capturing ribs & lung boundaries.
                </p>
              </div>

              {/* Card 4: Canny Anatomical Overlay */}
              <div style={{ background: '#020408', borderRadius: '10px', padding: '0.75rem', border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#34D399' }}>4. Contour Overlay</span>
                  <span style={{ fontSize: '0.68rem', color: '#34D399', background: 'rgba(52, 211, 153, 0.1)', padding: '0.1rem 0.35rem', borderRadius: '3px' }}>Anatomy</span>
                </div>
                <div style={{ height: '210px', background: '#05070D', borderRadius: '6px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <img src={visualPipeline.images.canny_overlay} alt="Canny Overlay" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                </div>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.5rem', lineHeight: '1.3' }}>
                  Cyan contours highlight tissue interfaces and infiltrate borders.
                </p>
              </div>

              {/* Card 5: Conv1 Neural Feature Maps */}
              <div style={{ background: '#020408', borderRadius: '10px', padding: '0.75rem', border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#F59E0B' }}>5. Conv1 Features</span>
                  <span style={{ fontSize: '0.68rem', color: '#F59E0B', background: 'rgba(245, 158, 11, 0.1)', padding: '0.1rem 0.35rem', borderRadius: '3px' }}>CNN Filters</span>
                </div>
                <div style={{ height: '210px', background: '#05070D', borderRadius: '6px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <img src={visualPipeline.images.conv_features} alt="Conv1 Activations" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                </div>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.5rem', lineHeight: '1.3' }}>
                  Top 8 filter responses from the model's first convolutional layer.
                </p>
              </div>
            </div>

            {/* Explanatory Step Walkthrough */}
            <div style={{ background: 'var(--bg-elevated)', borderRadius: '8px', padding: '1rem 1.25rem', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <Zap size={16} color="var(--accent-cyan)" />
                <h4 style={{ fontSize: '0.9rem' }}>How This Convolutional Architecture Recognizes Pneumonia:</h4>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.85rem' }}>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                  <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '0.2rem' }}>
                    1. Early Layers (Conv1 / Edge Detectors):
                  </strong>
                  The first convolutional layer operates with directional kernels (similar to Gabor wavelets). They act as learned edge filters, firing strongly on sharp gradient transitions—such as the borders between air-filled dark lungs and radiopaque white fluid infiltrates.
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                  <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '0.2rem' }}>
                    2. Intermediate Blocks (Texture & Shape):
                  </strong>
                  Deeper residual/MBConv blocks combine edge gradients into complex texture descriptors. They distinguish normal anatomical structures (straight rib arcs, cardiac borders) from irregular, fluffy, or hazy consolidation patterns.
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                  <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '0.2rem' }}>
                    3. Deep Receptive Field & Classification:
                  </strong>
                  The final pooling layers aggregate wide spatial regions across entire lung lobes. The network weights these regional activations to output the final binary probability (Normal vs. Pneumonia) with high sensitivity.
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </div>

      {/* Bottom Permanent Training History Table */}
      <div className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3>Permanent Epoch History Log (reports/training/{modelName}/history.json)</h3>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {history.length} Epoch(s) Completed
          </span>
        </div>

        {history.length > 0 ? (
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Epoch</th>
                  <th>Train Loss</th>
                  <th>Val Loss</th>
                  <th>Train Accuracy</th>
                  <th>Val Accuracy</th>
                  <th>Learning Rate</th>
                  <th>Duration</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h, idx) => (
                  <tr key={idx}>
                    <td className="mono" style={{ fontWeight: 700, color: 'var(--accent-cyan)' }}>
                      #{h.epoch}
                    </td>
                    <td className="mono">{h.train_loss.toFixed(4)}</td>
                    <td className="mono" style={{ color: '#6366F1' }}>{h.validation_loss.toFixed(4)}</td>
                    <td className="mono">{h.train_accuracy.toFixed(1)}%</td>
                    <td className="mono" style={{ color: '#10B981', fontWeight: 600 }}>{h.validation_accuracy.toFixed(1)}%</td>
                    <td className="mono">{h.learning_rate}</td>
                    <td className="mono">{h.epoch_duration}s</td>
                    <td className="mono" style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      {h.timestamp?.split('T')[1]?.substring(0, 8) || h.timestamp}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            <p className="empty-state-desc">
              Select model configuration and click "Start Training" to begin training and generate permanent history logs.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
