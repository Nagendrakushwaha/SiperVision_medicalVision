import React, { useState, useEffect } from 'react';
import { Eye, ShieldAlert, Sparkles, Layers, RefreshCw, ZoomIn } from 'lucide-react';
import ImageViewer from '../components/ImageViewer';
import { getRandomExample, getPatientDetails, predictPatient } from '../services/api';

export default function Explainability() {
  const [loading, setLoading] = useState(false);
  const [patientData, setPatientData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [modelName, setModelName] = useState('resnet18');
  const [errorMsg, setErrorMsg] = useState(null);

  const loadRandomPatientWithBBox = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      // Pick a positive pneumonia case with bounding box
      const sample = await getRandomExample({ target: 1, has_bbox: true });
      if (sample) {
        const details = await getPatientDetails(sample.patient_id);
        setPatientData(details);
        // Run inference to get real Grad-CAM
        try {
          const pred = await predictPatient(sample.patient_id, modelName, true);
          setPrediction(pred);
        } catch (predErr) {
          setErrorMsg(predErr.response?.data?.detail || 'Inference failed. Check that a model checkpoint exists.');
        }
      }
    } catch (err) {
      setErrorMsg(err.response?.data?.detail || err.message || 'Failed to load sample.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadRandomPatientWithBBox();
  }, [modelName]);

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Explainability & Saliency Analysis</h1>
          <p className="page-subtitle">
            Visualizing Neural Attention via Gradient-Weighted Class Activation Mapping (Grad-CAM)
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <select
            className="form-select"
            value={modelName}
            onChange={e => setModelName(e.target.value)}
          >
            <option value="resnet18">ResNet18 (Target: layer4[-1])</option>
            <option value="mobilenet">MobileNetV3 (Target: features[-1])</option>
            <option value="efficientnet">EfficientNet-B0 (Target: features[-1])</option>
          </select>

          <button
            onClick={loadRandomPatientWithBBox}
            disabled={loading}
            className="btn btn-secondary"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Next Sample
          </button>
        </div>
      </div>

      {/* Saliency & Clinical Safety Banner */}
      <div className="disclaimer-banner">
        <ShieldAlert size={20} color="#F87171" style={{ minWidth: '20px' }} />
        <div>
          <strong>Interpretability Disclaimer:</strong> Grad-CAM highlights image regions that contributed mathematically to the model's logit activation.
          It serves as an analytical transparency technique and does <strong>not</strong> establish clinical correctness or anatomical pathology boundaries.
        </div>
      </div>

      {errorMsg && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '8px',
          padding: '1rem',
          marginBottom: '1.5rem',
          color: '#F87171',
          fontSize: '0.85rem'
        }}>
          {errorMsg}
        </div>
      )}

      {/* 4-Panel Synchronized Visual Grid */}
      <div className="glass-card" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <h3>Multi-View Comparative Attribution (Study: {patientData?.patient_id || 'Loading...'})</h3>
          <span className="badge badge-cyan">Side-by-Side Diagnostic Alignment</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
          {/* Panel 1: Original */}
          <div style={{ background: '#020408', borderRadius: '8px', padding: '0.75rem', border: '1px solid var(--border-subtle)' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>
              1. Original Radiograph
            </span>
            {patientData?.image_base64 ? (
              <img
                src={patientData.image_base64}
                alt="Original"
                style={{ width: '100%', maxHeight: '280px', objectFit: 'contain' }}
              />
            ) : <div style={{ height: '240px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>Loading...</div>}
          </div>

          {/* Panel 2: Heatmap */}
          <div style={{ background: '#020408', borderRadius: '8px', padding: '0.75rem', border: '1px solid var(--border-subtle)' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)', display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>
              2. Grad-CAM Heatmap
            </span>
            {prediction?.gradcam?.heatmap_base64 ? (
              <img
                src={prediction.gradcam.heatmap_base64}
                alt="Grad-CAM Heatmap"
                style={{ width: '100%', maxHeight: '280px', objectFit: 'contain' }}
              />
            ) : <div style={{ height: '240px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>Requires Trained Model</div>}
          </div>

          {/* Panel 3: Overlay */}
          <div style={{ background: '#020408', borderRadius: '8px', padding: '0.75rem', border: '1px solid var(--border-subtle)' }}>
            <span style={{ fontSize: '0.8rem', color: '#38BDF8', display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>
              3. Heatmap Overlay
            </span>
            {prediction?.gradcam?.overlay_base64 ? (
              <img
                src={prediction.gradcam.overlay_base64}
                alt="Grad-CAM Overlay"
                style={{ width: '100%', maxHeight: '280px', objectFit: 'contain' }}
              />
            ) : <div style={{ height: '240px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>Requires Trained Model</div>}
          </div>

          {/* Panel 4: Ground Truth */}
          <div style={{ background: '#020408', borderRadius: '8px', padding: '0.75rem', border: '1px solid var(--border-subtle)' }}>
            <span style={{ fontSize: '0.8rem', color: '#F87171', display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>
              4. RSNA Ground-Truth BBoxes
            </span>
            {patientData?.annotated_image_base64 ? (
              <img
                src={patientData.annotated_image_base64}
                alt="Ground Truth Boxes"
                style={{ width: '100%', maxHeight: '280px', objectFit: 'contain' }}
              />
            ) : <div style={{ height: '240px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>No BBoxes for Normal case</div>}
          </div>
        </div>
      </div>

      {/* Methodological Context Card */}
      <div className="glass-card">
        <h3 style={{ marginBottom: '0.75rem' }}>How Grad-CAM Computes Attribution</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.6', marginBottom: '1rem' }}>
          Gradient-weighted Class Activation Mapping (Grad-CAM) calculates the gradient of the winning logit score
          with respect to the activation maps of the final convolutional layer (for ResNet-18, the last BasicBlock in <code>layer4</code>).
          These gradients undergo global average pooling to obtain importance weights &alpha;<sub>k</sub>.
          A rectified linear unit (ReLU) is then applied to the weighted combination, capturing only the anatomical features
          that positively contributed to the pneumonia prediction.
        </p>

        {patientData?.boxes && patientData.boxes.length > 0 && (
          <div style={{ background: 'var(--bg-elevated)', padding: '1rem', borderRadius: '8px', fontSize: '0.85rem' }}>
            <span style={{ color: '#F87171', fontWeight: 600, display: 'block', marginBottom: '0.35rem' }}>
              Ground-Truth Radiologist Annotations (RSNA Challenge):
            </span>
            {patientData.boxes.map((b, i) => (
              <div key={i} className="mono" style={{ color: 'var(--text-primary)', marginBottom: '0.2rem' }}>
                Opacification Zone #{i + 1}: x={Math.round(b.x)}, y={Math.round(b.y)}, width={Math.round(b.width)}px, height={Math.round(b.height)}px
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
