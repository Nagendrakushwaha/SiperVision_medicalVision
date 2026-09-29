import React from 'react';
import { ShieldAlert, BookOpen, Cpu, Award, Database, Code } from 'lucide-react';

export default function About() {
  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">About SmartMed Vision</h1>
          <p className="page-subtitle">
            Explainable Pneumonia Detection & Localization Using Deep Learning on RSNA Cohort
          </p>
        </div>
      </div>

      {/* Mandatory Medical Safety Directive */}
      <div className="glass-card" style={{
        border: '1px solid rgba(239, 68, 68, 0.4)',
        boxShadow: 'var(--shadow-glow-red)',
        marginBottom: '2rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem', color: '#EF4444' }}>
          <ShieldAlert size={28} />
          <h2 style={{ fontSize: '1.35rem', color: '#EF4444' }}>Strict Clinical & Medical Disclaimer</h2>
        </div>
        <p style={{ color: '#FCA5A5', lineHeight: '1.7', fontSize: '0.95rem', marginBottom: '1rem' }}>
          <strong>SmartMed Vision is strictly an educational, portfolio, and research platform.</strong>
          Model predictions, probability scores, bounding-box annotations, and Grad-CAM saliency heatmaps may be incorrect,
          incomplete, or misaligned, and <strong>MUST NOT</strong> be used for clinical diagnosis, patient triage,
          or healthcare decision-making.
        </p>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          Always consult a board-certified radiologist or licensed medical professional for clinical interpretation of chest radiographs.
        </p>
      </div>

      {/* Key Architectural Pillars */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <Database size={20} color="var(--accent-cyan)" />
            <h3>RSNA Challenge Dataset</h3>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', lineHeight: '1.6' }}>
            Built exclusively upon the Radiological Society of North America (RSNA) Pneumonia Detection Challenge dataset,
            consisting of 26,684 frontal chest radiographs with expert radiologist bounding-box annotations for lung opacity.
          </p>
        </div>

        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <Cpu size={20} color="#10B981" />
            <h3>CPU-First Engineering</h3>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', lineHeight: '1.6' }}>
            Engineered from the ground up for AMD Ryzen 5 5500U processors without requiring NVIDIA GPUs, CUDA,
            or cloud inference. Demonstrates that robust medical AI inference can run locally at sub-50ms latencies.
          </p>
        </div>

        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <Code size={20} color="#6366F1" />
            <h3>Zero Synthetic Metrics</h3>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', lineHeight: '1.6' }}>
            Every chart, ROC curve, confusion matrix, inference probability, and latency number originates
            strictly from actual PyTorch model execution and authentic DICOM image parsing.
          </p>
        </div>
      </div>
    </div>
  );
}
