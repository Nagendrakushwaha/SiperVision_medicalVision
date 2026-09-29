import React from 'react';
import { Activity, Clock, Cpu, FileCheck, AlertCircle, ShieldAlert } from 'lucide-react';

export default function PredictionCard({ prediction }) {
  if (!prediction) return null;

  const isPneumonia = prediction.target_class === 1 || prediction.prediction?.toLowerCase().includes('pneumonia');
  const pPneumonia = prediction.pneumonia_probability ?? 0;
  const pNormal = prediction.normal_probability ?? 0;

  return (
    <div className="glass-card" style={{
      border: `1px solid ${isPneumonia ? 'rgba(239, 68, 68, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`,
      boxShadow: isPneumonia ? 'var(--shadow-glow-red)' : '0 0 25px rgba(16, 185, 129, 0.2)',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Top Banner */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '1.25rem',
        flexWrap: 'wrap',
        gap: '0.75rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
          <div style={{
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            background: isPneumonia ? '#EF4444' : '#10B981',
            boxShadow: `0 0 10px ${isPneumonia ? '#EF4444' : '#10B981'}`
          }} />
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600 }}>
            Model Classification Result
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{
            fontSize: '0.75rem',
            padding: '0.25rem 0.65rem',
            borderRadius: '9999px',
            background: 'rgba(6, 182, 212, 0.12)',
            color: 'var(--accent-cyan)',
            fontWeight: 600
          }}>
            Latency: {prediction.latency_ms} ms
          </span>
          <span style={{
            fontSize: '0.75rem',
            padding: '0.25rem 0.65rem',
            borderRadius: '9999px',
            background: 'rgba(148, 163, 184, 0.12)',
            color: 'var(--text-secondary)',
            fontWeight: 600
          }}>
            Device: {prediction.device || 'CPU'}
          </span>
        </div>
      </div>

      {/* Main Prediction Headline */}
      <div style={{
        display: 'flex',
        alignItems: 'baseline',
        justifyContent: 'space-between',
        marginBottom: '1.5rem',
        flexWrap: 'wrap',
        gap: '1rem'
      }}>
        <div>
          <h1 style={{
            fontSize: '2.5rem',
            fontWeight: 800,
            color: isPneumonia ? '#EF4444' : '#10B981',
            letterSpacing: '-0.03em',
            lineHeight: 1
          }}>
            {isPneumonia ? 'PNEUMONIA DETECTED' : 'NORMAL / NO LUNG OPACITY'}
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.4rem' }}>
            Model output confidence: <strong style={{ color: 'var(--text-primary)' }}>{prediction.confidence}%</strong>
          </p>
        </div>

        <div style={{ textAlign: 'right' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Model Checkpoint</span>
          <p className="mono" style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
            {prediction.checkpoint || 'resnet18/best.pth'}
          </p>
        </div>
      </div>

      {/* Probability Visualization Bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', marginBottom: '1.5rem' }}>
        {/* Pneumonia bar */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.35rem' }}>
            <span style={{ color: '#F87171', fontWeight: 600 }}>Pneumonia (Lung Opacity)</span>
            <span className="mono" style={{ color: '#F87171', fontWeight: 700 }}>{pPneumonia.toFixed(1)}%</span>
          </div>
          <div style={{ width: '100%', height: '10px', background: 'var(--bg-elevated)', borderRadius: '9999px', overflow: 'hidden' }}>
            <div style={{
              width: `${pPneumonia}%`,
              height: '100%',
              background: 'linear-gradient(90deg, #F87171, #EF4444)',
              transition: 'width 0.5s ease-out'
            }} />
          </div>
        </div>

        {/* Normal bar */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.35rem' }}>
            <span style={{ color: '#34D399', fontWeight: 600 }}>Normal (No Opacity)</span>
            <span className="mono" style={{ color: '#34D399', fontWeight: 700 }}>{pNormal.toFixed(1)}%</span>
          </div>
          <div style={{ width: '100%', height: '10px', background: 'var(--bg-elevated)', borderRadius: '9999px', overflow: 'hidden' }}>
            <div style={{
              width: `${pNormal}%`,
              height: '100%',
              background: 'linear-gradient(90deg, #34D399, #10B981)',
              transition: 'width 0.5s ease-out'
            }} />
          </div>
        </div>
      </div>

      {/* Ground Truth Bounding Box pill if present */}
      {prediction.ground_truth && (
        <div style={{
          padding: '0.75rem 1rem',
          borderRadius: '8px',
          background: 'rgba(239, 68, 68, 0.08)',
          border: '1px solid rgba(239, 68, 68, 0.2)',
          marginBottom: '1rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '0.82rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <ShieldAlert size={16} color="#F87171" />
            <span>RSNA Ground Truth Label: <strong>{prediction.ground_truth.label}</strong></span>
          </div>
          <span className="mono" style={{ color: 'var(--text-secondary)' }}>
            {prediction.ground_truth.box_count} Ground Truth Box(es)
          </span>
        </div>
      )}

      {/* Safety Notice */}
      <div style={{
        fontSize: '0.75rem',
        color: 'var(--text-muted)',
        borderTop: '1px solid var(--border-subtle)',
        paddingTop: '0.85rem',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem'
      }}>
        <AlertCircle size={14} color="#94A3B8" />
        <span>Statistical probability does not constitute clinical diagnostic certainty. Educational and research use only.</span>
      </div>
    </div>
  );
}
