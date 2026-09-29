import React from 'react';

export default function MetricCard({ title, value, subtext, icon: Icon, badge, color = 'cyan' }) {
  const colorMap = {
    cyan: { text: 'var(--accent-cyan)', glow: 'rgba(6, 182, 212, 0.15)' },
    red: { text: '#F87171', glow: 'rgba(239, 68, 68, 0.15)' },
    green: { text: '#34D399', glow: 'rgba(16, 185, 129, 0.15)' },
    indigo: { text: '#818CF8', glow: 'rgba(99, 102, 241, 0.15)' },
    amber: { text: '#FBBF24', glow: 'rgba(245, 158, 11, 0.15)' }
  };

  const scheme = colorMap[color] || colorMap.cyan;

  return (
    <div className="glass-card metric-card">
      <div className="metric-header">
        <span>{title}</span>
        {Icon && (
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '8px',
            background: scheme.glow,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Icon size={18} color={scheme.text} />
          </div>
        )}
      </div>
      <div className="metric-value" style={{ color: scheme.text }}>
        {value !== undefined && value !== null ? value : '—'}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span className="metric-sub">{subtext}</span>
        {badge && <span className="badge badge-neutral">{badge}</span>}
      </div>
    </div>
  );
}
