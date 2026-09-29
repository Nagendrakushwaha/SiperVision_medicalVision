import React, { useEffect, useState } from 'react';
import { Cpu, ShieldAlert, CheckCircle2, AlertTriangle, Layers } from 'lucide-react';
import { getSystemStatus } from '../services/api';

export default function Header() {
  const [system, setSystem] = useState(null);
  const [connected, setConnected] = useState(true);

  useEffect(() => {
    getSystemStatus()
      .then(data => {
        setSystem(data);
        setConnected(true);
      })
      .catch(() => {
        setConnected(false);
      });
    
    // Poll system status occasionally
    const interval = setInterval(() => {
      getSystemStatus()
        .then(data => {
          setSystem(data);
          setConnected(true);
        })
        .catch(() => setConnected(false));
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  const hw = system?.hardware || {
    cpu: 'AMD Ryzen 5 5500U',
    ram: '16 GB',
    gpu: 'AMD Radeon Integrated',
    cuda: 'Not Available',
    execution_device: 'CPU'
  };

  return (
    <header style={{
      height: '68px',
      background: 'rgba(15, 23, 42, 0.75)',
      backdropFilter: 'blur(16px)',
      borderBottom: '1px solid var(--border-subtle)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 2rem',
      position: 'sticky',
      top: 0,
      zIndex: 30
    }}>
      {/* Hardware specs ticker */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', fontSize: '0.82rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)' }}>
          <Cpu size={16} color="var(--accent-cyan)" />
          <span>CPU: <strong style={{ color: 'var(--text-primary)' }}>{hw.cpu?.split('with')[0]?.trim() || 'Ryzen 5 5500U'}</strong></span>
        </div>
        <div style={{ width: '1px', height: '14px', background: 'var(--border-subtle)' }} />
        <span style={{ color: 'var(--text-secondary)' }}>RAM: <strong style={{ color: 'var(--text-primary)' }}>{hw.ram}</strong></span>
        <div style={{ width: '1px', height: '14px', background: 'var(--border-subtle)' }} />
        <span style={{ color: 'var(--text-secondary)' }}>GPU: <strong style={{ color: 'var(--text-primary)' }}>{hw.gpu}</strong></span>
        <div style={{ width: '1px', height: '14px', background: 'var(--border-subtle)' }} />
        <span style={{ color: 'var(--text-secondary)' }}>Execution: <strong style={{ color: 'var(--accent-cyan)' }}>{hw.execution_device}</strong></span>
      </div>

      {/* Right status badges */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {/* Active Model Indicator */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          padding: '0.35rem 0.75rem',
          borderRadius: '9999px',
          background: 'rgba(99, 102, 241, 0.12)',
          border: '1px solid rgba(99, 102, 241, 0.25)',
          fontSize: '0.78rem',
          fontWeight: 600,
          color: '#A5B4FC'
        }}>
          <Layers size={14} color="#818CF8" />
          <span>Active: ResNet18</span>
        </div>

        {/* Backend Connectivity Status */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.45rem',
          padding: '0.35rem 0.75rem',
          borderRadius: '9999px',
          background: connected ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)',
          border: `1px solid ${connected ? 'rgba(16, 185, 129, 0.25)' : 'rgba(239, 68, 68, 0.25)'}`,
          fontSize: '0.78rem',
          fontWeight: 600,
          color: connected ? '#34D399' : '#F87171'
        }}>
          <div style={{
            width: '7px',
            height: '7px',
            borderRadius: '50%',
            background: connected ? '#10B981' : '#EF4444',
            boxShadow: `0 0 8px ${connected ? '#10B981' : '#EF4444'}`
          }} />
          <span>{connected ? 'Backend Live' : 'Disconnected'}</span>
        </div>
      </div>
    </header>
  );
}
