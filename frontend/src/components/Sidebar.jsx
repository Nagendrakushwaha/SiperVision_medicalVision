import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  ScanEye,
  Database,
  Flame,
  LineChart,
  GitCompare,
  Eye,
  Zap,
  Clock,
  FileSpreadsheet,
  Cpu,
  Info,
  Activity
} from 'lucide-react';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/analyze', label: 'Analyze X-Ray', icon: ScanEye },
  { path: '/dataset', label: 'Dataset Explorer', icon: Database },
  { path: '/training', label: 'Training', icon: Flame },
  { path: '/evaluation', label: 'Evaluation', icon: LineChart },
  { path: '/comparison', label: 'Model Comparison', icon: GitCompare },
  { path: '/explainability', label: 'Explainability', icon: Eye },
  { path: '/benchmarks', label: 'Benchmarks', icon: Zap },
  { path: '/experiments', label: 'Experiments', icon: Clock },
  { path: '/reports', label: 'Reports', icon: FileSpreadsheet },
  { path: '/system', label: 'System Status', icon: Cpu },
  { path: '/about', label: 'About & Safety', icon: Info },
];

export default function Sidebar() {
  return (
    <aside style={{
      width: '260px',
      minWidth: '260px',
      background: 'rgba(11, 17, 32, 0.95)',
      backdropFilter: 'blur(20px)',
      borderRight: '1px solid var(--border-subtle)',
      display: 'flex',
      flexDirection: 'column',
      minHeight: '100vh',
      position: 'sticky',
      top: 0,
      zIndex: 40
    }}>
      {/* Brand Header */}
      <div style={{
        padding: '1.75rem 1.5rem',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem'
      }}>
        <div style={{
          width: '38px',
          height: '38px',
          borderRadius: '10px',
          background: 'linear-gradient(135deg, var(--accent-cyan), #0284C7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 15px rgba(6, 182, 212, 0.4)'
        }}>
          <Activity size={22} color="#03131A" strokeWidth={2.5} />
        </div>
        <div>
          <h2 style={{
            fontSize: '1.15rem',
            fontWeight: 800,
            letterSpacing: '0.04em',
            background: 'linear-gradient(135deg, #FFFFFF, #22D3EE)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            lineHeight: 1.1
          }}>
            SMARTMED
          </h2>
          <span style={{
            fontSize: '0.68rem',
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            color: 'var(--text-muted)',
            fontWeight: 600
          }}>
            RSNA Vision AI
          </span>
        </div>
      </div>

      {/* Navigation Links */}
      <nav style={{
        padding: '1.25rem 0.85rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.25rem',
        flex: 1,
        overflowY: 'auto'
      }}>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              style={({ isActive }) => ({
                display: 'flex',
                alignItems: 'center',
                gap: '0.85rem',
                padding: '0.7rem 0.95rem',
                borderRadius: '8px',
                color: isActive ? '#FFFFFF' : 'var(--text-secondary)',
                background: isActive ? 'linear-gradient(90deg, rgba(6, 182, 212, 0.15) 0%, rgba(99, 102, 241, 0.08) 100%)' : 'transparent',
                borderLeft: isActive ? '3px solid var(--accent-cyan)' : '3px solid transparent',
                textDecoration: 'none',
                fontSize: '0.88rem',
                fontWeight: isActive ? 600 : 500,
                transition: 'all 0.15s ease'
              })}
            >
              {({ isActive }) => (
                <>
                  <Icon size={18} color={isActive ? 'var(--accent-cyan)' : 'currentColor'} />
                  <span>{item.label}</span>
                </>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Device Badge Bottom */}
      <div style={{
        padding: '1rem 1.25rem',
        borderTop: '1px solid var(--border-subtle)',
        background: 'rgba(15, 23, 42, 0.5)'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '0.35rem'
        }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
            Device
          </span>
          <span style={{
            fontSize: '0.72rem',
            padding: '0.15rem 0.5rem',
            borderRadius: '4px',
            background: 'rgba(6, 182, 212, 0.15)',
            color: 'var(--accent-cyan)',
            fontWeight: 700
          }}>
            CPU / RYZEN
          </span>
        </div>
        <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
          AMD Ryzen 5 5500U
        </p>
      </div>
    </aside>
  );
}
