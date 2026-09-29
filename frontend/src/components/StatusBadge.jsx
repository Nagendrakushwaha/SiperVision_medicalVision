import React from 'react';

export default function StatusBadge({ status, label }) {
  const norm = String(status || '').toLowerCase();
  
  if (norm.includes('eval') || norm === 'available' || norm === 'completed' || norm === 'loaded') {
    return <span className="badge badge-normal">{label || status}</span>;
  }
  if (norm.includes('pneumonia') || norm === 'failed' || norm === 'missing') {
    return <span className="badge badge-pneumonia">{label || status}</span>;
  }
  if (norm.includes('train') || norm === 'preparing' || norm === 'saving') {
    return <span className="badge badge-warning">{label || status}</span>;
  }
  return <span className="badge badge-neutral">{label || status || 'Not trained'}</span>;
}
