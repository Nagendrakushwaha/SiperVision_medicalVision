import React, { useState, useEffect } from 'react';
import {
  Database,
  Search,
  Filter,
  Eye,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  AlertCircle,
  ExternalLink,
  Layers
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import StatusBadge from '../components/StatusBadge';
import ImageViewer from '../components/ImageViewer';
import { getDatasetExamples, getPatientDetails } from '../services/api';

export default function DatasetExplorer() {
  const navigate = useNavigate();
  const [examples, setExamples] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(false);

  // Filters
  const [splitFilter, setSplitFilter] = useState('all');
  const [targetFilter, setTargetFilter] = useState('');
  const [bboxFilter, setBboxFilter] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  // Selected Patient Details for Modal / Side Pane
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [patientDetails, setPatientDetails] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);

  const fetchExamples = () => {
    setLoading(true);
    const params = {
      page,
      page_size: 15,
      split: splitFilter !== 'all' ? splitFilter : undefined,
      target: targetFilter !== '' ? parseInt(targetFilter) : undefined,
      has_bbox: bboxFilter !== '' ? (bboxFilter === 'true') : undefined,
      search: searchQuery.trim() || undefined
    };

    getDatasetExamples(params)
      .then(res => {
        setExamples(res.items || []);
        setTotal(res.total || 0);
        setTotalPages(res.total_pages || 1);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchExamples();
  }, [page, splitFilter, targetFilter, bboxFilter]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setPage(1);
    fetchExamples();
  };

  const handleInspectPatient = (patient) => {
    setSelectedPatient(patient);
    setLoadingDetails(true);
    getPatientDetails(patient.patient_id)
      .then(data => {
        setPatientDetails(data);
        setLoadingDetails(false);
      })
      .catch(() => {
        setLoadingDetails(false);
      });
  };

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">RSNA Dataset Explorer</h1>
          <p className="page-subtitle">
            Browse and inspect 26,684 chest radiographs across leak-free patient-level partitions
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span className="badge badge-cyan" style={{ padding: '0.4rem 0.8rem', fontSize: '0.82rem' }}>
            Cohort: {total.toLocaleString()} Matching Studies
          </span>
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="glass-card" style={{ padding: '1.25rem', marginBottom: '1.5rem' }}>
        <form onSubmit={handleSearchSubmit} style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'center' }}>
          {/* Search Box */}
          <div style={{ flex: '1 1 240px', position: 'relative' }}>
            <input
              type="text"
              className="form-input"
              placeholder="Search by Patient UUID..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              style={{ width: '100%', paddingLeft: '2.4rem' }}
            />
            <Search size={16} color="var(--text-muted)" style={{ position: 'absolute', left: '12px', top: '12px' }} />
          </div>

          {/* Split Filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Split:</span>
            <select
              className="form-select"
              value={splitFilter}
              onChange={e => { setSplitFilter(e.target.value); setPage(1); }}
            >
              <option value="all">All Splits</option>
              <option value="train">Train (70%)</option>
              <option value="validation">Validation (15%)</option>
              <option value="test">Test (15%)</option>
            </select>
          </div>

          {/* Target Filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Class:</span>
            <select
              className="form-select"
              value={targetFilter}
              onChange={e => { setTargetFilter(e.target.value); setPage(1); }}
            >
              <option value="">All Classes</option>
              <option value="1">Pneumonia (Target = 1)</option>
              <option value="0">Normal (Target = 0)</option>
            </select>
          </div>

          {/* Bounding Box Filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Bounding Box:</span>
            <select
              className="form-select"
              value={bboxFilter}
              onChange={e => { setBboxFilter(e.target.value); setPage(1); }}
            >
              <option value="">All</option>
              <option value="true">With BBoxes</option>
              <option value="false">Without BBoxes</option>
            </select>
          </div>

          <button type="submit" className="btn btn-secondary">
            Apply Filters
          </button>
        </form>
      </div>

      {/* Main Content Layout: Table and Detail Inspector */}
      <div style={{ display: 'grid', gridTemplateColumns: selectedPatient ? '1fr 440px' : '1fr', gap: '1.5rem', alignItems: 'start' }}>
        {/* Table View */}
        <div className="glass-card" style={{ padding: '0', overflow: 'hidden' }}>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Patient Identifier</th>
                  <th>Target Class</th>
                  <th>Detailed Clinical Class</th>
                  <th>Partition</th>
                  <th>BBoxes</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan="6" style={{ textAlign: 'center', padding: '3rem' }}>
                      Loading cohort partition...
                    </td>
                  </tr>
                ) : examples.length === 0 ? (
                  <tr>
                    <td colSpan="6" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                      No patients match current filter parameters.
                    </td>
                  </tr>
                ) : (
                  examples.map((p) => {
                    const isSelected = selectedPatient?.patient_id === p.patient_id;
                    return (
                      <tr
                        key={p.patient_id}
                        style={{
                          background: isSelected ? 'rgba(6, 182, 212, 0.08)' : undefined,
                          cursor: 'pointer'
                        }}
                        onClick={() => handleInspectPatient(p)}
                      >
                        <td className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                          {p.patient_id}
                        </td>
                        <td>
                          <StatusBadge
                            status={p.target === 1 ? 'Pneumonia' : 'Normal'}
                            label={p.target === 1 ? 'Pneumonia' : 'Normal'}
                          />
                        </td>
                        <td style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                          {p.detailed_class}
                        </td>
                        <td>
                          <span style={{
                            fontSize: '0.75rem',
                            padding: '0.2rem 0.5rem',
                            borderRadius: '4px',
                            background: p.split === 'train' ? 'rgba(99, 102, 241, 0.15)' : (p.split === 'validation' ? 'rgba(245, 158, 11, 0.15)' : 'rgba(16, 185, 129, 0.15)'),
                            color: p.split === 'train' ? '#A5B4FC' : (p.split === 'validation' ? '#FDE68A' : '#6EE7B7'),
                            textTransform: 'uppercase',
                            fontWeight: 600
                          }}>
                            {p.split}
                          </span>
                        </td>
                        <td className="mono" style={{ color: p.boxes?.length > 0 ? '#F87171' : 'var(--text-muted)' }}>
                          {p.boxes?.length || 0}
                        </td>
                        <td>
                          <button
                            onClick={(e) => { e.stopPropagation(); handleInspectPatient(p); }}
                            className="btn btn-secondary"
                            style={{ padding: '0.35rem 0.75rem', fontSize: '0.75rem' }}
                          >
                            <Eye size={13} />
                            Inspect
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination Footer */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '1rem 1.5rem',
            borderTop: '1px solid var(--border-subtle)',
            background: 'rgba(15, 23, 42, 0.5)'
          }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Page <strong style={{ color: 'var(--text-primary)' }}>{page}</strong> of {totalPages}
            </span>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page <= 1 || loading}
                className="btn btn-secondary"
                style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
              >
                <ChevronLeft size={16} />
                Previous
              </button>
              <button
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages || loading}
                className="btn btn-secondary"
                style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
              >
                Next
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        </div>

        {/* Selected Patient Detail Inspector Side Pane */}
        {selectedPatient && (
          <div className="glass-card" style={{ position: 'sticky', top: '88px', maxHeight: 'calc(100vh - 110px)', overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)', textTransform: 'uppercase', fontWeight: 600 }}>
                  Patient Study Inspector
                </span>
                <h3 className="mono" style={{ fontSize: '0.95rem', wordBreak: 'break-all', marginTop: '0.2rem' }}>
                  {selectedPatient.patient_id}
                </h3>
              </div>
              <button
                onClick={() => setSelectedPatient(null)}
                className="btn btn-secondary"
                style={{ padding: '0.2rem 0.5rem', fontSize: '0.8rem' }}
              >
                &times;
              </button>
            </div>

            {loadingDetails ? (
              <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                Loading full DICOM pixel array...
              </div>
            ) : patientDetails ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {/* Image Render */}
                <div style={{ background: '#020408', borderRadius: '8px', overflow: 'hidden', textAlign: 'center' }}>
                  <img
                    src={patientDetails.annotated_image_base64 || patientDetails.image_base64}
                    alt="Study DICOM"
                    style={{ width: '100%', maxHeight: '300px', objectFit: 'contain' }}
                  />
                </div>

                {/* Patient Properties */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.82rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Class Label:</span>
                    <strong style={{ color: patientDetails.target === 1 ? '#EF4444' : '#10B981' }}>
                      {patientDetails.label}
                    </strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Detailed Status:</span>
                    <span>{patientDetails.detailed_class}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Partition:</span>
                    <span style={{ textTransform: 'uppercase' }}>{patientDetails.split}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>DICOM Dimensions:</span>
                    <span className="mono">{patientDetails.metadata?.rows}×{patientDetails.metadata?.columns}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Modality / Photometric:</span>
                    <span className="mono">{patientDetails.metadata?.modality} ({patientDetails.metadata?.photometric_interpretation})</span>
                  </div>
                </div>

                {/* Ground Truth Bounding Boxes Coordinates */}
                {patientDetails.boxes && patientDetails.boxes.length > 0 && (
                  <div style={{
                    background: 'rgba(239, 68, 68, 0.08)',
                    border: '1px solid rgba(239, 68, 68, 0.25)',
                    borderRadius: '8px',
                    padding: '0.85rem'
                  }}>
                    <span style={{ fontSize: '0.8rem', color: '#F87171', fontWeight: 600, display: 'block', marginBottom: '0.5rem' }}>
                      Ground-Truth Bounding Box Coordinates:
                    </span>
                    {patientDetails.boxes.map((b, idx) => (
                      <div key={idx} className="mono" style={{ fontSize: '0.78rem', color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                        Box #{idx + 1}: x={Math.round(b.x)}, y={Math.round(b.y)}, w={Math.round(b.width)}, h={Math.round(b.height)}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
