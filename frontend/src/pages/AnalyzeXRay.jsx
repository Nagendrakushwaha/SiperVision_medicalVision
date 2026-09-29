import React, { useState } from 'react';
import {
  UploadCloud,
  FileText,
  Activity,
  Layers,
  Sparkles,
  AlertCircle,
  Dna,
  RefreshCw,
  Eye
} from 'lucide-react';
import { Link } from 'react-router-dom';
import ImageViewer from '../components/ImageViewer';
import PredictionCard from '../components/PredictionCard';
import { predictImage, predictPatient, getRandomExample, getPatientDetails } from '../services/api';

export default function AnalyzeXRay() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [patientId, setPatientId] = useState('');
  const [modelName, setModelName] = useState('resnet18');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [metadata, setMetadata] = useState(null);

  // Drag and drop handlers
  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setError(null);
    setPrediction(null);
    setPatientId('');

    // If standard image, create object URL for quick preview
    if (!file.name.toLowerCase().endsWith('.dcm')) {
      const url = URL.createObjectURL(file);
      setFilePreview(url);
    } else {
      setFilePreview(null);
    }

    setMetadata({
      filename: file.name,
      size_kb: (file.size / 1024).toFixed(1),
      type: file.name.toLowerCase().endsWith('.dcm') ? 'DICOM Medical Radiograph' : file.type
    });
  };

  // Run Inference
  const handleRunInference = async () => {
    if (!selectedFile && !patientId) {
      setError('Please upload a file or choose an RSNA sample first.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      let res;
      if (selectedFile) {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model_name', modelName);
        formData.append('include_gradcam', 'true');
        if (patientId) formData.append('patient_id', patientId);
        res = await predictImage(formData);
      } else {
        res = await predictPatient(patientId, modelName, true);
      }

      setPrediction(res);
      if (res.gradcam?.original_base64) {
        setFilePreview(res.gradcam.original_base64);
      }
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Inference failed.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  // Sample loader from real dataset
  const handleLoadRandomSample = async (targetFilter = null) => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    setSelectedFile(null);

    try {
      const sample = await getRandomExample({ target: targetFilter });
      if (sample) {
        setPatientId(sample.patient_id);
        const details = await getPatientDetails(sample.patient_id);
        setFilePreview(details.image_base64);
        setMetadata({
          patient_id: sample.patient_id,
          target: sample.target,
          class: details.detailed_class,
          dimensions: `${details.metadata.rows}×${details.metadata.columns}`,
          modality: details.metadata.modality,
          boxes_count: details.box_count,
          photometric: details.metadata.photometric_interpretation
        });
      }
    } catch (err) {
      setError('Failed to fetch RSNA sample: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      {/* Page Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Analyze Chest Radiograph</h1>
          <p className="page-subtitle">
            Explainable AI Inference with Grad-CAM Activation and RSNA Ground Truth Overlay
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Model:</span>
            <select
              className="form-select"
              value={modelName}
              onChange={e => setModelName(e.target.value)}
              style={{ padding: '0.45rem 0.85rem' }}
            >
              <option value="resnet18">ResNet-18 (Default)</option>
              <option value="mobilenet">MobileNetV3-Small</option>
              <option value="efficientnet">EfficientNet-B0</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main Workspace Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(340px, 400px) 1fr', gap: '1.5rem', alignItems: 'start' }}>
        {/* Left Control Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Upload Card */}
          <div
            className="glass-card"
            onDragOver={e => e.preventDefault()}
            onDrop={handleDrop}
            style={{
              border: '2px dashed var(--border-subtle)',
              textAlign: 'center',
              padding: '2rem 1.5rem',
              cursor: 'pointer',
              transition: 'border-color 0.2s ease'
            }}
          >
            <input
              type="file"
              id="file-upload"
              accept=".dcm,.png,.jpg,.jpeg"
              onChange={e => e.target.files && handleFileSelect(e.target.files[0])}
              style={{ display: 'none' }}
            />
            <label htmlFor="file-upload" style={{ cursor: 'pointer', display: 'block' }}>
              <div style={{
                width: '54px',
                height: '54px',
                borderRadius: '50%',
                background: 'rgba(6, 182, 212, 0.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 1rem auto'
              }}>
                <UploadCloud size={26} color="var(--accent-cyan)" />
              </div>
              <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.35rem' }}>
                Drag & Drop Radiograph
              </h3>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                Supports DICOM (.dcm), PNG, JPG (1024×1024)
              </p>
              <span className="btn btn-secondary" style={{ padding: '0.45rem 1rem', fontSize: '0.82rem' }}>
                Browse Files
              </span>
            </label>
          </div>

          {/* Quick RSNA Sample Selector */}
          <div className="glass-card" style={{ padding: '1.25rem' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
              Load Real RSNA Challenge Study
            </span>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginTop: '0.75rem' }}>
              <button
                onClick={() => handleLoadRandomSample(1)}
                disabled={loading}
                className="btn btn-secondary"
                style={{ fontSize: '0.8rem', padding: '0.5rem' }}
              >
                Pneumonia Case
              </button>
              <button
                onClick={() => handleLoadRandomSample(0)}
                disabled={loading}
                className="btn btn-secondary"
                style={{ fontSize: '0.8rem', padding: '0.5rem' }}
              >
                Normal Study
              </button>
            </div>
          </div>

          {/* Metadata Card if file/patient loaded */}
          {metadata && (
            <div className="glass-card" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <FileText size={16} color="var(--accent-cyan)" />
                <h4 style={{ fontSize: '0.9rem' }}>Study Metadata</h4>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.82rem' }}>
                {metadata.patient_id && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Patient ID:</span>
                    <span className="mono">{metadata.patient_id.substring(0, 16)}...</span>
                  </div>
                )}
                {metadata.filename && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>File Name:</span>
                    <span className="mono">{metadata.filename}</span>
                  </div>
                )}
                {metadata.dimensions && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Dimensions:</span>
                    <span className="mono">{metadata.dimensions}</span>
                  </div>
                )}
                {metadata.modality && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Modality:</span>
                    <span className="mono">{metadata.modality}</span>
                  </div>
                )}
                {metadata.photometric && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Photometric:</span>
                    <span className="mono">{metadata.photometric}</span>
                  </div>
                )}
                {metadata.boxes_count !== undefined && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>RSNA BBoxes:</span>
                    <span className="mono" style={{ color: metadata.boxes_count > 0 ? '#F87171' : 'inherit' }}>
                      {metadata.boxes_count}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Action Button */}
          <button
            onClick={handleRunInference}
            disabled={loading || (!selectedFile && !patientId)}
            className="btn btn-primary"
            style={{ width: '100%', padding: '0.85rem', fontSize: '1rem' }}
          >
            {loading ? (
              <>
                <RefreshCw size={18} className="animate-spin" />
                Analyzing on CPU...
              </>
            ) : (
              <>
                <Sparkles size={18} />
                Execute Explainable Inference
              </>
            )}
          </button>

          {/* Error Message Display */}
          {error && (
            <div style={{
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '8px',
              padding: '1rem',
              color: '#F87171',
              fontSize: '0.85rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem', fontWeight: 600 }}>
                <AlertCircle size={16} />
                <span>Notice</span>
              </div>
              <p>{error}</p>
              {error.toLowerCase().includes('train') && (
                <Link to="/training" className="btn btn-secondary" style={{ marginTop: '0.75rem', padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>
                  Go to Training Lab &rarr;
                </Link>
              )}
            </div>
          )}
        </div>

        {/* Right Viewer & Prediction Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Prediction Result Card */}
          {prediction && (
            <PredictionCard prediction={prediction} />
          )}

          {/* Interactive Image & Explainability Viewer */}
          {filePreview ? (
            <ImageViewer
              originalSrc={filePreview}
              overlaySrc={prediction?.gradcam?.overlay_base64}
              heatmapSrc={prediction?.gradcam?.heatmap_base64}
              annotatedSrc={prediction?.ground_truth?.bounding_box_image_base64}
              title={metadata?.patient_id ? `Study: ${metadata.patient_id}` : (metadata?.filename || 'Chest X-Ray Study')}
            />
          ) : (
            <div className="glass-card" style={{
              minHeight: '440px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center',
              padding: '3rem 1.5rem'
            }}>
              <div style={{
                width: '64px',
                height: '64px',
                borderRadius: '50%',
                background: 'rgba(255, 255, 255, 0.03)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '1rem'
              }}>
                <Eye size={32} color="var(--text-muted)" />
              </div>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                No Radiograph Selected
              </h3>
              <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', maxWidth: '420px', marginBottom: '1.5rem' }}>
                Upload a DICOM file or click "Pneumonia Case" / "Normal Study" to load an actual RSNA challenge study with ground truth boxes.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
