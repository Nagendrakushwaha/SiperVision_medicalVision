import React, { useState, useRef } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
  Minimize2,
  Sliders,
  Eye,
  Crosshair,
  Layers
} from 'lucide-react';

export default function ImageViewer({
  originalSrc,
  overlaySrc,
  heatmapSrc,
  annotatedSrc,
  title = "Chest X-Ray Study",
  showControls = true,
  allowModes = true
}) {
  const [zoom, setZoom] = useState(1);
  const [brightness, setBrightness] = useState(1);
  const [contrast, setContrast] = useState(1);
  const [fullscreen, setFullscreen] = useState(false);
  const [activeLayer, setActiveLayer] = useState('original'); // 'original', 'overlay', 'heatmap', 'annotated', 'side_by_side'
  const containerRef = useRef(null);

  const resetAdjustments = () => {
    setZoom(1);
    setBrightness(1);
    setContrast(1);
  };

  const toggleFullscreen = () => {
    if (!fullscreen) {
      if (containerRef.current?.requestFullscreen) {
        containerRef.current.requestFullscreen();
      }
      setFullscreen(true);
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
      setFullscreen(false);
    }
  };

  // Determine which image to render based on activeLayer
  const getRenderSrc = () => {
    if (activeLayer === 'overlay' && overlaySrc) return overlaySrc;
    if (activeLayer === 'heatmap' && heatmapSrc) return heatmapSrc;
    if (activeLayer === 'annotated' && annotatedSrc) return annotatedSrc;
    return originalSrc;
  };

  return (
    <div
      ref={containerRef}
      className="glass-card"
      style={{
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        background: '#040711',
        overflow: 'hidden',
        border: '1px solid rgba(255, 255, 255, 0.1)'
      }}
    >
      {/* Top Toolbar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0.75rem 1rem',
        borderBottom: '1px solid var(--border-subtle)',
        background: 'rgba(15, 23, 42, 0.8)',
        flexWrap: 'wrap',
        gap: '0.75rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Crosshair size={16} color="var(--accent-cyan)" />
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            {title}
          </span>
        </div>

        {/* View Mode Switcher */}
        {allowModes && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <button
              onClick={() => setActiveLayer('original')}
              className={`btn ${activeLayer === 'original' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '0.35rem 0.65rem', fontSize: '0.75rem' }}
            >
              Original
            </button>
            {overlaySrc && (
              <button
                onClick={() => setActiveLayer('overlay')}
                className={`btn ${activeLayer === 'overlay' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '0.35rem 0.65rem', fontSize: '0.75rem' }}
              >
                Grad-CAM Overlay
              </button>
            )}
            {heatmapSrc && (
              <button
                onClick={() => setActiveLayer('heatmap')}
                className={`btn ${activeLayer === 'heatmap' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '0.35rem 0.65rem', fontSize: '0.75rem' }}
              >
                Heatmap
              </button>
            )}
            {annotatedSrc && (
              <button
                onClick={() => setActiveLayer('annotated')}
                className={`btn ${activeLayer === 'annotated' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '0.35rem 0.65rem', fontSize: '0.75rem' }}
              >
                Ground Truth Box
              </button>
            )}
            {overlaySrc && annotatedSrc && (
              <button
                onClick={() => setActiveLayer('side_by_side')}
                className={`btn ${activeLayer === 'side_by_side' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '0.35rem 0.65rem', fontSize: '0.75rem' }}
              >
                Side-by-Side
              </button>
            )}
          </div>
        )}

        {/* Zoom & Adjustment Controls */}
        {showControls && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button
              onClick={() => setZoom(z => Math.min(z + 0.25, 3.5))}
              className="btn btn-secondary"
              title="Zoom In"
              style={{ padding: '0.4rem', borderRadius: '6px' }}
            >
              <ZoomIn size={15} />
            </button>
            <button
              onClick={() => setZoom(z => Math.max(z - 0.25, 0.5))}
              className="btn btn-secondary"
              title="Zoom Out"
              style={{ padding: '0.4rem', borderRadius: '6px' }}
            >
              <ZoomOut size={15} />
            </button>
            <button
              onClick={resetAdjustments}
              className="btn btn-secondary"
              title="Reset Adjustments"
              style={{ padding: '0.4rem', borderRadius: '6px' }}
            >
              <RotateCcw size={15} />
            </button>
            <button
              onClick={toggleFullscreen}
              className="btn btn-secondary"
              title="Fullscreen"
              style={{ padding: '0.4rem', borderRadius: '6px' }}
            >
              {fullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
            </button>
          </div>
        )}
      </div>

      {/* Main Viewport */}
      <div style={{
        position: 'relative',
        minHeight: '380px',
        maxHeight: fullscreen ? '80vh' : '520px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        background: '#020408',
        padding: '1rem'
      }}>
        {activeLayer === 'side_by_side' ? (
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr 1fr',
            gap: '0.75rem',
            width: '100%',
            height: '100%'
          }}>
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>Original</p>
              <img
                src={originalSrc}
                alt="Original X-ray"
                style={{
                  width: '100%',
                  maxHeight: '340px',
                  objectFit: 'contain',
                  filter: `brightness(${brightness}) contrast(${contrast})`
                }}
              />
            </div>
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)', marginBottom: '0.35rem' }}>Grad-CAM Overlay</p>
              <img
                src={overlaySrc}
                alt="Grad-CAM Overlay"
                style={{
                  width: '100%',
                  maxHeight: '340px',
                  objectFit: 'contain',
                  filter: `brightness(${brightness}) contrast(${contrast})`
                }}
              />
            </div>
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '0.75rem', color: '#F87171', marginBottom: '0.35rem' }}>Ground Truth Box</p>
              <img
                src={annotatedSrc}
                alt="Ground Truth Bounding Box"
                style={{
                  width: '100%',
                  maxHeight: '340px',
                  objectFit: 'contain',
                  filter: `brightness(${brightness}) contrast(${contrast})`
                }}
              />
            </div>
          </div>
        ) : (
          <img
            src={getRenderSrc()}
            alt="Chest X-Ray"
            style={{
              maxWidth: '100%',
              maxHeight: fullscreen ? '78vh' : '480px',
              objectFit: 'contain',
              transform: `scale(${zoom})`,
              filter: `brightness(${brightness}) contrast(${contrast})`,
              transition: 'transform 0.15s ease-out, filter 0.1s ease',
              borderRadius: '6px'
            }}
          />
        )}
      </div>

      {/* Bottom Display Adjustment Sliders */}
      {showControls && (
        <div style={{
          padding: '0.75rem 1.25rem',
          borderTop: '1px solid var(--border-subtle)',
          background: 'rgba(15, 23, 42, 0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
          fontSize: '0.8rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', flex: 1, maxWidth: '500px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: 1 }}>
              <span style={{ color: 'var(--text-secondary)', minWidth: '70px' }}>Brightness:</span>
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.05"
                value={brightness}
                onChange={e => setBrightness(parseFloat(e.target.value))}
              />
              <span className="mono" style={{ minWidth: '35px', color: 'var(--accent-cyan)' }}>
                {brightness.toFixed(2)}x
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: 1 }}>
              <span style={{ color: 'var(--text-secondary)', minWidth: '60px' }}>Contrast:</span>
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.05"
                value={contrast}
                onChange={e => setContrast(parseFloat(e.target.value))}
              />
              <span className="mono" style={{ minWidth: '35px', color: 'var(--accent-cyan)' }}>
                {contrast.toFixed(2)}x
              </span>
            </div>
          </div>

          <div style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>
            Display adjustments are visualization-only and do not alter neural network input tensors.
          </div>
        </div>
      )}
    </div>
  );
}
