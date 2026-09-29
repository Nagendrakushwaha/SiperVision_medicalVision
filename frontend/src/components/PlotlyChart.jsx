import React, { useEffect, useRef } from 'react';

export default function PlotlyChart({ data, layout, config, style, className }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !window.Plotly) return;

    const defaultLayout = {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: {
        family: 'Inter, sans-serif',
        color: '#94A3B8',
        size: 12
      },
      margin: { l: 48, r: 24, t: 36, b: 40 },
      hoverlabel: {
        bgcolor: '#1E293B',
        bordercolor: '#06B6D4',
        font: { family: 'JetBrains Mono, monospace', color: '#FFFFFF' }
      },
      ...layout
    };

    const defaultConfig = {
      responsive: true,
      displayModeBar: false,
      ...config
    };

    window.Plotly.react(containerRef.current, data, defaultLayout, defaultConfig);

    const handleResize = () => {
      if (containerRef.current && window.Plotly) {
        window.Plotly.Plots.resize(containerRef.current);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [data, layout, config]);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%', minHeight: '320px', ...style }}
      className={className}
    />
  );
}
