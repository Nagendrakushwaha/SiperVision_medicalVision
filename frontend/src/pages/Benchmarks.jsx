import React, { useState, useEffect } from 'react';
import { Zap, Cpu, Play, RotateCcw, Award, Clock } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import PlotlyChart from '../components/PlotlyChart';
import { listBenchmarks, runBenchmark } from '../services/api';

export default function Benchmarks() {
  const [benchmarks, setBenchmarks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [benchmarking, setBenchmarking] = useState(false);
  const [targetModel, setTargetModel] = useState('resnet18');
  const [runs, setRuns] = useState(30);

  const fetchBenchmarks = () => {
    setLoading(true);
    listBenchmarks()
      .then(res => {
        setBenchmarks(res.benchmarks || []);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchBenchmarks();
  }, []);

  const handleRunBenchmark = async () => {
    setBenchmarking(true);
    try {
      await runBenchmark({
        model_name: targetModel,
        num_runs: runs
      });
      fetchBenchmarks();
    } catch (err) {
      alert('Benchmark failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setBenchmarking(false);
    }
  };

  // Bar Chart of Latencies
  const latencyData = benchmarks.length > 0 ? [
    {
      x: benchmarks.map(b => b.model_name),
      y: benchmarks.map(b => b.avg_latency_ms),
      name: 'Average Latency (ms)',
      type: 'bar',
      marker: { color: '#06B6D4' }
    },
    {
      x: benchmarks.map(b => b.model_name),
      y: benchmarks.map(b => b.p95_latency_ms),
      name: 'P95 Latency (ms)',
      type: 'bar',
      marker: { color: '#EF4444' }
    }
  ] : [];

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">CPU Inference Benchmarking</h1>
          <p className="page-subtitle">
            Empirical latency profiling & throughput quantification on AMD Ryzen 5 5500U CPU
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <select
            className="form-select"
            value={targetModel}
            onChange={e => setTargetModel(e.target.value)}
          >
            <option value="resnet18">ResNet18</option>
            <option value="mobilenet">MobileNetV3-Small</option>
            <option value="efficientnet">EfficientNet-B0</option>
          </select>

          <select
            className="form-select"
            value={runs}
            onChange={e => setRuns(parseInt(e.target.value))}
          >
            <option value="15">15 Runs (Quick)</option>
            <option value="30">30 Runs (Standard)</option>
            <option value="50">50 Runs (High Precision)</option>
          </select>

          <button
            onClick={handleRunBenchmark}
            disabled={benchmarking}
            className="btn btn-primary"
          >
            {benchmarking ? (
              <>
                <RotateCcw size={16} className="animate-spin" />
                Measuring CPU Latency...
              </>
            ) : (
              <>
                <Play size={16} fill="currentColor" />
                Run Benchmark
              </>
            )}
          </button>
        </div>
      </div>

      {/* Top Level Cards if benchmark exists */}
      {benchmarks.length > 0 && (
        <div className="metrics-grid">
          <MetricCard
            title="Fastest Model Throughput"
            value={`${Math.max(...benchmarks.map(b => b.throughput_fps || 0)).toFixed(1)} FPS`}
            subtext="Real CPU inferences / sec"
            icon={Zap}
            color="cyan"
          />
          <MetricCard
            title="Median CPU Latency (P50)"
            value={`${Math.min(...benchmarks.map(b => b.p50_latency_ms || 0)).toFixed(1)} ms`}
            subtext="AMD Ryzen 5 5500U"
            icon={Cpu}
            color="green"
          />
          <MetricCard
            title="Smallest Footprint"
            value={`${Math.min(...benchmarks.map(b => b.model_size_mb || 0)).toFixed(1)} MB`}
            subtext="Memory footprint"
            icon={Award}
            color="indigo"
          />
        </div>
      )}

      {/* Benchmarks Master Table */}
      <div className="glass-card" style={{ marginBottom: '2rem', padding: '0', overflow: 'hidden' }}>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Device</th>
                <th>Avg Latency</th>
                <th>P50 Latency</th>
                <th>P95 Latency</th>
                <th>Throughput</th>
                <th>Parameters</th>
                <th>Model Size</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.length === 0 ? (
                <tr>
                  <td colSpan="9" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                    No benchmarks recorded yet. Select an architecture and click "Run Benchmark".
                  </td>
                </tr>
              ) : (
                benchmarks.map((b, idx) => (
                  <tr key={idx}>
                    <td>
                      <strong style={{ color: 'var(--text-primary)' }}>{b.model_name}</strong>
                    </td>
                    <td>
                      <span className="mono" style={{ fontSize: '0.78rem', color: 'var(--accent-cyan)' }}>
                        {b.device}
                      </span>
                    </td>
                    <td className="mono" style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
                      {b.avg_latency_ms} ms
                    </td>
                    <td className="mono">{b.p50_latency_ms} ms</td>
                    <td className="mono" style={{ color: '#F87171' }}>{b.p95_latency_ms} ms</td>
                    <td className="mono" style={{ color: '#10B981', fontWeight: 600 }}>
                      {b.throughput_fps} FPS
                    </td>
                    <td className="mono" style={{ fontSize: '0.8rem' }}>
                      {b.parameters?.toLocaleString()}
                    </td>
                    <td className="mono" style={{ fontSize: '0.8rem' }}>
                      {b.model_size_mb} MB
                    </td>
                    <td>
                      <span className="badge badge-normal">{b.status}</span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Latency Comparison Chart */}
      {benchmarks.length > 0 && (
        <div className="glass-card">
          <h3 style={{ marginBottom: '1rem' }}>Average vs P95 Latency (Lower is Better)</h3>
          <PlotlyChart
            data={latencyData}
            layout={{
              barmode: 'group',
              yaxis: { title: 'Latency (Milliseconds)', gridcolor: 'rgba(255,255,255,0.06)' },
              height: 300,
              legend: { orientation: 'h', y: -0.2 }
            }}
          />
        </div>
      )}
    </div>
  );
}
