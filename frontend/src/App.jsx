import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Header from './components/Header';

// Pages
import Dashboard from './pages/Dashboard';
import AnalyzeXRay from './pages/AnalyzeXRay';
import DatasetExplorer from './pages/DatasetExplorer';
import Training from './pages/Training';
import Evaluation from './pages/Evaluation';
import ModelComparison from './pages/ModelComparison';
import Explainability from './pages/Explainability';
import Benchmarks from './pages/Benchmarks';
import Experiments from './pages/Experiments';
import Reports from './pages/Reports';
import SystemStatus from './pages/SystemStatus';
import About from './pages/About';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-container">
        <Sidebar />
        <div className="main-content">
          <Header />
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analyze" element={<AnalyzeXRay />} />
            <Route path="/dataset" element={<DatasetExplorer />} />
            <Route path="/training" element={<Training />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/comparison" element={<ModelComparison />} />
            <Route path="/explainability" element={<Explainability />} />
            <Route path="/benchmarks" element={<Benchmarks />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/system" element={<SystemStatus />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}
