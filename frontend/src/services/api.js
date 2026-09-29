import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 120000, // 2 minutes timeout for CPU evaluations/training ops
});

// System & Health
export const getHealth = () => api.get('/health').then(res => res.data);
export const getSystemStatus = () => api.get('/system').then(res => res.data);

// Dataset
export const getDatasetSummary = () => api.get('/dataset/summary').then(res => res.data);
export const getDatasetSplit = () => api.get('/dataset/split').then(res => res.data);
export const getDatasetExamples = (params) => api.get('/dataset/examples', { params }).then(res => res.data);
export const getRandomExample = (params) => api.get('/dataset/random', { params }).then(res => res.data);
export const getPatientDetails = (patientId) => api.get(`/dataset/patient/${patientId}`).then(res => res.data);

// Models
export const listModels = () => api.get('/models').then(res => res.data);
export const getModel = (modelName) => api.get(`/models/${modelName}`).then(res => res.data);
export const setActiveModel = (modelName) => api.post('/models/active', { model_name: modelName }).then(res => res.data);

// Training
export const startTraining = (config) => api.post('/training/start', config).then(res => res.data);
export const stopTraining = () => api.post('/training/stop').then(res => res.data);
export const getTrainingStatus = () => api.get('/training/status').then(res => res.data);
export const getTrainingHistory = (modelName) => api.get('/training/history', { params: { model_name: modelName } }).then(res => res.data);
export const getVisualProcessing = (params) => api.get('/training/visualize-processing', { params }).then(res => res.data);


// Inference & Predictions
export const predictImage = (formData) => api.post('/predict', formData, {
  headers: { 'Content-Type': 'multipart/form-data' }
}).then(res => res.data);

export const predictPatient = (patientId, modelName = 'resnet18', includeGradcam = true) => 
  api.post('/predict/patient', {
    patient_id: patientId,
    model_name: modelName,
    include_gradcam: includeGradcam
  }).then(res => res.data);

export const getPredictionHistory = () => api.get('/prediction/history').then(res => res.data);
export const clearPredictionHistory = () => api.post('/prediction/clear').then(res => res.data);

// Evaluation
export const getEvaluation = (modelName = 'resnet18') => api.get('/evaluation', { params: { model_name: modelName } }).then(res => res.data);
export const runEvaluation = (config) => api.post('/evaluation/run', config).then(res => res.data);

// Benchmarks
export const listBenchmarks = () => api.get('/benchmarks').then(res => res.data);
export const runBenchmark = (config) => api.post('/benchmarks', config).then(res => res.data);

// Experiments
export const listExperiments = () => api.get('/experiments').then(res => res.data);

// Technical Reports
export const getReport = (modelName = 'resnet18') => api.get('/reports', { params: { model_name: modelName } }).then(res => res.data);

export default api;
