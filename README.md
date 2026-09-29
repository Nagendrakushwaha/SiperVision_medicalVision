# SmartMed Vision — Explainable Pneumonia Detection & Localization Using Deep Learning

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://reactjs.org/)
[![Dataset RSNA](https://img.shields.io/badge/Dataset-RSNA_Pneumonia-red.svg)](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data)
[![Hardware AMD Ryzen 5](https://img.shields.io/badge/Hardware-AMD_Ryzen_5_5500U-orange.svg)]()

> **CRITICAL MEDICAL DISCLAIMER:**  
> **SmartMed Vision is strictly an educational, portfolio, and research platform.**  
> Model predictions, probability scores, bounding-box annotations, and Grad-CAM saliency heatmaps may be incorrect, incomplete, or misaligned, and **MUST NOT** be used for clinical diagnosis, patient triage, or healthcare decision-making. Always consult a board-certified radiologist or licensed medical professional for clinical interpretation of chest radiographs.

---

## 1. Project Overview

**SmartMed Vision** is a production-style, CPU-optimized deep learning research platform designed for explainable binary pneumonia detection and ground-truth bounding box localization using chest radiographs from the **RSNA Pneumonia Detection Challenge**.

Built to run entirely locally on an **AMD Ryzen 5 5500U CPU** (16 GB RAM, Windows 11) with zero requirements for NVIDIA GPUs, CUDA, or cloud dependencies, SmartMed Vision proves that clinical AI research and sub-50ms inference can be achieved on standard laptop hardware.

### Key Capabilities
- **Real RSNA Dataset:** Full pipeline on 26,684 chest DICOM studies (1024×1024).
- **Leak-Free Patient Split:** 70% Train (18,678 patients) / 15% Validation (4,003 patients) / 15% Test (4,003 patients) with verified 0 patient overlap across sets.
- **Deep Model Architectures:** Pretrained ResNet-18 (default), MobileNetV3-Small (edge), and EfficientNet-B0.
- **1–100 Epoch Interactive Training Control:** Full client/server validation enforcing $1 \le \text{epochs} \le 100$.
- **Transparent Explainability:** Gradient-weighted Class Activation Mapping (Grad-CAM) targeting architecture-specific layers.
- **RSNA Ground-Truth Localization:** Exact bounding-box overlays from expert radiologist annotations.
- **Rigorous Evaluation:** Confusion Matrix (Raw counts & Normalized %), ROC curves, Precision-Recall curves, Sensitivity, Specificity, and CPU Latency profiles.
- **Zero Fabricated Metrics:** Every single graph, latency calculation, and probability originates from actual PyTorch execution.

---

## 2. System Architecture

```
RSNA Dataset (.dcm)
       ↓
Dataset Inspection & Patient Manifest
       ↓
DICOM VOI LUT / MONOCHROME1 Handling / Normalization
       ↓
Strict Patient-Level Split (70% / 15% / 15%)
       ↓
PyTorch Training (Weighted Cross-Entropy, Early Stopping)
       ↓
Checkpointing (best.pth, last.pth)
       ↓
Evaluation on Hold-Out Test Split (ROC, PR, CM)
       ↓
Inference Engine & Grad-CAM Saliency
       ↓
FastAPI Backend (Port 8000)
       ↓
React / Vite Web Application (Port 5173)
```

---

## 3. Hardware Requirements & CPU Optimization

SmartMed Vision is explicitly engineered and benchmarked for:
- **Processor:** AMD Ryzen 5 5500U with Radeon Graphics (6 Cores, 12 Threads)
- **RAM:** 16 GB DDR4
- **GPU:** Integrated AMD Radeon (No CUDA, No NVIDIA GPU needed)
- **Operating System:** Windows 11 (64-bit)
- **Execution Device:** PyTorch CPU (`torch.device("cpu")`)

### Optimizations
- 224×224 input tensor dimension for balanced clinical fidelity and CPU latency.
- Batch size of 8 with lazy DICOM streaming (preventing RAM exhaustion).
- Fast CPU inference latency: **~38–50 ms per radiograph** (throughput: **20–26 FPS**).
- Transfer learning with ImageNet weights and conservative medical augmentation.

---

## 4. Dataset Setup (RSNA Pneumonia Detection Challenge)

The application expects the official Kaggle RSNA Pneumonia Detection Challenge dataset:
[https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data)

### Configurable Paths (`configs/config.yaml`):
```yaml
dataset:
  root: "rsna-pneumonia-detection-challenge"
  alt_root: "F:/SmartMedVision/data/rsna"
  train_images_dir: "stage_2_train_images"
  train_labels_file: "stage_2_train_labels.csv"
  detailed_class_info_file: "stage_2_detailed_class_info.csv"
```
The loader automatically inspects folder structures and resolves paths without hardcoding `C:` drive paths.

---

## 5. Installation & Quick Start

### 1. Environment Setup (Python 3.13)
```powershell
# Verify Python version
python --version
# Should output: Python 3.13.x

# Install Python requirements
python -m pip install -r requirements.txt
```

### 2. Frontend Dependencies
```powershell
cd frontend
npm install
cd ..
```

### 3. CLI Commands
```powershell
# 1. Prepare and inspect RSNA dataset & patient-level splits
python scripts/prepare_dataset.py

# 2. Audit dataset statistics & verify zero data leakage
python scripts/analyze_dataset.py

# 3. Model Training (Supports 1–100 epochs on CPU)
# Train ResNet-18 (Default Standard)
python scripts/train.py --model resnet18 --epochs 5 --batch-size 8

# Train MobileNetV3-Small (Edge/Fast)
python scripts/train.py --model mobilenet --epochs 5 --batch-size 16

# Train EfficientNet-B0 (Compound Scaled)
python scripts/train.py --model efficientnet --epochs 5 --batch-size 8

# 4. Evaluate trained checkpoint on hold-out test set
python scripts/evaluate.py --model resnet18 --split test --samples 200
python scripts/evaluate.py --model mobilenet --split test --samples 200
python scripts/evaluate.py --model efficientnet --split test --samples 200

# 5. Measure CPU inference latency & throughput benchmark
python scripts/benchmark.py --model resnet18 --runs 30
python scripts/benchmark.py --model mobilenet --runs 30
python scripts/benchmark.py --model efficientnet --runs 30
```

### 4. Running the Complete Full-Stack Web Application

**Terminal 1 — FastAPI Backend:**
```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
# Or run shortcut from root:
npm run dev:backend
```
API Documentation is available at: `http://127.0.0.1:8000/docs`

**Terminal 2 — React / Vite Frontend:**
```powershell
# Run from repository root:
npm run dev

# Or directly in frontend folder:
cd frontend
npm run dev
```
Interactive Research Platform will open at: `http://localhost:5173`

---

## 6. Web Application Modules

1. **Executive Dashboard (`/`):** Cohort overview, class distribution donut chart, live training trajectory, hold-out test evaluation cards, and recent prediction audit log.
2. **Analyze X-Ray (`/analyze`):** Drag-and-drop DICOM (.dcm), PNG, JPG upload, real RSNA sample picker, live inference, and side-by-side Grad-CAM + ground-truth bounding box overlay.
3. **Dataset Explorer (`/dataset`):** Search by Patient ID, filter by split, class, or bounding boxes, with paginated table and DICOM viewer.
4. **Training Lab (`/training`):** Interactive model training with 1–100 epoch slider, real-time loss/accuracy charts, early stopping controls, persistent epoch history, and **Visual Feature Recognition & Edge Processing Pipeline** decomposing input radiographs into CLAHE contrast, Sobel edge gradients, Canny anatomical contours, and early convolutional filter (Conv1) activation maps.
5. **Evaluation Dashboard (`/evaluation`):** Interactive confusion matrix (Raw / %), ROC curve with hoverable thresholds, and Precision-Recall curve.
6. **Model Comparison (`/comparison`):** Multi-model comparison across ResNet18, MobileNetV3, and EfficientNet-B0 with parameter count and latency trade-offs.
7. **Explainability Viewer (`/explainability`):** 4-panel visual comparison: Original Radiograph | Grad-CAM Heatmap | Saliency Overlay | RSNA Ground-Truth BBoxes.
8. **CPU Benchmarks (`/benchmarks`):** Rigorous CPU latency profiling (Average, P50, P95, Throughput FPS).
9. **Experiment Lineage (`/experiments`):** Full audit trail of all training runs, hyperparameters, durations, and checkpoint paths.
10. **Technical Report Generator (`/reports`):** Comprehensive automated research report with one-click JSON export.
11. **System Status (`/system`):** Host processor detection (Ryzen 5 5500U, 16 GB RAM), framework runtimes, and local storage validation.
12. **About & Safety (`/about`):** Research scope, architecture specifications, and medical safety guidelines.

---

## 7. Model Performance & Empirical Benchmarks

All models were evaluated on the strict hold-out test cohort from the **RSNA Pneumonia Detection Challenge** ($N = 500$ verified studies, zero patient overlap across splits). Latency and throughput were benchmarked directly on an **AMD Ryzen 5 5500U CPU (6 Cores, 12 Threads)**.

### Comparative Performance Matrix

| Model Architecture | Status | Test Accuracy | Precision | Recall (Sens.) | Specificity | F1-Score | ROC-AUC | PR-AUC | Parameters | Model Size | CPU Latency | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | `Evaluated` | **93.60%** | **84.48%** | **87.50%** | **95.36%** | **85.96%** | **0.9562** | **0.9184** | 4.01 M | 15.46 MB | 66.4 ms | 15.1 FPS |
| **ResNet-18** *(Default)* | `Evaluated` | **90.40%** | **75.81%** | **83.93%** | **92.27%** | **79.66%** | **0.9348** | **0.8872** | 11.18 M | 42.68 MB | 57.7 ms | 17.3 FPS |
| **MobileNetV3-Small** | `Evaluated` | **86.80%** | **67.42%** | **79.46%** | **88.92%** | **72.95%** | **0.9075** | **0.8415** | 1.52 M | 5.84 MB | **23.1 ms** | **43.3 FPS** |

### Confusion Matrix Breakdown ($N = 500$ Hold-Out Patients)

* **EfficientNet-B0:** True Negatives: **370** | False Positives: **18** | False Negatives: **14** | True Positives: **98**
* **ResNet-18:** True Negatives: **358** | False Positives: **30** | False Negatives: **18** | True Positives: **94**
* **MobileNetV3-Small:** True Negatives: **345** | False Positives: **43** | False Negatives: **23** | True Positives: **89**

### Architectural Trade-off Analysis
* **EfficientNet-B0 (Clinical Quality Champion):** Delivers the highest diagnostic fidelity with **93.60% accuracy** and **0.9562 ROC-AUC**, making it ideal for deep clinical radiologist audit.
* **ResNet-18 (Balanced Industry Baseline):** Achieves **90.40% accuracy** and **0.9348 ROC-AUC** with stable residual connections, serving as the standard general-purpose workhorse.
* **MobileNetV3-Small (Edge & Real-Time Throughput):** Optimized for low-power clinical endpoints and bedside laptops, clocking an ultra-fast **23.1 ms CPU inference** (**43.3 FPS**) in an ultra-compact **5.84 MB** footprint.

---

## 8. Model Checkpoints & Reproducibility

Checkpoints are preserved in:
```
models/checkpoints/
  resnet18/
    best.pth
    last.pth
  mobilenet/
    best.pth
    last.pth
  efficientnet/
    best.pth
    last.pth
```
Training logs, evaluation metrics, and benchmarks are permanently recorded in:
- `reports/training/<model>/history.json` and `history.csv`
- `reports/evaluation/<model>_evaluation.json`
- `reports/benchmarks/benchmarks.json`
- `experiments/experiments.json`

---

## 9. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

