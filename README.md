# Goose Bay Phased-Array Radar Target Validation System

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Modern%20Async-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Tactical%20HUD-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Tuned%20RBF--SVM-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Devansh%20Yaduvanshi-blueviolet.svg)](https://github.com/ItsDevannshhh)

An end-to-end Machine Learning and Digital Signal Processing (DSP) system for **Military Phased-Array Radar Target Disambiguation** using the UCI Ionosphere dataset (Goose Bay, Labrador 16-antenna high-frequency radar array).

Developed & maintained by **[Devansh Yaduvanshi](https://github.com/ItsDevannshhh)**.

---

## 🎯 Key Academic Highlights (Viva-Ready)

1. **Radar Physics Preprocessing**:
   - 17 pulse transmission lags yield 34 complex autocorrelation values ($17 \times \text{Real} + 17 \times \text{Imaginary}$).
   - **Mathematical Zero-Variance Pruning**: Column `a02` (Lag 0 Imaginary) is mathematically $\equiv 0.0$ across all 351 instances ($\sigma^2 = 0.000000$). Dropping `a02` prevents Hessian singularity and stabilizes radial kernel SVM decision boundaries.
2. **Domain Feature Engineering**:
   - `mean_pulse_amplitude`: Mean envelope $\mu(\sqrt{Re^2 + Im^2})$ across all 17 lags.
   - `std_pulse_amplitude`: Pulse envelope variance over transmission time.
   - `lag_smoothness`: Mean absolute lag-to-lag jump $\mu(|\Delta Re|)$ (structural returns decay smoothly, electromagnetic clutter jumps erratically).
3. **Multi-Model Benchmark & Optimization**:
   - Evaluated 5 architectures: Tuned RBF-SVM (👑 **97.2% Accuracy, 0.988 ROC-AUC**), Random Forest, k-NN, Logistic Regression, and MLP Neural Network.

---

## 🛠️ Architecture & Interfaces

- **Modern Web Interface (`frontend/workflow.html`)**:
  - 4-Node Live Canvas: Radar Input $\rightarrow$ DSP Jamming $\rightarrow$ Physics Extractor $\rightarrow$ Model Router.
  - Functional $360^\circ$ Military Radar HUD with rotating beam ($18\text{ RPM}$), range rings, and Web Audio API sonar ping.
  - Viva Testing Deck for unseen data (presets, raw 34-value paste, CSV dropzone).
- **FastAPI Backend (`backend/main.py`)**:
  - Real-time `/api/validate` inference endpoint.
  - Telemetry and dataset CSV export endpoints.
- **Streamlit Tactical HUD (`app.py`)**:
  - Scifi Glass-Brutalism dashboard featuring Plotly 3D PCA cluster projections, confusion matrix, and ROC curves.

---

## 🚀 Quickstart

### 1. Clone & Setup Environment
```bash
git clone https://github.com/ItsDevannshhh/radar-target-detector.git
cd radar-target-detector
```

#### Create & Activate Virtual Environment:
- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  Set-ExecutionPolicy -Scope Process Bypass -Force; .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train & Evaluate Models
```bash
python train_and_evaluate.py
```
*Generates EDA plots, benchmarks 5 models, runs GridSearchCV on RBF-SVM, and exports `model.pkl` and `neural_network_pipeline.pkl`.*

### 4. Launch the Modern Web Dashboard & API (FastAPI)
```bash
uvicorn backend.main:app --reload --port 8000
```
- Open [http://localhost:8000/workflow.html](http://localhost:8000/workflow.html) for the interactive 4-node pipeline & radar HUD.
- Open [http://localhost:8000/evaluation.html](http://localhost:8000/evaluation.html) for model metrics & ROC curves.
- Open [http://localhost:8000/index.html](http://localhost:8000/index.html) for 3D PCA cluster analysis.

### 5. Launch the Streamlit Tactical HUD (Alternative)
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Benchmark Results

| Model Architecture | 5-Fold CV Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (Tuned RBF)** 👑 | **94.64%** | **97.18%** | **96.8%** | **98.4%** | **0.976** | **0.988** |
| **Random Forest (100t)** | 93.21% | 94.37% | 93.9% | 96.0% | 0.949 | 0.975 |
| **Neural Network (MLP)** | 91.79% | 92.96% | 92.0% | 94.0% | 0.930 | 0.952 |
| **k-NN (k=3)** | 86.43% | 88.73% | 89.1% | 91.2% | 0.901 | 0.924 |
| **Logistic Regression** | 86.07% | 87.32% | 86.5% | 90.0% | 0.882 | 0.912 |

---

## 📂 Project Directory Structure

```plaintext
radar-target-detector/
├── backend/
│   └── main.py                     # FastAPI REST API & physics validation engine
├── frontend/
│   ├── workflow.html               # 4-Node visual canvas & 360° radar HUD
│   ├── index.html                  # 3D PCA eigen-projection & dataset matrix
│   ├── evaluation.html             # Multi-model ROC studio & confusion matrix
│   ├── pca_dataset.html            # Standalone PCA analysis view
│   └── studio.html                 # Signal generator & test studio
├── 1_radar_waveform_comparison.png # 17-lag real vs noise autocorrelation plot
├── 2_pca_clusters.png              # 2D PCA cluster visualization
├── 3_roc_curves.png                # Multi-model ROC comparison
├── 4_confusion_matrix_svm.png      # Tuned SVM confusion matrix heatmap
├── app.py                          # Streamlit tactical military HUD
├── ionosphere.csv                  # UCI Ionosphere raw dataset (351 rows x 35 columns)
├── ionosphere_svm_pipeline.pkl     # Tuned RBF-SVM trained pipeline
├── model.pkl                       # Production serialized model
├── neural_network_pipeline.pkl     # Trained MLP neural network pipeline
├── requirements.txt                # Python package dependencies
├── train_and_evaluate.py           # End-to-end ML pipeline & model trainer
└── README.md                       # Comprehensive documentation
```

---

## 👤 Author

**Devansh Yaduvanshi**
- GitHub: [@ItsDevannshhh](https://github.com/ItsDevannshhh)

## 📄 License

This project is licensed under the [MIT License](LICENSE).
