# Multi-Horizon CKD Trajectory Prediction: A Deep Sequential & Stacked Ensemble Framework

[![Paper](https://img.shields.io/badge/IEEE-Research_Paper-blue)](https://ieeexplore.ieee.org/abstract/document/11429208)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[YOUR_USERNAME]/CKD_Trajectory_Project/blob/main/notebooks/Trajectory_Research_Main.ipynb)

##  Research Overview
This repository contains the official implementation of our research on predicting **Chronic Kidney Disease (CKD)** progression. By integrating **Deep Sequential Modeling** with **Tree-based Ensembles**, this framework provides high-fidelity forecasting of eGFR (estimated Glomerular Filtration Rate) trajectories across 3, 6, and 12-month horizons.

This implementation serves as a methodological precursor to my PhD research at the **Software Research Lab (SRLab)**, focusing on **SIVF (Semantic Integrity Verification Framework)** for software evolution.

##  Core Methodology

### 1. Scalable Ingestion & Metadata Synchronization
The pipeline utilizes optimized **Parquet-based storage architectures** to facilitate high-throughput ingestion of longitudinal EHR data. This ensures efficient memory mapping and metadata consistency across heterogeneous clinical sources, a skill honed through 15+ years of managing massive banking ICT infrastructures.

### 2. Robust Pre-processing & Imputation
We implement a multi-stage cleansing pipeline involving automated schema normalization and **non-parametric median imputation**. This directly addresses the "informative missingness" inherent in clinical datasets, minimizing stochastic noise in longitudinal biomarkers.

### 3. Temporal Engineering via Vectorized Sequencing
A high-performance **vectorized sliding window algorithm** transforms irregular clinical time-series into fixed-length temporal tensors. This approach captures the dynamic evolution of renal indicators while strictly preserving temporal dependencies.

### 4. Hybrid Modeling via Stacked Ensembles
The framework features a sophisticated **Stacked Ensemble** that fuses:
- **Deep Sequential Models:** LSTM (Long Short-Term Memory) and Transformers.
- **Gradient Boosted Trees:** XGBoost and CatBoost.
- **Meta-Learner:** A Linear Regression model orchestrates shared latent representations to optimize multi-horizon forecasting accuracy.

### 5. Clinically-Aligned Explainable AI (XAI)
Achieving a high-fidelity **R² score of 0.89**, the system integrates **SHAP (SHapley Additive exPlanations)** to quantify feature importance, ensuring "Model Transparency" and identifying pathophysiological drivers behind predicted renal decline.

##  Key Results
- **Performance:** Achieved an $R^2$ of **0.8943** in multi-horizon testing.
- **Fidelity:** Successfully modeled non-linear trajectories with significant reduction in Mean Absolute Error (MAE).
- **Interpretability:** Validated top clinical drivers (Creatinine, Blood Pressure, Age) using SHapley values.

## Repository Structure
- `src/`: Modular Python source code (Data Loaders, Architectures, Trainers).
- `notebooks/`: Interactive Research Dashboard (Ready-to-run on Google Colab).
- `outputs/`: Quantitative research artifacts and trajectory visualization plots.
- `data/`: Sample data schema and metadata definitions.

##  Technical Stack
- **Deep Learning:** TensorFlow, Keras (LSTM, Transformers)
- **ML Ensembles:** XGBoost, CatBoost, Scikit-learn
- **Optimization:** Optuna (Bayesian Optimization)
- **Interpretability:** SHAP
- **Data Engineering:** Pandas, NumPy, PyArrow

### Ethical Compliance & Data Disclaimer (MIMIC-IV)\n","This research was conducted using the **MIMIC-IV (v2.2)** database. In strict adherence to the **PhysioNet Data Use Agreement (DUA)** and HIPAA privacy regulations, original patient-level data is **NOT** shared in this repository.\n","\n","*   **Synthetic Data Note:** The execution results currently shown (e.g., $R^2 \\approx 0.99$) are generated using a **Privacy-Safe Synthetic Sample** provided in the `/data` folder. This sample is strictly for **System Pipeline Validation** (to prove the code runs to completion).\n","*   **Scientific Validation:** The high-fidelity results reported in my associated IEEE publication ($R^2 = 0.89$) were achieved on the full, non-randomized clinical cohort.\n","*   **Access:** To reproduce the original results, researchers must obtain authorized access through [PhysioNet](https://physionet.org/).
---
*Developed by **Abdhullah-Al-Mamun**, Senior ICT Professional and PhD Candidate, focusing on Trustworthy AI and Software Evolution.*
