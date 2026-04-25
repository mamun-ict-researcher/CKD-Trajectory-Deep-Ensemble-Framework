# Multi-Horizon CKD Trajectory Prediction: A Deep Sequential & Stacked Ensemble Framework

[![Paper](https://img.shields.io/badge/IEEE-Research_Paper-blue)](https://ieeexplore.ieee.org/abstract/document/11429208)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamun-ict-researcher/CKD_Trajectory_Project/blob/main/notebooks/Trajectory_Research_Main.ipynb)

##  Research Overview
This repository contains the official implementation of our research on predicting **Chronic Kidney Disease (CKD)** progression. By integrating **Deep Sequential Modeling** with **Tree-based Ensembles**, this framework provides high-fidelity forecasting of eGFR (estimated Glomerular Filtration Rate) trajectories across 3, 6, and 12-month horizons.

This implementation serves as a methodological precursor to my PhD research, focusing on **SIVF (Semantic Integrity Verification Framework)** for software evolution.

##  Core Methodology

<div align="center">
  <img src="./assets/Fig1_Visual flow of multi-stage data preprocessing pipeline.png" width="800">
  <p><i>Fig. 1: Visual flow of the memory-efficient multi-stage data preprocessing pipeline (Python/Dask) for deriving 13 CKD features from MIMIC-IV from our IEEE publication.</i></p>
</div>

<div align="center">
  <img src="./assets/Fig2_End-to-End Methodology Pipeline.png" width="800">
  <p><i>Fig. 2: End-to-End Methodology Pipeline for Multi-Horizon eGFR Prediction from our IEEE publication.</i></p>
</div>

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
Achieving a high-fidelity **R² score of 0.778** for 12 month prediction, the system integrates **SHAP (SHapley Additive exPlanations)** to quantify feature importance, ensuring "Model Transparency" and identifying pathophysiological drivers behind predicted renal decline.

## Key Results & Validation

To validate the framework's fidelity, we compared predicted vs. actual eGFR trajectories and evaluated various architectures.

### 1. Predictive Fidelity (Actual vs. Predicted)
The following visualization demonstrates our model's ability to capture non-linear renal decline patterns accurately across the test cohort.

<div align="center">
  <img src="./outputs/Fig.%205_Predicted%20versus%20Actual%20eGFR.png" width="700">
  <p><i>Fig. 5: Predicted versus Actual eGFR for a sample patient (ID: 12596559). The plot shows how well the model follows the patient’s speed and direction of kidney function change from our IEEE publication.</i></p>
</div>

### 2. Quantitative Performance Metrics
Our Hybrid Stacked Ensemble significantly outperformed baseline models across all horizons. Detailed metrics (R², MAE, RMSE) are summarized below from our research findings.

<div align="center">
  <img src="./outputs/TABLE%20II_Model%20Performance%20on%20Test%20Set.png" width="600">
  <p><i>TABLE II: Model Performance on Test Set (3, 6, 12-Month Horizons) from our IEEE publication</i></p>
</div>

- **Interpretability:** Validated top clinical drivers (Creatinine, Blood Pressure, Age) using SHapley values.
- **Fidelity:** Successfully modeled non-linear trajectories with a significant reduction in Mean Absolute Error (MAE).


## Repository Structure
- `src/`: Modular Python source code (Data Loaders, Architectures, Trainers).
- `notebooks/`: Interactive Research Dashboard (Ready-to-run on Google Colab).
- `outputs/`: Quantitative research artifacts and trajectory visualization plots.
- `data/`: Sample data schema and metadata definitions.
- `assets/`: Visual diagram of the pipeline.

##  Technical Stack
- **Deep Learning:** TensorFlow, Keras (LSTM, Transformers)
- **ML Ensembles:** XGBoost, CatBoost, Scikit-learn
- **Optimization:** Optuna (Bayesian Optimization)
- **Interpretability:** SHAP
- **Data Engineering:** Pandas, NumPy, PyArrow

### Ethical Compliance & Data Disclaimer (MIMIC-IV)\n","This research was conducted using the **MIMIC-IV (v2.2)** database. In strict adherence to the **PhysioNet Data Use Agreement (DUA)** and HIPAA privacy regulations, original patient-level data is **NOT** shared in this repository.\n","\n","*   **Synthetic Data Note:** The execution results currently shown (e.g., $R^2 \\approx 0.99$) are generated using a **Privacy-Safe Synthetic Sample** provided in the `/data` folder. This sample is strictly for **System Pipeline Validation** (to prove the code runs to completion).\n","*   **Scientific Validation:** The high-fidelity results reported in my associated IEEE publication ($R^2 = 0.89$) were achieved on the full, non-randomized clinical cohort.\n","*   **Access:** To reproduce the original results, researchers must obtain authorized access through [PhysioNet](https://physionet.org/).

## Citation
If you find this research or codebase useful for your work, please cite our IEEE publication:

```bibtex
@inproceedings{mamun2026multi,
  title={Multi-Horizon Chronic Kidney Disease Trajectory Prediction via Interpretable Stacked Ensemble and Deep Sequential Modeling},
  author={Mamun, Abdhullah-Al and Hasan, Nusrat},
  booktitle={2026 5th International Conference on Electrical, Computer \& Telecommunication Engineering (ICECTE)},
  pages={1--6},
  year={2026},
  organization={IEEE}
}
```

## Contact & Collaboration
I am **Abdhullah-Al-Mamun**, a Senior ICT Professional (Banking/Fintech Sector) and PhD Researcher. My research focuses on **Trustworthy AI**, **Predictive Analytics**, and **Software Integrity** in mission-critical environments such as **Fintech** and **Healthcare Informatics**. I bridge the gap between 15+ years of industrial expertise in banking ICT and advanced deep learning research.


[![Google Scholar](https://shields.io)](https://scholar.google.com/citations?user=Yx7SwvIAAAAJ&hl=en) 
[![LinkedIn](https://shields.io)](https://linkedin.com) 
[![Email](https://shields.io)](mailto:khan.mamun3.14@gmail.com)

---
*Developed by **Abdhullah-Al-Mamun**, Senior ICT Professional and PhD Candidate, focusing on Trustworthy AI and Software Evolution.*
