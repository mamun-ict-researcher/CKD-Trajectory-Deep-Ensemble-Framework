# Trajectory Modeling with Uncertainty-Aware Deep Ensembles
(Applied to Chronic Kidney Disease Progression)

[![Paper](https://img.shields.io/badge/IEEE-Research_Paper-blue)](https://ieeexplore.ieee.org/abstract/document/11429208)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamun-ict-researcher/CKD_Trajectory_Project/blob/main/notebooks/Trajectory_Research_Main.ipynb)

##  Research Overview
This repository presents an uncertainty-aware deep ensemble framework for longitudinal trajectory modeling, applied to Chronic Kidney Disease (CKD) progression using eGFR time-series data.

While the application domain is healthcare, the core contribution of this work lies in modeling complex temporal systems under uncertainty, which is directly transferable to other domains involving evolving entities, such as software systems, code evolution, and semantic change detection.

## Novel Contribution

This study empirically shows that uncertainty-aware ensembles provide more reliable predictions than single models under irregular temporal sampling, particularly in sparse observation regimes.This work goes beyond standard CKD prediction by focusing on **robust longitudinal modeling under uncertainty**, addressing limitations in existing deep learning approaches that rely solely on point estimates.

### Key Contributions

1. **Uncertainty-Aware Longitudinal Modeling**

   * Unlike conventional CKD prediction models that output deterministic predictions, this framework incorporates **deep ensemble-based uncertainty estimation** to quantify prediction confidence across time.
   * This enables identification of **high-risk prediction regions**, particularly under sparse or irregular observations.

2. **Multi-Horizon Trajectory Forecasting**

   * Instead of single-step prediction, the model performs **multi-horizon forecasting**, capturing the *evolution pattern* of CKD progression rather than static outcomes.
   * This provides a richer representation of temporal dynamics and aligns with real-world progression analysis.

3. **Robustness to Irregular Time-Series**

   * The framework is designed to handle **non-uniform temporal intervals**, a common issue in real-world longitudinal datasets.
   * This improves generalization compared to models assuming fixed time steps.

4. **Ensemble-Based Stability Analysis**

   * The use of deep ensembles enables analysis of **prediction variance across models**, offering insights into model reliability rather than only accuracy.

---

### Research Insight

This work demonstrates that:

> *Modeling uncertainty in temporal systems is as important as modeling the signal itself.*

This insight is directly transferable to other domains involving evolving entities, such as **software systems**, where:

* Code evolution resembles temporal trajectories
* Semantic drift corresponds to uncertain deviations
* Reliable prediction requires calibrated confidence, not just accuracy

This forms the methodological foundation for future research on **uncertainty-aware semantic change detection in evolving software systems**.


##  Research Positioning

This work serves as a **foundational study** for my proposed PhD research:

> *“Towards Robust and Scalable Semantic Change Detection: An Uncertainty-Aware Multi-View Framework for Evolving Software”*

The methodological parallels are as follows:

| This Work (Healthcare)      | PhD Direction (Software Engineering)         |
| --------------------------- | -------------------------------------------- |
| Patient trajectory modeling | Code evolution trajectory modeling           |
| eGFR progression            | Semantic drift across versions               |
| Deep ensemble learning      | Multi-view representation learning           |
| Predictive uncertainty      | Uncertainty calibration for semantic changes |
| Longitudinal time-series    | Version history dynamics                     |

This project demonstrates my ability to design **robust, uncertainty-aware models for sequential data**, which I aim to extend to **semantic analysis of evolving software systems**.

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

## Methodological Transferability

The techniques developed in this work are **domain-agnostic** and directly applicable to software engineering research:

* Temporal models → **Code evolution sequence modeling**
* Ensemble learning → **Multi-view fusion of code representations (AST, embeddings, behavior)**
* Uncertainty estimation → **Confidence scoring for semantic drift detection**
* Multi-horizon prediction → **Future defect or regression forecasting**

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

### Ethical Compliance & Data Disclaimer (MIMIC-IV)
This research was conducted using the **MIMIC-IV (v2.2)** database. In strict adherence to the **PhysioNet Data Use Agreement (DUA)** and HIPAA privacy regulations, original patient-level data is **NOT** shared in this repository.

**Synthetic Data Note:** The execution results currently shown (e.g., $R^2 \\approx 0.99$) are generated using a **Privacy-Safe Synthetic Sample** provided in the `/data` folder. This sample is strictly for **System Pipeline Validation** (to prove the code runs to completion).   
**All reported scientific conclusions are based on real clinical data (IEEE publication). Synthetic results are included solely for pipeline reproducibility.**

**Scientific Validation:** The high-fidelity results reported in my associated IEEE publication ($R^2 = 0.89$) were achieved on the full, non-randomized clinical cohort.

**Access:** To reproduce the original results, researchers must obtain authorized access through [PhysioNet](https://physionet.org/).

## Limitations

- Performance degrades under extremely sparse patient observations
- Ensemble uncertainty may be miscalibrated in highly imbalanced cohorts
- Temporal windowing may lose long-term dependencies

These limitations motivate future work in more robust temporal representation learning.

## Industry Context

With over **15 years of experience in banking and mission-critical systems**, I have observed that:

* Real-world systems evolve continuously
* Small structural changes can lead to significant behavioral impact
* Lack of uncertainty awareness often leads to overlooked risks

This motivated my interest in **robust, interpretable, and uncertainty-aware modeling**, bridging **industry challenges and academic research**.

## Future Work

Building on this foundation, future research will focus on:

* Multi-view representation learning for code (AST, embeddings, execution behavior)
* Semantic change detection across software versions
* Uncertainty calibration for developer decision support
* Cognitive alignment in code review processes

## Relevance to PhD Research

This repository demonstrates:

* Ability to design **end-to-end machine learning systems**
* Experience with **longitudinal and sequential data modeling**
* Understanding of **uncertainty quantification**
* Capability to translate **real-world problems into research frameworks**

These skills directly support my proposed research in **semantic analysis of evolving software systems**.

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


<p align="left">
  <a href="https://scholar.google.com/citations?user=Yx7SwvIAAAAJ&hl=en">
    <img src="https://img.shields.io/badge/Google%20Scholar-Profile-4285F4?logo=google-scholar&logoColor=white"/>
  </a>
  <a href="https://linkedin.com/in/YOUR_ID">
    <img src="https://img.shields.io/badge/LinkedIn-Profile-0A66C2?logo=linkedin&logoColor=white"/>
  </a>
  <a href="mailto:khan.mamun3.14@gmail.com">
    <img src="https://img.shields.io/badge/Email-Contact-D14836?logo=gmail&logoColor=white"/>
  </a>
</p>

---
*Developed by **Abdhullah-Al-Mamun**, Senior ICT Professional and PhD Candidate, focusing on Trustworthy AI and Software Evolution.*
