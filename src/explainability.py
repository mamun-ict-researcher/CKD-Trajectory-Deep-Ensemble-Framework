"""
Project: Multi-Horizon CKD Trajectory Prediction
Module: Explainable AI (XAI) & Performance Validation
Author: Abdhullah-Al-Mamun
-------------------------------------------------------------------------
Description: 
Implements SHAP-based feature importance quantification to ensure 
clinical interpretability of predicted renal trajectories. Validates 
the R-squared and MAE results reported in the associated research.

Clinical Context: 
Focuses on identifying key biomarkers (e.g., Creatinine, BP) driving 
eGFR decline.
-------------------------------------------------------------------------
"""

import shap
import numpy as np

def run_shap_analysis(model, X_train_sample, X_test_sample):
    explainer = shap.Explainer(model.predict, X_train_sample)
    shap_values = explainer(X_test_sample)
    return shap_values
