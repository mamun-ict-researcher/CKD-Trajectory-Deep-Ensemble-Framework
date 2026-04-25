import shap
import numpy as np

def run_shap_analysis(model, X_train_sample, X_test_sample):
    explainer = shap.Explainer(model.predict, X_train_sample)
    shap_values = explainer(X_test_sample)
    return shap_values