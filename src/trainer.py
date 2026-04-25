"""
Project: Multi-Horizon CKD Trajectory Prediction
Module: Model Training & Optimization Pipeline
Author: Abdhullah-Al-Mamun
-------------------------------------------------------------------------
Description: 
Orchestrates the training lifecycle of the ensemble framework. Includes 
multi-horizon loss optimization and Optuna-based Bayesian hyperparameter 
tuning to minimize MAE and maximize forecasting fidelity.

Status: Configured for full-cohort training and pipeline validation.
-------------------------------------------------------------------------
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import math

class StackingManager:
    def __init__(self):
        self.meta_learner = LinearRegression()
    def train_meta(self, X_meta, y_true):
        self.meta_learner.fit(X_meta, y_true)
    def predict(self, X_meta):
        return self.meta_learner.predict(X_meta)

def calculate_metrics(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred, multioutput='uniform_average'),
        'rmse': math.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }
