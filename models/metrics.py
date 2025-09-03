
"""Evaluation metrics for regression on price/returns."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    # Directional accuracy if both are arrays
    try:
        acc = (np.sign(np.diff(np.r_[y_true[0], y_true])) == np.sign(np.diff(np.r_[y_pred[0], y_pred]))).mean()
    except Exception:
        acc = np.nan
    return {"mae": float(mae), "rmse": float(rmse), "directional_acc": float(acc)}
