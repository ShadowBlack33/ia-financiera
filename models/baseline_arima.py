"""Baseline ARIMA per ticker using statsmodels."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Tuple
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import logging

def train_eval_arima(csv_path: str | Path, target: str="Close", order: Tuple[int,int,int]=(1,1,1), test_size: float=0.2) -> dict:
    p = Path(csv_path)
    df = pd.read_csv(p, parse_dates=["Datetime"])
    df = df.sort_values("Datetime")
    y = df[target].astype(float).values
    n = len(y)
    split = int(n*(1-test_size))
    y_train, y_test = y[:split], y[split:]
    model = ARIMA(y_train, order=order)
    res = model.fit()
    forecast = res.forecast(steps=len(y_test))
    mae = mean_absolute_error(y_test, forecast)
    rmse = mean_squared_error(y_test, forecast, squared=False)
    out = {"file": str(p), "mae": float(mae), "rmse": float(rmse), "n": int(n), "order": str(order)}
    logging.info(f"ARIMA {order} | {p.name} | MAE={mae:.4f} RMSE={rmse:.4f}")
    return out

def run_for_folder(folder: str | Path, pattern: str="*_1d.csv", metrics_out: str | Path="models/metrics.csv"):
    folder = Path(folder)
    rows = []
    for f in folder.glob(pattern):
        try:
            rows.append(train_eval_arima(f))
        except Exception as e:
            logging.warning(f"Failed on {f}: {e}")
    if rows:
        Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(metrics_out, index=False)
        logging.info(f"Saved metrics to {metrics_out}")
