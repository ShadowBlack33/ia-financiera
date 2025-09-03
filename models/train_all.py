from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def _load_file(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df

def _feature_cols(df: pd.DataFrame, target: str) -> List[str]:
    drop = {"Datetime","Ticker","Interval"}
    cols = [c for c in df.columns if c not in drop]
    return [c for c in cols if c != target]

def _make_target(df: pd.DataFrame, target: str, horizon: int) -> pd.Series:
    if target == "ret":
        return df["ret"].shift(-horizon)
    else:
        return df["Close"].shift(-horizon)

def _walk_splits(n: int, test_size: int=200, embargo: int=5):
    start = max(test_size + embargo, int(n*0.5))
    splits = []
    while start < n - test_size:
        tr_idx = np.arange(0, start - embargo)
        te_idx = np.arange(start, start + test_size)
        splits.append((tr_idx, te_idx))
        start += test_size
    if not splits:
        te = np.arange(max(0, n-test_size), n)
        tr = np.arange(0, max(0, n-test_size))
        splits.append((tr, te))
    return splits

def run_for_file(
    p: Path,
    metrics: List[dict],
    target="ret",
    horizon=1,
    test_size=200,
    embargo=5,
    save_preds: bool=False,
    preds_dir: Path=Path("data/preds")
):
    df = _load_file(p)
    y = _make_target(df, target, horizon=horizon)
    X_cols = _feature_cols(df, target)
    data = df[X_cols].copy()
    data["y"] = y
    data = data.dropna().copy()
    if len(data) < 300: return
    X = data[X_cols].values; yv = data["y"].values
    idx = data.index.values
    models = {
        "linreg": Pipeline([("scaler", StandardScaler()), ("m", LinearRegression())]),
        "rf": RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=42),
        "svr": Pipeline([("scaler", StandardScaler()), ("m", SVR(C=10.0, epsilon=0.001))]),
    }
    preds_dir.mkdir(parents=True, exist_ok=True)
    split_id = 0
    for tr, te in _walk_splits(len(data), test_size=test_size, embargo=embargo):
        xtr, xte = X[tr], X[te]; ytr, yte = yv[tr], yv[te]
        dte = df.loc[idx[te], ["Datetime","Ticker"]].reset_index(drop=True)
        stack = []
        for name, model in models.items():
            model.fit(xtr, ytr)
            pred = model.predict(xte)
            rmse = mean_squared_error(yte, pred, squared=False)
            mae = mean_absolute_error(yte, pred)
            metrics.append({"file": p.name, "ticker": p.stem.split("_")[0], "model": name, "split": split_id, "n_train": len(tr), "n_test": len(te), "rmse": rmse, "mae": mae})
            if save_preds:
                out = dte.copy(); out["y_true"] = yte; out["y_pred"] = pred
                out.to_csv(preds_dir / f"{p.stem}_{name}_split{split_id}.csv", index=False)
            stack.append(pred)
        if stack and save_preds:
            ens = np.column_stack(stack).mean(axis=1)
            out = dte.copy(); out["y_true"] = yte; out["y_pred"] = ens
            # (FIX) guardar correctamente el CSV del ensemble
            out.to_csv(preds_dir / f"{p.stem}_ensemble_weighted_split{split_id}.csv", index=False)
        split_id += 1

def run_for_folder(
    folder: str="data/raw",
    pattern: str="*_1d.csv",
    metrics_out: str="models/metrics_full.csv",
    target: str="ret",
    horizon: int=1,
    test_size: int=200,
    embargo: int=5,
    save_preds: bool=False
) -> str:
    folder = Path(folder)
    metrics = []
    for p in sorted(folder.glob(pattern)):
        try:
            run_for_file(p, metrics, target=target, horizon=horizon, test_size=test_size, embargo=embargo, save_preds=save_preds, preds_dir=Path("data/preds"))
        except Exception:
            pass
    out = Path(metrics_out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(out, index=False)
    return str(out)
