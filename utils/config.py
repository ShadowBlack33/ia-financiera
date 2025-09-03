from __future__ import annotations
from pathlib import Path
import yaml

def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {
            "tickers": ["SPY","QQQ","AAPL","MSFT"],
            "start_date": "2015-01-01",
            "end_date": None,
            "interval": "1d",
            "data_dir": "data/raw",
            "logs_dir": "logs",
            "features": {
                "returns":"log",
                "sma_windows":[10,20,50],
                "ema_windows":[12,26],
                "rsi_windows":[14],
                "macd":{"fast":12,"slow":26,"signal":9},
                "bollinger":{"window":20,"k":2.0},
                "atr_window":14,
                "lags":[1,2,3,5],
            },
            "seed": 42,
        }
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
