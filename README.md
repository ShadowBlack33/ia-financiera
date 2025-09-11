# 💹 IA-FINANCIERA · ETL → Machine Learning → KPIs & Visuals
![CI](https://github.com/ShadowBlack33/ia-financiera/actions/workflows/ci.yml/badge.svg)

> **Author**: Carlos Andrés Orozco Caicedo · Colombia 🇨🇴  
> End-to-end financial ML pipeline with ETL, technical indicators, ensemble modeling, and dashboards — production-ready with CI/CD.

---

## 🌐 Overview

**IA-FINANCIERA** is a complete AI pipeline for directional prediction in financial markets. It extracts and transforms OHLCV time series data, computes technical indicators, trains ensemble models, ranks assets based on upward probability (PROBA_UP), and displays results via a dashboard and heatmaps. The project is reproducible, modular, and integrated with GitHub Actions (CI).

---

## 🎯 Goal

- Build a clean ETL → ML pipeline to predict market direction (UP/DOWN).
- Generate ensemble-based probability scores (LogReg + RandomForest).
- Visualize results in heatmaps, ranked lists, and a dashboard.
- Provide reproducible, modular, and CI-enabled workflow.

---

## ✨ Features

- 🔁 ETL from Yahoo Finance via `yfinance`
- 📈 Technical indicators: RSI, MACD, SMA, EMA, Bollinger Bands, ATR
- 🤖 Ensemble classifier (Logistic Regression + Random Forest)
- 📊 Ranked signals: Top-N bullish and bearish tickers
- 🖼️ Probability heatmap (PNG)
- 🖥️ Streamlit dashboard (optional)
- 📄 Outputs in CSV, PNG, and colored console
- 🧪 Optional backtesting per asset
- ✅ GitHub Actions with deterministic CI backtest and smoke test
- 📦 Professional project structure (etl/, models/, images/, reports/, etc.)

---

## 🖼️ Preview

<p align="center">
  <img src="images/heatmap_probs.png" alt="Probability Heatmap" width="75%" />
</p>

> Generated via `scripts/plot_heatmap.py` using `models/prob_summary.csv`

---

## 🏗️ Architecture

```mermaid
flowchart LR
  A[yfinance tickers] --> B[etl/extract.py]
  B --> C[etl/transform.py]
  C --> D[etl/load.py]
  D --> R[data/raw/*.csv]
  R --> E[models/train_all.py]
  R --> F[models/train_direction.py]
  F --> S[models/prob_summary.csv]
  F --> T[models/traces/*.csv]
  S --> H[images/heatmap_probs.png]
  S --> W[apps/dashboard_app.py]
````

---

## 📦 Star Schema

```mermaid
erDiagram
  SIGNALS {
    datetime datetime
    string   ticker
    float    proba_logreg
    float    proba_rf
    float    proba_ens
    string   pred
  }

  TICKER {
    string ticker
    string asset_class
  }

  DATE {
    datetime datetime
    int      year
    int      month
    int      day
  }

  MODEL {
    string name
    string type
    string notes
  }

  SIGNALS ||--o{ TICKER : has
  SIGNALS ||--o{ DATE   : occurs_on
  SIGNALS ||--o{ MODEL  : generated_by
```

---

## 📚 Data Dictionary

| Column         | Type   | Description                        |
| -------------- | ------ | ---------------------------------- |
| `ticker`       | string | Asset symbol (e.g., AAPL, BTC-USD) |
| `date`         | date   | Prediction date                    |
| `proba_logreg` | float  | Logistic Regression UP probability |
| `proba_rf`     | float  | Random Forest UP probability       |
| `proba_ens`    | float  | Ensemble average probability       |
| `pred`         | string | Final prediction: `UP` or `DOWN`   |

---

## ⚙️ Installation & Usage

```bash
# Clone and navigate
git clone https://github.com/ShadowBlack33/ia-financiera.git
cd ia-financiera

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run full pipeline via menu
python menu.py
```

---

## 🧾 Configuration (config.yaml)

```yaml
start_date: "2015-01-01"
end_date: ""          # empty = up to latest
interval: "1d"
default_top_n: 10

features:
  rsi:
    periods: [14]
  macd:
    fast: 12
    slow: 26
    signal: 9
  sma:
    windows: [10, 20, 50, 200]
  ema:
    windows: [12, 26]
  bollinger:
    window: 20
    n_std: 2
  atr:
    window: 14
```

---

## 📊 KPIs & Model Details

| Metric          | Description                               |
| --------------- | ----------------------------------------- |
| PROBA\_UP       | Probability of upward movement (ensemble) |
| Top‑N Ranking   | Sort by `proba_ens` (bullish / bearish)   |
| CSV Summary     | `models/prob_summary.csv`                 |
| Heatmap         | `images/heatmap_probs.png`                |
| Optional Traces | `models/traces/*.csv` per ticker          |

* Hold-out split on last `N` bars (e.g., 200)
* Optional embargo window to prevent leakage

---

## 🖥️ Dashboard (optional)

```bash
streamlit run apps/dashboard_app.py
```

Includes:

* Top‑N ranking
* Probability heatmap (logreg, RF, ensemble)
* Summary table

---

## 📈 Backtest (optional)

```bash
python -m models.backtest
```

Outputs:

* CSVs in `reports/`
* Per-ticker traces in `models/traces/`

---

## 🔁 Reproducibility & CI

* `config.yaml` controls features, intervals, date range
* `.gitignore` excludes unnecessary outputs
* **GitHub Actions** (CI) includes:

  * 🔹 Deterministic backtest: `models/backtest_ci.py`
  * 🔹 Smoke test: `tests/test_smoke.py`
  * 🔹 CI badge: ![CI](https://github.com/ShadowBlack33/ia-financiera/actions/workflows/ci.yml/badge.svg)

Artifact: `backtest-ci-report.csv` (available in each CI run)

---

## 🛠️ Troubleshooting

| Issue                        | Solution                                                 |
| ---------------------------- | -------------------------------------------------------- |
| Mermaid not rendered         | Use fenced code blocks with `mermaid` syntax             |
| Heatmap not generated        | Run `menu.py` first and check `.gitignore` exceptions    |
| Streamlit dashboard error    | Ensure `models/prob_summary.csv` exists                  |
| CI fails on pandas           | Only lightweight tests run in CI (test\_smoke.py)        |
| API or rate limit from Yahoo | Reduce date range or increase retries (built-in support) |

---

## 🗺️ Roadmap

* [ ] Add support for XGBoost and LightGBM
* [ ] Walk-forward cross-validation
* [ ] Backtest metrics: Sharpe ratio, drawdown, win rate
* [ ] Streamlit Cloud / HuggingFace deployment
* [ ] Model registry and versioning

---

## 📜 License

**MIT License © 2025 — Carlos Andrés Orozco Caicedo**

> This project is for educational and research purposes only.
> **It does not constitute financial advice. Use at your own risk.**

---

## 👤 Author

**Carlos Andrés Orozco Caicedo**
📍 Colombia · Data Engineering & AI Student

GitHub: [ShadowBlack33](https://github.com/ShadowBlack33)
Project Owner and Developer — ETL, ML models, visualizations, CI/CD pipeline.

````