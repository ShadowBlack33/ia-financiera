# 💹 IA-FINANCIERA · ETL → Machine Learning → KPIs & Visuals

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e)](https://scikit-learn.org/)
[![streamlit](https://img.shields.io/badge/Streamlit-Dashboard-E64A19)](https://streamlit.io/)

Project that implements a full ML pipeline to extract and transform financial data, compute technical indicators, train classification models, generate UP/DOWN predictions and produce KPIs, visualizations and dashboards.

> ✅ All predictions and visuals are computed from the transformed dataset.  
> 📌 The raw data comes from Yahoo Finance via yfinance.

---

## 🧭 Table of Contents
- [Goal](#goal)
- [Preview](#preview)
- [Features](#features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Install & Run](#install--run)
- [Backtest](#backtest)
- [Dashboard](#dashboard)
- [Credits](#credits)

---

## 🎯 Goal

- Build an ML pipeline for financial signal classification.
- Predict **probability of upward movement** (PROBA_UP) using ensemble models.
- Compute KPIs and generate charts for **Top-N bullish/bearish signals**.
- Visualize results with **heatmaps**, **tables**, and **dashboards**.
- Enable reproducibility and reusability with modular structure.

---

## 🖼️ Preview

<p align="center">
  <img src="images/heatmap_probs.png" alt="Heatmap Probabilities" width="47%" />
  <img src="reports/backtest_traces_equity.png" alt="Backtest Equity Curve" width="47%" />
</p>

> Visuals are generated automatically from model outputs.

---

## ✨ Features

- 🔁 Full ETL: download, transform, deduplicate and enrich OHLCV time series.
- 🧠 Technical indicators: RSI, MACD, EMA, SMA, Bollinger Bands, ATR...
- 🔍 Classification: Logistic Regression + Random Forest + Ensemble
- 📈 Top-N Calls & Puts (UP/DOWN predictions)
- 📊 Heatmap & Summary CSV
- 📤 Backtest & Metrics
- 🧪 Logging + Seed control
- ⚡ Streamlit dashboard (interactive)

---

## 🏗️ Architecture

### Data Flow

```mermaid
flowchart LR
  A["Tickers: yfinance"]
  B["Extract"]
  C["Transform"]
  D["Clean CSVs"]
  E["Train Regressor"]
  F["Train Classifier"]
  G["Top-N & Prob Summary"]
  H["Trace Logs"]
  I["Heatmap / Charts"]
  J["Dashboard"]

  A --> B --> C --> D
  D --> E --> I
  D --> F --> G --> J
  F --> H
```

---

## 🧱 Star Schema – Market Signal Model

```mermaid
erDiagram
  SIGNALS ||--o{ TICKER : has
  SIGNALS ||--o{ DATE : on
  SIGNALS ||--o{ MODEL : generated_by

  SIGNALS {
    datetime datetime PK
    ticker string PK
    proba_logreg float
    proba_rf float
    proba_ens float
    prediction string
  }

  TICKER {
    ticker string PK
    asset_class string
  }

  DATE {
    datetime datetime PK
    year int
    month int
    day int
  }

  MODEL {
    name string PK
    type string
    description string
  }
```

---

## 📁 Repository Structure

```
ia-financiera/
├─ data/                 # raw & transformed market data
├─ models/               # training + prediction + backtest
├─ etl/                  # extract, transform, load
├─ reports/              # CSV summaries, backtest outputs
├─ images/               # charts (heatmap, topN, equity curve)
├─ apps/                 # Streamlit dashboard
├─ scripts/              # utility scripts (heatmap, topN)
├─ utils/                # config, logging, seed tools
├─ menu.py               # 🔁 main entrypoint (ETL + Train + Predict)
├─ config.yaml           # configuration
└─ README.md             # this file
```

---

## ⚙️ Install & Run

```bash
# 1. Clone
git clone https://github.com/ShadowBlack33/ia-financiera.git
cd ia-financiera

# 2. Create environment
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run pipeline (ETL → Train → Predict)
python menu.py
```

---

## 📈 Backtest

```bash
python -m models.backtest
```

Generates:
- `reports/backtest_preds_summary.csv`
- `reports/backtest_traces_summary.csv`
- `reports/backtest_traces_equity.png`

---

## 📊 Dashboard

```bash
streamlit run apps/dashboard_app.py
```

Visuals:
- 🔼 Top-N Bullish
- 🔽 Top-N Bearish
- 🟡 Probability Heatmap
- 🧾 Summary Table

---

## 👤 Credits

**Carlos Andrés Orozco Caicedo**  
`IA-FINANCIERA` — Machine Learning · Finance · Python · Dashboard  
🇨🇴 Universidad & Portfolio Project — 2025

---