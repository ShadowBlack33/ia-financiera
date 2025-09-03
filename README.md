# 📈 IA-FINANCIERA — Predicción de dirección de precios financieros / Financial Price Direction Prediction

Este proyecto implementa un pipeline completo de ETL, entrenamiento, análisis y visualización para predecir la **dirección del mercado (SUBE/BAJA)** usando datos históricos de tickers financieros.

> ✅ Proyecto académico y profesional para portafolio en Ingeniería de Datos e Inteligencia Artificial.

---

This project implements a full ETL, training, analysis, and visualization pipeline to predict the **financial market direction (UP/DOWN)** using historical ticker data.

> ✅ Academic and professional project for a Data Engineering & AI portfolio.

---

## 🧠 Tecnologías / Technologies Used

- **Python 3.10+**
- **scikit-learn** (Logistic Regression, Random Forest, ensemble)
- **yfinance** for financial data extraction
- **matplotlib / seaborn / plotly** for visualization
- **streamlit** for dashboards
- **pandas / numpy / statsmodels** for analysis
- **config.yaml** for reproducibility and configuration
- 🔁 **Random seed control** for reproducible results

---

## 📊 Visuals

**Heatmap de probabilidades (última corrida):**
![Heatmap](images/heatmap_probs.png)

**Equity por señales del clasificador:**
![Backtest Equities](reports/backtest_traces_equity.png)

### 📈 Diagrama de flujo de datos (pipeline)
```mermaid
flowchart LR
    A[Usuario] -->|Fechas & Guardar CSV| M[menu.py]

    subgraph ETL
      M --> E[etl/extract.py]
      E --> T[etl/transform.py]
      T --> L[etl/load.py]
      L --> R1[data/raw/*.csv]
    end

    subgraph Entrenamiento
      R1 --> REG[models/train_all.py<br/>Regresión (RMSE/MAE + preds)]
      R1 --> CLS[models/train_direction.py<br/>Clasificación (proba_up)]
      REG --> P[data/preds/*.csv]
      CLS --> S1[models/prob_summary.csv]
      CLS --> TR[models/traces/*_trace.csv]
    end

    subgraph Salidas
      S1 -->|Top-N, tabla| D1[apps/dashboard_app.py]
      S1 -->|Heatmap| IMG[images/heatmap_probs.png]
      P --> BT[models/backtest.py]
      TR --> BT
      BT --> REP[reports/*.csv, *.png]
    end
```

---

## 📁 Estructura / Project Structure

```bash
ia-financiera/
├── data/                  ← Datos crudos y predicciones
├── models/                ← Entrenamiento y backtesting
├── etl/                   ← Extracción y transformación
├── scripts/               ← Gráficas y utilidades
├── apps/                  ← Dashboard interactivo (streamlit)
├── reports/               ← Métricas y salidas
├── images/                ← Visualizaciones generadas
├── utils/                 ← Configuración y limpieza de logs
├── menu.py                ← 🎯 Pipeline principal
├── config.yaml            ← Parámetros globales
├── requirements.txt       ← Dependencias
└── README.md              ← Este archivo
```

---

## 🚀 ¿Cómo ejecutar el proyecto? / How to run the project

```bash
git clone https://github.com/ShadowBlack33/ia-financiera.git
cd ia-financiera

python -m venv .venv
.venv\Scripts\activate        # en Windows
# source .venv/bin/activate    # en Linux/Mac

pip install -r requirements.txt

python menu.py
```

📌 El sistema te preguntará:
- Fecha inicio y fin
- Si deseas guardar el resumen (`models/prob_summary.csv`)

---

## 📊 Dashboard interactivo / Interactive Dashboard

```bash
streamlit run apps/dashboard_app.py
```

Incluye:
- Top‑N alcistas y bajistas
- Mapa de calor (LogReg / RF / Ensemble)
- Tabla detallada por ticker

---

## 📈 Backtest de señales / Signal Backtesting

```bash
python -m models.backtest
```

Genera:
- `backtest_preds_summary.csv`
- `backtest_traces_summary.csv`
- `backtest_traces_equity.png`

---

## 👤 Autor / Author

**Carlos Andrés Orozco Caicedo**  
Proyecto académico y profesional de Ingeniería de Datos e Inteligencia Artificial 🇨🇴  
Academic & professional project — Data Engineering & AI — Colombia

---