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

## ⚙️ Funcionalidades / Features

- Extrae y transforma datos históricos (`SPY`, `QQQ`, `AMZN`, `BTC-USD`, etc.)
- Aplica indicadores técnicos: RSI, MACD, SMA, Bollinger, ATR...
- Entrena modelos de:
  - Regresión (retornos esperados)
  - Clasificación direccional (probabilidad de subida)
- Genera KPIs y visualizaciones:
  - Top-N alcistas y bajistas (calls / puts)
  - Tabla de probabilidades y señales
  - Mapa de calor de modelos
- Dashboard interactivo (`streamlit`)
- Backtesting básico de señales

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
git clone https://github.com/tuusuario/ia-financiera.git
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

## 📊 Visualización y análisis / Visualization & Analysis

### 🔥 Dashboard interactivo / Interactive Dashboard

```bash
streamlit run apps/dashboard_app.py
```

Incluye / Includes:
- Top‑N alcistas y bajistas
- Mapa de calor (LogReg / RF / Ensemble)
- Tabla detallada por ticker

### 📈 Backtest de señales / Signal Backtesting

```bash
python -m models.backtest
```

Genera / Generates:
- `backtest_preds_summary.csv`
- `backtest_traces_summary.csv`
- `backtest_traces_equity.png`

---

## ✅ Estado del proyecto / Project Status

| Módulo / Module     | Estado / Status |
|---------------------|------------------|
| ETL                 | ✅ Completo / Complete |
| Regresión / Regression | ✅ |
| Clasificación / Classification | ✅ |
| Top‑N + Probabilidades | ✅ |
| Heatmap + Dashboard | ✅ |
| Backtesting         | ✅ |
| Logging + Seeds     | ✅ |
| Publicación / Repo listo | ✅ |

---

## 👤 Autor / Author

**Carlos Andrés Orozco Caicedo**  
Proyecto académico y profesional de Ingeniería de Datos e Inteligencia Artificial 🇨🇴  
Academic & professional project — Data Engineering & AI — Colombia

---