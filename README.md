# 💹 IA-FINANCIERA · ETL → Machine Learning → KPIs & Visuals
![CI](https://github.com/ShadowBlack33/ia-financiera/actions/workflows/ci.yml/badge.svg)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e)](https://scikit-learn.org/)
[![streamlit](https://img.shields.io/badge/Streamlit-Dashboard-E64A19)](https://streamlit.io/)

**Proyecto completo de IA financiera** con pipeline ETL, clasificación direccional (UP/DOWN) y visualizaciones.  
Incluye procesamiento de datos financieros, entrenamiento de modelos, ranking de activos, heatmap de probabilidades y dashboard interactivo.

> 📈 Datos obtenidos desde Yahoo Finance (`yfinance`).  
> ⚠️ Este proyecto es educativo — **no representa asesoría financiera.**

---

## 🧭 Tabla de contenido

- [🎯 Objetivo](#-objetivo)
- [🖼️ Vista previa](#-vista-previa)
- [✨ Funcionalidades](#-funcionalidades)
- [🏗️ Arquitectura](#-arquitectura)
- [🧱 Esquema Estrella](#-esquema-estrella)
- [📚 Diccionario de datos](#-diccionario-de-datos)
- [⚙️ Instalación y uso](#️-instalación-y-uso)
- [🧾 Configuración YAML](#-configuración-yaml)
- [📊 KPIs y detalles del modelo](#-kpis-y-detalles-del-modelo)
- [🖥️ Dashboard](#-dashboard)
- [📈 Backtest](#-backtest)
- [🔁 Reproducibilidad y CI](#-reproducibilidad-y-ci)
- [🛠️ Troubleshooting](#-troubleshooting)
- [🗺️ Roadmap](#-roadmap)
- [📜 Licencia](#-licencia)
- [👤 Autor](#-autor)

---

## 🎯 Objetivo

- Desarrollar un pipeline ETL + ML para predecir la **dirección del mercado** (UP/DOWN).
- Usar ensamble de modelos para obtener `PROBA_UP` (probabilidad de alza).
- Visualizar y rankear los activos más alcistas/bajistas.
- Ofrecer resultados en consola, CSV, PNG y dashboard.

---

## 🖼️ Vista previa

<p align="center">
  <img src="images/heatmap_probs.png" alt="Heatmap de probabilidades" width="70%">
</p>

> Generado por `scripts/plot_heatmap.py`

---

## ✨ Funcionalidades

- 🔁 **ETL**: descarga y transforma series OHLCV.
- 📈 **Indicadores**: RSI, MACD, SMA, EMA, Bollinger Bands, ATR.
- 🤖 **Modelos**: Logistic Regression + Random Forest → Ensemble.
- 📊 **Top‑N ranking**: calls (alcistas) / puts (bajistas).
- 🌡️ **Heatmap visual**: probabilidades por ticker.
- 🖥️ **Dashboard interactivo** con Streamlit.
- 🧪 **Backtest opcional** y métricas por archivo.
- ✅ **CI determinístico** (backtest + test mínimo en GitHub Actions).

---

## 🏗️ Arquitectura

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

## 🧱 Esquema Estrella

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

## 📚 Diccionario de datos

**`models/prob_summary.csv`**

| Campo          | Tipo   | Descripción                           |
| -------------- | ------ | ------------------------------------- |
| `ticker`       | string | Símbolo del activo (AAPL, BTC, etc.)  |
| `date`         | date   | Fecha de predicción                   |
| `proba_logreg` | float  | Probabilidad de subida (LogReg)       |
| `proba_rf`     | float  | Probabilidad de subida (RandomForest) |
| `proba_ens`    | float  | Promedio del ensamble                 |
| `pred`         | string | Predicción final: UP / DOWN           |

---

## ⚙️ Instalación y uso

```bash
# Clonar y entrar al repo
git clone https://github.com/ShadowBlack33/ia-financiera.git
cd ia-financiera

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/macOS

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el pipeline completo
python menu.py
```

---

## 🧾 Configuración YAML

**config.yaml**

```yaml
start_date: "2015-01-01"
end_date: ""          # vacío = hasta hoy
interval: "1d"

data_dir: "data/raw"
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

## 📊 KPIs y detalles del modelo

* **PROBA\_UP**: probabilidad estimada de alza (ensamble).
* **Top-N**: activos con mayor o menor probabilidad.
* **Split**: hold-out temporal (`test_size=200`, `horizon=1`).
* **Embargo**: opcional para prevenir leakage (ej. 5 velas).

---

## 🖥️ Dashboard

```bash
streamlit run apps/dashboard_app.py
```

Incluye:

* Ranking Top‑N
* Heatmap de probabilidades
* Tabla resumen

---

## 📈 Backtest

```bash
python -m models.backtest
```

> Salida en `reports/` y trazas por ticker en `models/traces/`.

---

## 🔁 Reproducibilidad y CI

* 🎯 `config.yaml` para hiperparámetros y rutas.
* 📦 `.gitignore` evita subir archivos generados innecesarios.
* 🧪 **CI en GitHub Actions** con:

  * `models/backtest_ci.py` (backtest determinístico)
  * `tests/test_smoke.py` (test mínimo)
* 📁 Artifacts del run: `backtest_ci_report.csv`

---

## 🛠️ Troubleshooting

| Problema                      | Solución                                                   |
| ----------------------------- | ---------------------------------------------------------- |
| Mermaid no se ve en GitHub    | Usar bloque `mermaid` y evitar comentarios `//`            |
| Heatmap no aparece            | Asegurar que no esté ignorado en `.gitignore`              |
| `pandas` no encontrado en CI  | Se excluyen tests pesados usando CI determinístico         |
| Streamlit no encuentra el CSV | Ejecutar `menu.py` primero para generar `prob_summary.csv` |
| Falla `yfinance` o timeout    | Reintentar con fechas más cortas o conexión estable        |

---

## 🗺️ Roadmap

* [ ] Agregar XGBoost y LightGBM
* [ ] Backtesting más robusto con métricas tipo Sharpe
* [ ] Despliegue del dashboard (Streamlit Cloud o Render)
* [ ] Walk-forward CV y calibration
* [ ] Registro de modelos por versión

---

## 📜 Licencia

**MIT License © 2025 — Carlos Andrés Orozco Caicedo**

> Este proyecto es educativo.
> **No constituye asesoría financiera ni promueve decisiones de inversión.**

---

## 👤 Autor

**Carlos Andrés Orozco Caicedo**
*Data Engineering & Machine Learning · Colombia 🇨🇴*

GitHub: [ShadowBlack33](https://github.com/ShadowBlack33)
Proyecto original: ETL + ML + CI + Visualizaciones para predicción direccional.

---