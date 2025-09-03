from __future__ import annotations
import sys
import logging
from pathlib import Path
from utils.config import load_config
from utils.logging_cfg import setup_logging
from utils.log_cleanup import cleanup_logs  
from etl.extract import fetch_tickers
from etl.transform import transform_frame
from etl.load import save_csv_idempotent
# Regresión
from models.train_all import run_for_folder as run_regression_folder
# Clasificación direccional 
from models.train_direction import run_folder as run_classif_folder

# Intervalos válidos 
VALID_INTERVALS = {
    "1m","2m","5m","15m","30m","60m","90m","1h","4h",
    "1d","5d","1wk","1mo","3mo"
}

PRESETS = {
    "Principales (SPY, QQQ, índices y mega-caps)": [
        "SPY","QQQ","DIA","IWM","AAPL","MSFT","TSLA","NVDA",
        "AMZN","GOOGL","META","GLD","TLT","UUP","USO"
    ],
    "Mega-cap Tech": ["AAPL","MSFT","TSLA","NVDA","AMZN","GOOGL","META"],
    "ETFs Macro": ["SPY","QQQ","DIA","IWM","GLD","TLT","UUP","USO","HYG"],
    "Cripto (YF)": ["BTC-USD","ETH-USD"],
}

def _input(prompt: str, default: str | None = None) -> str:
    s = input(f"{prompt}{' ['+default+']' if default else ''}: ").strip()
    return s if s else (default or "")

def _yesno(prompt: str, default: bool = False) -> bool:
    d = "s" if default else "n"
    s = input(f"{prompt} (s/n) [{d}]: ").strip().lower()
    if not s:
        s = d
    return s == "s"

def _all_tickers_from_presets() -> list[str]:
    seen = set()
    out: list[str] = []
    for lst in PRESETS.values():
        for t in lst:
            u = t.upper().strip()
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    return out

def run_everything_once(cfg, logger):
    print("\n======== IA-FINANCIERA — EJECUCIÓN AUTOMÁTICA ========")

    # Preguntas mínimas
    start = _input("Fecha inicio YYYY-MM-DD", cfg.get("start_date","2015-01-01"))
    end   = _input("Fecha fin YYYY-MM-DD (Enter=hoy)", cfg.get("end_date","")) or None
    save_csv = _yesno("¿Guardar CSV resumen de probabilidades?", default=True)

    # Parámetros globales
    interval = (cfg.get("interval","1d") or "1d").lower()
    if interval not in VALID_INTERVALS:
        interval = "1d"

    top_n = int(cfg.get("default_top_n", 10))
    data_dir = Path(cfg.get("data_dir","data/raw")); data_dir.mkdir(parents=True, exist_ok=True)
    features = cfg.get("features",{})

    # Universos de mercados 
    tickers = _all_tickers_from_presets()
    print(f"\nMercados a procesar ({len(tickers)}): {', '.join(tickers)}")

    # ETL con reintentos
    failed, ok = [], 0
    for t in tickers:
        try:
            logger.info(f"[ETL] {t} {start}->{end} @ {interval}")
            df_raw = fetch_tickers([t], start=start, end=end, interval=interval)
            if df_raw is None or len(df_raw) == 0:
                print(f"  ⚠ Sin datos {t}")
                failed.append(t)
                continue
            df_t = df_raw if "Ticker" not in df_raw.columns else df_raw[df_raw["Ticker"] == t].copy()
            df_tf = transform_frame(df_t, features_cfg=features, ticker=t)
            if "Interval" not in df_tf.columns:
                df_tf["Interval"] = interval
            out_path = data_dir / f"{t}_{interval}.csv"
            save_csv_idempotent(df_tf, out_path, dedupe_keys=["Datetime","Ticker"])
            print(f"  ✅ {out_path}")
            ok += 1
        except Exception as e:
            logger.exception(f"ETL falló {t}")
            print(f"  ❌ {t}: {e}")
            failed.append(t)

    if failed:
        print(f"\n↻ Reintentando tickers fallidos ({len(failed)}): {', '.join(failed)}")
        still = []
        for t in failed:
            try:
                df_raw = fetch_tickers([t], start=None, end=None, interval=interval)  
                if df_raw is None or len(df_raw) == 0:
                    print(f"  ⚠ Sin datos tras reintento {t}")
                    still.append(t)
                    continue
                df_t = df_raw if "Ticker" not in df_raw.columns else df_raw[df_raw["Ticker"] == t].copy()
                df_tf = transform_frame(df_t, features_cfg=features, ticker=t)
                if "Interval" not in df_tf.columns:
                    df_tf["Interval"] = interval
                out_path = data_dir / f"{t}_{interval}.csv"
                save_csv_idempotent(df_tf, out_path, dedupe_keys=["Datetime","Ticker"])
                print(f"  ✅ {out_path} (reintento)")
                ok += 1
            except Exception as e:
                logger.exception(f"ETL reintento falló {t}")
                print(f"  ❌ {t} reintento: {e}")
                still.append(t)
        if still:
            print(f"  ⚠ Tickers sin datos tras reintentos: {', '.join(still)}")

    if ok == 0:
        print("No hubo CSVs transformados. Abortando.")
        return

    # Entrenamiento de regresión 
    print("\n→ Entrenando regresión (silencioso, guardando predicciones)...")
    run_regression_folder(
        folder=str(data_dir),
        target="ret",
        horizon=1,
        embargo=5,
        save_preds=True   
    )
    print("  ✅ Métricas regresión: models/metrics_full.csv")

    # Entrenamiento de clasificación (Top‑N + colores + CSV opcional + trazas)
    print("\n→ Entrenando clasificación (Top‑N + CSV opcional + trazas)...")
    run_classif_folder(
        folder=data_dir,
        pattern=f"*_{interval}.csv",
        horizon=1,
        initial_train=None,
        test_size=200,
        top_n=top_n,
        save_summary=save_csv,
        summary_path=Path("models/prob_summary.csv"),
        print_summary=True,
        save_trace=True,                   
        trace_dir=Path("models/traces")
    )
    if save_csv:
        print("  ✅ Resumen: models/prob_summary.csv")

def main():
    setup_logging()
    logger = logging.getLogger("financial_project")

    # 🔁 Limpieza/rotación de logs (conserva los 5 más recientes)
    cleanup_logs(keep=5)

    try:
        cfg = load_config("config.yaml")
    except Exception as e:
        print(f"❌ No pude cargar config.yaml: {e}")
        return

    # Ejecuta una sola vez el pipeline completo
    run_everything_once(cfg, logger)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido")
        sys.exit(1)
