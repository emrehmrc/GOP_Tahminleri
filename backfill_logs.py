"""
backfill_logs.py — Faz 0 Kısmi Backfill (bir kez çalışır)
=============================================================
output/archive/*_full48h.parquet + oof_history.parquet + weather_history.parquet
-> forecast_log / actuals_log.

Doldurulan: y_pred_*, y_pred_final, y_actual, wx_*_actual, day_type/flag_*
(bunlar Datetime'dan bağımsız hesaplanır — feature_matrix snapshot'ına ihtiyaç
duymaz). DOLDURULAMAYAN (ileriye dönük yakalanmadıysa geri getirilemez):
meta_method/meta_w_*, override_*, subst_*, pv_bias_delta, wx_*_fcst,
model_versions, feature_snapshot_ref — bunlar null kalır.

Kullanım:
    python backfill_logs.py
"""

import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config_live import (
    ARCHIVE_DIR, OOF_HISTORY_PATH, WEATHER_HISTORY_PARQUET, EDAS_ID,
    RAW_TARGET_COL, RAW_DATE_COL, RAW_HOUR_COL, TEST_SIZE,
)
from src.forecast_logger import (
    FORECAST_LOG_SCHEMA, _write_typed_parquet, _forecast_log_path,
    _upsert_by_date, derive_data_quality_flags, WEATHER_STATION_TEMP_COLS,
    compute_calendar_fields,
)

ARCHIVE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_run_(.+)_full48h\.parquet$")


def backfill_forecast_log() -> dict:
    n_files = 0
    n_rows = 0
    for f in sorted(ARCHIVE_DIR.glob("*_full48h.parquet")):
        m = ARCHIVE_RE.match(f.name)
        if not m:
            continue
        issue_date, _target_str = m.groups()
        run_id = f"{issue_date}_backfill"

        df = pd.read_parquet(f)
        if "Datetime" not in df.columns:
            continue
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.sort_values("Datetime").reset_index(drop=True)

        n = len(df)
        pos = np.arange(n)
        horizon_day = np.where(pos < TEST_SIZE // 2, "T+1", "T+2")
        issue_ts = pd.Timestamp(issue_date)
        lead_time_h = ((df["Datetime"] - issue_ts) / pd.Timedelta(hours=1)).round().astype("int32")

        cal_fields = compute_calendar_fields(pd.DatetimeIndex(df["Datetime"]))

        def col(name):
            return df[name] if name in df.columns else pd.Series([None] * n)

        out = pd.DataFrame({
            "edas_id": EDAS_ID,
            "run_id": run_id,
            "config_hash": None,
            "issue_ts": issue_ts,
            "target_ts": df["Datetime"],
            "target_date": df["Datetime"].dt.strftime("%Y-%m-%d"),
            "horizon_day": horizon_day,
            "lead_time_h": lead_time_h,
            "y_pred_xgb": col("XGB_Pred"),
            "y_pred_lgbm": col("LGBM_Pred"),
            "y_pred_cat": col("CAT_Pred"),
            "y_pred_chronos": col("CHRONOS_Pred"),
            "cat_present": "CAT_Pred" in df.columns,
            "chronos_ok": None,
            "y_pred_ens_raw": None,
            "meta_method": None,
            "meta_w_xgb": None, "meta_w_lgbm": None, "meta_w_cat": None,
            "meta_w_chronos": None, "meta_intercept": None,
            "override_active": None, "override_delta": None,
            "subst_active": None, "subst_delta": None, "pv_bias_delta": None,
            "y_pred_final": col("Final_Pred"),
            "wx_temp_fcst": None, "wx_ghi_fcst": None,
            "day_type": cal_fields["day_type"].to_numpy(),
            "flag_holiday": cal_fields["flag_holiday"].to_numpy(),
            "flag_bridge": cal_fields["flag_bridge"].to_numpy(),
            "flag_ramadan": cal_fields["flag_ramadan"].to_numpy(),
            "model_versions": "{}",
            "feature_snapshot_ref": None,
        })

        for target_date, part in out.groupby("target_date"):
            path = _forecast_log_path(EDAS_ID, target_date, run_id)
            if path.exists():
                continue  # idempotent: gerçek run zaten yazmışsa üzerine yazma
            _write_typed_parquet(part, FORECAST_LOG_SCHEMA, path)
            n_rows += len(part)
        n_files += 1

    return {"status": "ok", "files": n_files, "rows": n_rows}


def backfill_actuals_from_oof() -> dict:
    if not OOF_HISTORY_PATH.exists():
        return {"status": "no_oof"}
    oof = pd.read_parquet(OOF_HISTORY_PATH)
    oof["target_ts"] = pd.to_datetime(oof["date"]) + pd.to_timedelta(oof["hour"], unit="h")
    quality = derive_data_quality_flags(oof["actual"])

    updates = pd.DataFrame({
        "target_ts": oof["target_ts"].to_numpy(),
        "y_actual": oof["actual"].to_numpy(),
        "data_quality_flag": quality.to_numpy(),
    })

    n_rows = _upsert_by_date(EDAS_ID, updates)
    return {"status": "ok", "rows": n_rows}


def backfill_actuals_weather_full(since_date: pd.Timestamp) -> dict:
    """update_actuals_log_weather'ın 30-günlük kısıtı olmayan versiyonu — ama yine
    de `since_date`'ten öncesini atlar. weather_history.parquet 2018'e kadar
    EĞİTİM verisini tutuyor; Faz 0 sadece CANLI dönemi izliyor, 8 yıllık geçmişi
    actuals_log'a (binlerce tek-günlük parquet dosyası) bindirmenin anlamı yok."""
    if not WEATHER_HISTORY_PARQUET.exists():
        return {"status": "no_weather_history"}
    wh = pd.read_parquet(WEATHER_HISTORY_PARQUET)
    wh[RAW_DATE_COL] = pd.to_datetime(wh[RAW_DATE_COL])
    wh = wh[wh[RAW_DATE_COL] >= since_date]

    temp_cols = [c for c in WEATHER_STATION_TEMP_COLS if c in wh.columns]
    if not temp_cols or "GHI_ADM_Weighted" not in wh.columns:
        return {"status": "missing_weather_cols"}

    wx_temp_actual = wh[temp_cols].mean(axis=1)
    wx_ghi_actual = wh["GHI_ADM_Weighted"]
    has_actual = wx_temp_actual.notna() | wx_ghi_actual.notna()
    if not has_actual.any():
        return {"status": "no_actuals_yet"}

    wh = wh[has_actual]
    target_ts = wh[RAW_DATE_COL] + pd.to_timedelta(wh[RAW_HOUR_COL], unit="h")
    updates = pd.DataFrame({
        "target_ts": target_ts.to_numpy(),
        "wx_temp_actual": wx_temp_actual[has_actual].to_numpy(),
        "wx_ghi_actual": wx_ghi_actual[has_actual].to_numpy(),
    })

    n_rows = _upsert_by_date(EDAS_ID, updates)
    return {"status": "ok", "rows": n_rows}


def _live_period_start() -> pd.Timestamp:
    """Canlı ürün dönemi başlangıcı: en eski arşiv run'ının issue_date'i (- birkaç gün pay)."""
    dates = []
    for f in ARCHIVE_DIR.glob("*_full48h.parquet"):
        m = ARCHIVE_RE.match(f.name)
        if m:
            dates.append(pd.Timestamp(m.group(1)))
    if not dates:
        return pd.Timestamp.today().normalize() - pd.Timedelta(days=14)
    return min(dates) - pd.Timedelta(days=3)


if __name__ == "__main__":
    print("[Backfill] forecast_log <- output/archive/*_full48h.parquet")
    print(" ", backfill_forecast_log())
    print("[Backfill] actuals_log (y_actual) <- oof_history.parquet")
    print(" ", backfill_actuals_from_oof())
    since = _live_period_start()
    print(f"[Backfill] actuals_log (wx_actual) <- weather_history.parquet (>= {since.date()}, canlı dönem)")
    print(" ", backfill_actuals_weather_full(since))

    from src.forecast_logger import rebuild_duckdb_views
    print("[Backfill] DuckDB view rebuild")
    print(" ", rebuild_duckdb_views())
