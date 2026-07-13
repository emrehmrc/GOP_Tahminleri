"""
forecast_logger.py — ADM ince shim (Faz 2, 2026-07-10)
=============================================================
Şema kaynağı: stlf_forecast_log_tasarim.md §2 (forecast_log), §3 (actuals_log).

Paylaşımlı mantık (rebuild_duckdb_views, heal_forecast_log_gaps, schema,
compute_calendar_fields, upsert yardımcıları, backup) artık
monitoring/forecast_logger.py'de yaşıyor (ADM+GDZ ortak — bkz. o paketin
docstring'i). Burada SADECE ADM'ye ÖZGÜ kalan iki şey var:

  1. write_forecast_log() — postprocessed_predictions.parquet'in ADM'ye özgü
     kolon şeması (XGB_Pred/LGBM_Pred doğrudan, GDZ'nin T1/T2 ayrı-kolon
     coalesce'i YOK) + raw_predictions_meta.json/manifest.json sidecar'ları.
  2. update_actuals_log() / update_actuals_log_weather() — ADM'nin
     Tarih+Saat ayrı kolon şeması (GDZ'de tekil Tarih datetime).

Diğer her şey (FORECAST_LOG_SCHEMA, _write_typed_parquet, _forecast_log_path,
compute_calendar_fields, rebuild_duckdb_views, heal_forecast_log_gaps,
backup_logs_zip) monitoring/'den re-export edilir — eski çağıranlar
(backfill_logs.py, asof_regen.py, perfect_prog_rerun.py, run_daily.py,
ui/tab_tahmin_uret.py) HİÇBİR DEĞİŞİKLİK gerektirmez, imzalar birebir aynı.

Çağrı sırası:
  04_predict_48h  -> compute_calendar_fields(), compute_horizon_fields() (yardımcı)
  05_postprocess  -> (kendi delta hesaplarını yapar, forecast_logger'a dokunmaz)
  run_daily/UI    -> write_forecast_log(ctx)      (06'dan sonra)
  01_ingest       -> update_actuals_log(...)      (D+1 yük dalgası)
  02_fetch_weather-> update_actuals_log_weather()  (~D+6 hava dalgası)
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config_live as C
from monitoring import forecast_logger as _shared
from monitoring.schema import FORECAST_LOG_SCHEMA, ACTUALS_LOG_SCHEMA

log = logging.getLogger("adm_live")

DATA_DIR = C.DATA_DIR
RAW_TARGET_COL = C.RAW_TARGET_COL
RAW_DATE_COL = C.RAW_DATE_COL
RAW_HOUR_COL = C.RAW_HOUR_COL

RAW_PREDICTIONS_PATH = DATA_DIR / "weather_cache" / "raw_predictions.parquet"
RAW_PREDICTIONS_META_PATH = DATA_DIR / "weather_cache" / "raw_predictions_meta.json"
POSTPROC_PATH = DATA_DIR / "weather_cache" / "postprocessed_predictions.parquet"

WEATHER_STATION_TEMP_COLS = [f"{s}_app_temp_actual" for s in C.WEATHER_STATIONS]

# ── Paylaşımlı katmandan re-export (geriye-uyumluluk — çağıranlar değişmedi) ───
_write_typed_parquet = _shared._write_typed_parquet
compute_calendar_fields = _shared.compute_calendar_fields
derive_data_quality_flags = _shared.derive_data_quality_flags


def _forecast_log_path(edas_id: str, target_date: str, run_id: str) -> Path:
    return _shared._forecast_log_path(C.TENANT, target_date, run_id)


def _upsert_by_date(edas_id: str, updates: pd.DataFrame) -> int:
    return _shared.upsert_by_date(C.TENANT, updates)


def _upsert_actuals(edas_id: str, target_date: str, updates: pd.DataFrame) -> None:
    _shared.upsert_actuals(C.TENANT, target_date, updates)


def rebuild_duckdb_views() -> dict:
    return _shared.rebuild_duckdb_views(C.TENANT)


def heal_forecast_log_gaps(edas_id: Optional[str] = None) -> dict:
    return _shared.heal_forecast_log_gaps(C.TENANT)


def backup_logs_zip(keep_days: int = 30) -> Path:
    return _shared.backup_logs_zip(C.TENANT, keep_days)


def reconcile() -> dict:
    """Faz 3: heal + tamlık kontrolü. Gerçek kayıp (arşiv de yok) varsa
    logs/gaps/<date>.json'a yazar (bkz. monitoring/reconcile.py)."""
    from monitoring.reconcile import reconcile as _reconcile
    return _reconcile(C.TENANT)


# ── Yardımcılar: 04'ün ürettiği calendar/horizon alanları ─────────────────────

def compute_horizon_fields(idx: pd.DatetimeIndex, issue_ts: datetime, test_size: int) -> pd.DataFrame:
    """horizon_day (T+1/T+2) + lead_time_h — is_t2 maskesi 05'teki ile aynı (TEST_SIZE//2)."""
    pos = np.arange(len(idx))
    horizon_day = np.where(pos < test_size // 2, "T+1", "T+2")
    issue_ts = pd.Timestamp(issue_ts)
    lead_time_h = ((idx - issue_ts) / pd.Timedelta(hours=1)).to_numpy().round().astype("int32")
    return pd.DataFrame({"horizon_day": horizon_day, "lead_time_h": lead_time_h}, index=idx)


# ── forecast_log yazımı (ADM'ye özgü — postprocessed_predictions.parquet kolon şeması) ──

def write_forecast_log(ctx: dict, postproc_path: Optional[Path] = None,
                        meta_path: Optional[Path] = None, source: str = "live") -> dict:
    """postprocessed_predictions.parquet + raw_predictions_meta.json sidecar + manifest.json
    -> forecast_log parquet (target_date başına partition, 06'dan sonra çağrılır).

    Girdi kolonları 04/05 tarafından üretilir (bkz. dosya başlığı). Eksikse
    (henüz enstrümante edilmemiş run) NaN/None ile doldurulur — sessizce atlanmaz,
    ama pipeline'ı bozmaz.

    `postproc_path`/`meta_path`: varsayılan (None) gerçek canlı yolları (POSTPROC_PATH/
    RAW_PREDICTIONS_META_PATH) okur — normal `run_daily.py`/UI çağrısı budur. Sandbox'lı
    bir regen'den (asof_regen.regen_one) gelen sonuçları loglarken (bkz. backtest_walkforward.py)
    çağıran bu regen'e ait *_models_REGEN.parquet/*_meta_REGEN.json yollarını AÇIKÇA vermeli —
    aksi halde bu fonksiyon o an diskteki GERÇEK (regen'le ilgisiz, bayat) postproc/meta'yı
    okur ve forecast_log'a yanlış tahmin/ağırlık yazar (bu bug'ın canlıya sızmadan
    bulunup düzeltildiği yer: STLF_MONITORING_REFACTOR_PLAN.md §6 Faz 4).
    """
    postproc_path = postproc_path or POSTPROC_PATH
    meta_path = meta_path or RAW_PREDICTIONS_META_PATH

    if not postproc_path.exists():
        return {"status": "no_postproc"}

    df = pd.read_parquet(postproc_path)
    if "Datetime" not in df.columns:
        return {"status": "no_datetime_col"}
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    manifest = {}
    manifest_path = C.MODEL_ARCHIVE_DIR / ctx["run_id"] / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_snapshot = manifest.get("feature_snapshot_ref")

    n = len(df)

    def col(name, default=None):
        return df[name] if name in df.columns else pd.Series([default] * n, index=df.index)

    issue_date = pd.Timestamp(ctx["issue_date"]).date()
    horizon_days = (df["Datetime"].dt.date - issue_date).apply(lambda d: d.days).astype("int8")

    out = pd.DataFrame({
        "edas_id": ctx["edas_id"],
        "run_id": ctx["run_id"],
        "config_hash": ctx["config_hash"],
        "issue_ts": pd.Timestamp(ctx["started_at"]),
        "target_ts": df["Datetime"],
        "target_date": df["Datetime"].dt.strftime("%Y-%m-%d"),
        "horizon_day": col("horizon_day"),
        "issue_date": issue_date,
        "horizon_days": horizon_days,
        "source": source,
        "lead_time_h": col("lead_time_h"),
        "y_pred_xgb": col("XGB_Pred"),
        "y_pred_lgbm": col("LGBM_Pred"),
        "y_pred_cat": col("CAT_Pred"),
        "y_pred_chronos": col("CHRONOS_Pred"),
        "cat_present": bool(meta.get("cat_present")) if "cat_present" in meta else col("cat_present", False),
        "chronos_ok": bool(meta.get("chronos_ok")) if "chronos_ok" in meta else col("chronos_ok", False),
        "y_pred_ens_raw": col("Ensemble_Pred_Raw"),
        "meta_method": meta.get("meta_method"),
        "meta_w_xgb": meta.get("meta_w_xgb"),
        "meta_w_lgbm": meta.get("meta_w_lgbm"),
        "meta_w_cat": meta.get("meta_w_cat"),
        "meta_w_chronos": meta.get("meta_w_chronos"),
        "meta_intercept": meta.get("meta_intercept"),
        "override_active": col("override_active", False),
        "override_delta": col("override_delta", 0.0),
        "subst_active": col("subst_active", False),
        "subst_delta": col("subst_delta", 0.0),
        "pv_bias_delta": col("pv_bias_delta", 0.0),
        "y_pred_final": col("Final_Pred"),
        "wx_temp_fcst": col("wx_temp_fcst"),
        "wx_ghi_fcst": col("wx_ghi_fcst"),
        "day_type": col("day_type"),
        "flag_holiday": col("flag_holiday", False),
        "flag_bridge": col("flag_bridge", False),
        "flag_ramadan": col("flag_ramadan", False),
        "model_versions": json.dumps(manifest.get("model_versions", {}), ensure_ascii=False),
        "feature_snapshot_ref": feature_snapshot,
    })

    n_written = 0
    for target_date, part in out.groupby("target_date"):
        path = _forecast_log_path(ctx["edas_id"], target_date, ctx["run_id"])
        _write_typed_parquet(part, FORECAST_LOG_SCHEMA, path)
        n_written += len(part)

    log.info(f"[ForecastLog] {n_written} satır yazıldı -> {C.FORECAST_LOG_DIR}")
    return {"status": "ok", "rows": n_written, "target_dates": sorted(out["target_date"].unique().tolist())}


# ── actuals_log: D+1 yük dalgası (ADM'ye özgü — Tarih+Saat ayrı kolon) ─────────

def update_actuals_log(day_df: pd.DataFrame, edas_id: Optional[str] = None) -> dict:
    """01_ingest_actual'dan çağrılır. day_df: RAW_DATE_COL/RAW_HOUR_COL/RAW_TARGET_COL,
    tek bir günün (genelde 24) satırı — validate() sonrası, henüz upsert edilmemiş hali de olur.
    """
    edas_id = edas_id or C.EDAS_ID
    if day_df.empty:
        return {"status": "no_data"}

    target_ts = pd.to_datetime(day_df[RAW_DATE_COL]) + pd.to_timedelta(day_df[RAW_HOUR_COL], unit="h")
    quality = derive_data_quality_flags(day_df[RAW_TARGET_COL])
    events = _shared.load_known_events(C.TENANT)

    updates = pd.DataFrame({
        "target_ts": target_ts.to_numpy(),
        "y_actual": day_df[RAW_TARGET_COL].to_numpy(),
        "data_quality_flag": quality.to_numpy(),
        "known_event": [_shared.known_event_for(edas_id, ts, events) for ts in target_ts],
    })

    n_written = _shared.upsert_by_date(C.TENANT, updates)

    log.info(f"[ActualsLog] {n_written} satır (y_actual) upsert edildi")
    return {"status": "ok", "rows": n_written}


# ── actuals_log: hava gerçekleşme dalgası (~D+6, ADM'ye özgü kolon adları) ─────

def update_actuals_log_weather(edas_id: Optional[str] = None, lookback_days: int = 30) -> dict:
    """02_fetch_weather'dan çağrılır. weather_history.parquet'teki dolu (_actual)
    satırları actuals_log'a upsert eder. Sadece son `lookback_days` günü tarar
    (performans — tam geçmiş backfill_logs.py'nin işi)."""
    edas_id = edas_id or C.EDAS_ID
    if not C.WEATHER_HISTORY_PARQUET.exists():
        return {"status": "no_weather_history"}

    wh = pd.read_parquet(C.WEATHER_HISTORY_PARQUET)
    wh[RAW_DATE_COL] = pd.to_datetime(wh[RAW_DATE_COL])

    cutoff = pd.Timestamp(date.today() - timedelta(days=lookback_days))
    wh = wh[wh[RAW_DATE_COL] >= cutoff].copy()

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

    n_written = _shared.upsert_by_date(C.TENANT, updates)

    log.info(f"[ActualsLog] {n_written} satır (wx_actual) upsert edildi")
    return {"status": "ok", "rows": n_written}
