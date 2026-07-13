"""
schema.py — forecast_log / actuals_log pyarrow şemaları (ADM + GDZ ortak).

stlf_forecast_log_tasarim.md §2/§3 kaynaklı. Faz 1'de (2026-07-10) issue_date/
horizon_days/source eklendi (horizon_day string geriye-uyumluluk için kaldı).
"""

import pyarrow as pa

FORECAST_LOG_SCHEMA = pa.schema([
    ("edas_id", pa.string()),
    ("run_id", pa.string()),
    ("config_hash", pa.string()),
    ("issue_ts", pa.timestamp("us")),
    ("target_ts", pa.timestamp("us")),
    ("target_date", pa.string()),
    ("horizon_day", pa.string()),          # "T+1" / "T+2" — geriye-uyumluluk (UI hala bunu okuyor)
    ("issue_date", pa.date32()),           # Faz 1: bu run'ın mantıksal issue günü
    ("horizon_days", pa.int8()),           # Faz 1: (target_date-issue_date).days — kanonik alan
    ("source", pa.string()),               # Faz 1: "live" | "archive_heal" | "backfill" (Faz 4: walkforward)
    ("lead_time_h", pa.int32()),
    # Alt-model ham tahminleri
    ("y_pred_xgb", pa.float64()),
    ("y_pred_lgbm", pa.float64()),
    ("y_pred_cat", pa.float64()),
    ("y_pred_chronos", pa.float64()),
    ("cat_present", pa.bool_()),
    ("chronos_ok", pa.bool_()),
    # Stacking / meta katman
    ("y_pred_ens_raw", pa.float64()),
    ("meta_method", pa.string()),
    ("meta_w_xgb", pa.float64()),
    ("meta_w_lgbm", pa.float64()),
    ("meta_w_cat", pa.float64()),
    ("meta_w_chronos", pa.float64()),
    ("meta_intercept", pa.float64()),
    # Düzeltme zinciri
    ("override_active", pa.bool_()),
    ("override_delta", pa.float64()),
    ("subst_active", pa.bool_()),
    ("subst_delta", pa.float64()),
    ("pv_bias_delta", pa.float64()),
    ("y_pred_final", pa.float64()),
    # Kullanılan dış girdiler
    ("wx_temp_fcst", pa.float64()),
    ("wx_ghi_fcst", pa.float64()),
    # Takvim & segment
    ("day_type", pa.string()),
    ("flag_holiday", pa.bool_()),
    ("flag_bridge", pa.bool_()),
    ("flag_ramadan", pa.bool_()),
    # Sürüm & tekrarlanabilirlik
    ("model_versions", pa.string()),        # json
    ("feature_snapshot_ref", pa.string()),
])

ACTUALS_LOG_SCHEMA = pa.schema([
    ("edas_id", pa.string()),
    ("target_ts", pa.timestamp("us")),
    ("target_date", pa.string()),
    ("y_actual", pa.float64()),
    ("wx_temp_actual", pa.float64()),
    ("wx_ghi_actual", pa.float64()),
    ("data_quality_flag", pa.string()),
    ("known_event", pa.string()),
])
