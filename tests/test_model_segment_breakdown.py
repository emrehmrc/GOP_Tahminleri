"""Faz 2 2a-2 (2026-07-13) — monitoring/scorecard.py:model_segment_breakdown
testleri. analyze_models_30d.py'nin artik *_models_REGEN.parquet yerine bu
fonksiyonla forecast_log_v/actuals_log_v'den beslendigini dogrular: model x
saat-blogu x gun-tipi MAPE kirilimi canli veriyle her gun kosulabilmeli.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monitoring.forecast_logger import _write_typed_parquet, rebuild_duckdb_views
from monitoring.schema import ACTUALS_LOG_SCHEMA, FORECAST_LOG_SCHEMA
from monitoring.scorecard import load_hourly_report, model_segment_breakdown
from monitoring.tenant_config import TenantConfig


def _config(tmp_path: Path) -> TenantConfig:
    return TenantConfig(
        edas_id="TST", logger_name="test_adm_live", log_root=tmp_path,
        archive_dir=tmp_path / "archive", weather_history_parquet=tmp_path / "wh.parquet",
        known_events_csv=tmp_path / "known_events.csv", alerts_dir=tmp_path / "alerts",
        log_backup_dir=tmp_path / "backup",
    )


def _forecast_rows(target_date: str, horizon_day: str, day_type: str,
                    y_pred_xgb: list[float], y_pred_final: list[float]) -> pd.DataFrame:
    n = len(y_pred_final)
    return pd.DataFrame({
        "edas_id": ["TST"] * n, "run_id": ["r1"] * n, "config_hash": ["h"] * n,
        "issue_ts": [pd.Timestamp(f"{target_date} 00:00")] * n,
        "target_ts": pd.date_range(f"{target_date} 00:00", periods=n, freq="h"),
        "target_date": [target_date] * n, "horizon_day": [horizon_day] * n,
        "issue_date": [pd.Timestamp(target_date).date()] * n, "horizon_days": [1] * n,
        "source": ["live"] * n, "lead_time_h": list(range(n)),
        "y_pred_xgb": y_pred_xgb, "y_pred_lgbm": y_pred_final,
        "y_pred_cat": y_pred_final, "y_pred_chronos": y_pred_final,
        "cat_present": [True] * n, "chronos_ok": [True] * n,
        "y_pred_ens_raw": y_pred_final, "meta_method": ["static"] * n,
        "meta_w_xgb": [0.25] * n, "meta_w_lgbm": [0.25] * n, "meta_w_cat": [0.25] * n,
        "meta_w_chronos": [0.25] * n, "meta_intercept": [0.0] * n,
        "override_active": [False] * n, "override_delta": [0.0] * n,
        "subst_active": [False] * n, "subst_delta": [0.0] * n, "pv_bias_delta": [0.0] * n,
        "y_pred_final": y_pred_final, "wx_temp_fcst": [25.0] * n, "wx_ghi_fcst": [500.0] * n,
        "day_type": [day_type] * n, "flag_holiday": [False] * n,
        "flag_bridge": [False] * n, "flag_ramadan": [False] * n,
        "model_versions": ["{}"] * n, "feature_snapshot_ref": [None] * n,
    })


def _actuals_rows(target_date: str, y_actual: list[float]) -> pd.DataFrame:
    n = len(y_actual)
    return pd.DataFrame({
        "edas_id": ["TST"] * n,
        "target_ts": pd.date_range(f"{target_date} 00:00", periods=n, freq="h"),
        "target_date": [target_date] * n,
        "y_actual": y_actual,
        "wx_temp_actual": [None] * n, "wx_ghi_actual": [None] * n,
        "data_quality_flag": [""] * n, "known_event": [None] * n,
    })


def _write_forecast(config: TenantConfig, fc: pd.DataFrame, target_date: str, run_id: str = "r1") -> None:
    path = config.forecast_log_dir / "edas_id=TST" / f"target_date={target_date}" / f"run_{run_id}.parquet"
    _write_typed_parquet(fc, FORECAST_LOG_SCHEMA, path)


def _write_actuals(config: TenantConfig, ac: pd.DataFrame, target_date: str) -> None:
    path = config.actuals_log_dir / "edas_id=TST" / f"{target_date}.parquet"
    _write_typed_parquet(ac, ACTUALS_LOG_SCHEMA, path)


def test_segment_breakdown_splits_by_hour_block_and_daytype(tmp_path):
    config = _config(tmp_path)

    # Pazar (gun-tipi grubu "pazar"): XGB gece saatlerinde (0-5) kotu, geri kalanda mukemmel.
    actual = [1000.0] * 24
    xgb_pred = [1200.0] * 6 + [1000.0] * 18  # gece saatleri %20 sapiyor
    final_pred = [1000.0] * 24               # final her saat mukemmel
    _write_actuals(config, _actuals_rows("2026-07-12", actual), "2026-07-12")
    _write_forecast(config, _forecast_rows("2026-07-12", "T+2", "pazar", xgb_pred, final_pred), "2026-07-12")

    rebuild_duckdb_views(config)

    seg = model_segment_breakdown(config, window_days=30, horizon="T+2")
    assert not seg.empty

    xgb_night = seg[(seg["model"] == "xgb") & (seg["hour_block"] == "night")
                     & (seg["day_type_group"] == "pazar")].iloc[0]
    assert xgb_night["mape"] > 15  # gece bloğunda XGB'nin bariz kötü olduğu görülmeli

    xgb_pv = seg[(seg["model"] == "xgb") & (seg["hour_block"] == "pv")
                  & (seg["day_type_group"] == "pazar")].iloc[0]
    assert xgb_pv["mape"] == 0.0  # PV bloğunda XGB de mükemmel

    final_night = seg[(seg["model"] == "final") & (seg["hour_block"] == "night")
                       & (seg["day_type_group"] == "pazar")].iloc[0]
    assert final_night["mape"] == 0.0  # final her yerde mükemmel -- modeller arası ayrım doğrulanıyor


def test_segment_breakdown_empty_without_monitoring_db(tmp_path):
    config = _config(tmp_path)
    assert model_segment_breakdown(config).empty


def test_load_hourly_report_adds_hour_block_and_daytype_group(tmp_path):
    config = _config(tmp_path)
    actual = [1000.0] * 24
    _write_actuals(config, _actuals_rows("2026-07-11", actual), "2026-07-11")  # cumartesi
    _write_forecast(config, _forecast_rows("2026-07-11", "T+2", "cumartesi", actual, actual), "2026-07-11")

    rebuild_duckdb_views(config)

    hourly = load_hourly_report(config, window_days=30, horizon="T+2")
    assert not hourly.empty
    assert set(hourly["hour_block"].unique()) <= {"night", "morning", "pv", "evening", None}
    assert (hourly["day_type_group"] == "cumartesi").all()
