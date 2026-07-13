"""Faz 2 2a-3 (2026-07-13) — monitoring/postmortem.py testleri.

"Günün Karnesi"nin veri hali: bir target_date icin per-model MAPE, en kotu 3
saat, naive lag168 farki, gun-tipi baglami tek dict/md/json'da toplaniyor mu?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monitoring.forecast_logger import _write_typed_parquet, rebuild_duckdb_views
from monitoring.postmortem import build_postmortem, render_postmortem_md, write_postmortem
from monitoring.schema import ACTUALS_LOG_SCHEMA, FORECAST_LOG_SCHEMA
from monitoring.tenant_config import TenantConfig


def _config(tmp_path: Path) -> TenantConfig:
    return TenantConfig(
        edas_id="TST", logger_name="test_adm_live", log_root=tmp_path,
        archive_dir=tmp_path / "archive", weather_history_parquet=tmp_path / "wh.parquet",
        known_events_csv=tmp_path / "known_events.csv", alerts_dir=tmp_path / "alerts",
        log_backup_dir=tmp_path / "backup",
    )


def _forecast_rows(target_date: str, horizon_day: str, day_type: str,
                    y_pred_final: list[float], wx_temp_fcst: float = 25.0) -> pd.DataFrame:
    n = len(y_pred_final)
    return pd.DataFrame({
        "edas_id": ["TST"] * n, "run_id": ["r1"] * n, "config_hash": ["h"] * n,
        "issue_ts": [pd.Timestamp(f"{target_date} 00:00")] * n,
        "target_ts": pd.date_range(f"{target_date} 00:00", periods=n, freq="h"),
        "target_date": [target_date] * n, "horizon_day": [horizon_day] * n,
        "issue_date": [pd.Timestamp(target_date).date()] * n, "horizon_days": [1] * n,
        "source": ["live"] * n, "lead_time_h": list(range(n)),
        "y_pred_xgb": y_pred_final, "y_pred_lgbm": y_pred_final,
        "y_pred_cat": y_pred_final, "y_pred_chronos": y_pred_final,
        "cat_present": [True] * n, "chronos_ok": [True] * n,
        "y_pred_ens_raw": y_pred_final, "meta_method": ["static"] * n,
        "meta_w_xgb": [0.25] * n, "meta_w_lgbm": [0.25] * n, "meta_w_cat": [0.25] * n,
        "meta_w_chronos": [0.25] * n, "meta_intercept": [0.0] * n,
        "override_active": [False] * n, "override_delta": [0.0] * n,
        "subst_active": [False] * n, "subst_delta": [0.0] * n, "pv_bias_delta": [0.0] * n,
        "y_pred_final": y_pred_final, "wx_temp_fcst": [wx_temp_fcst] * n, "wx_ghi_fcst": [500.0] * n,
        "day_type": [day_type] * n, "flag_holiday": [False] * n,
        "flag_bridge": [False] * n, "flag_ramadan": [False] * n,
        "model_versions": ["{}"] * n, "feature_snapshot_ref": [None] * n,
    })


def _actuals_rows(target_date: str, y_actual: list[float], wx_temp_actual: float | None = None) -> pd.DataFrame:
    n = len(y_actual)
    return pd.DataFrame({
        "edas_id": ["TST"] * n,
        "target_ts": pd.date_range(f"{target_date} 00:00", periods=n, freq="h"),
        "target_date": [target_date] * n,
        "y_actual": y_actual,
        "wx_temp_actual": [wx_temp_actual] * n, "wx_ghi_actual": [None] * n,
        "data_quality_flag": [""] * n, "known_event": [None] * n,
    })


def _write_forecast(config: TenantConfig, fc: pd.DataFrame, target_date: str, run_id: str = "r1") -> None:
    path = config.forecast_log_dir / "edas_id=TST" / f"target_date={target_date}" / f"run_{run_id}.parquet"
    _write_typed_parquet(fc, FORECAST_LOG_SCHEMA, path)


def _write_actuals(config: TenantConfig, ac: pd.DataFrame, target_date: str) -> None:
    path = config.actuals_log_dir / "edas_id=TST" / f"{target_date}.parquet"
    _write_typed_parquet(ac, ACTUALS_LOG_SCHEMA, path)


def test_postmortem_no_monitoring_db(tmp_path):
    config = _config(tmp_path)
    result = build_postmortem(config, "2026-07-12")
    assert result["status"] == "no_monitoring_db"


def test_postmortem_no_actuals_yet(tmp_path):
    """actuals_log_v view baska bir gunun verisiyle var, ama hedef gunun actual'i
    henuz gelmemis -- join bos kalir, crash etmez."""
    config = _config(tmp_path)
    _write_actuals(config, _actuals_rows("2026-07-01", [1400.0] * 24), "2026-07-01")
    _write_forecast(config, _forecast_rows("2026-07-12", "T+2", "pazar", [1400.0] * 24), "2026-07-12")
    rebuild_duckdb_views(config)
    result = build_postmortem(config, "2026-07-12", horizon="T+2")
    assert result["status"] == "no_actuals"


def test_postmortem_12_temmuz_style_worst_hours_and_naive(tmp_path):
    """12 Temmuz senaryosu: model duz devam ediyor (1600), gercek dusuk (1400),
    gecen hafta zaten dusuktu (1420) -- naive'i gecmemis olmali."""
    config = _config(tmp_path)
    _write_actuals(config, _actuals_rows("2026-07-05", [1420.0] * 24), "2026-07-05")
    _write_actuals(config, _actuals_rows("2026-07-12", [1400.0] * 24, wx_temp_actual=24.0), "2026-07-12")
    _write_forecast(config, _forecast_rows("2026-07-12", "T+2", "pazar", [1600.0] * 24, wx_temp_fcst=26.0),
                     "2026-07-12")
    rebuild_duckdb_views(config)

    result = build_postmortem(config, "2026-07-12", horizon="T+2")
    assert result["status"] == "ok"
    assert result["day_type"] == "pazar"
    assert result["beats_naive_lag168"] is False
    assert result["vs_naive_lag168_bps"] < 0
    assert len(result["worst_hours"]) == 3
    assert result["weather"]["temp_fcst_error_mean"] == 2.0  # 26 - 24

    md = render_postmortem_md(result)
    assert "TST" in md
    assert "GEÇEMEDİ" in md


def test_write_postmortem_creates_md_and_json(tmp_path):
    config = _config(tmp_path)
    _write_actuals(config, _actuals_rows("2026-07-12", [1400.0] * 24), "2026-07-12")
    _write_forecast(config, _forecast_rows("2026-07-12", "T+2", "pazar", [1400.0] * 24), "2026-07-12")
    rebuild_duckdb_views(config)

    out_dir = tmp_path / "output" / "daily" / "2026-07-12"
    result = write_postmortem(config, "2026-07-12", out_dir, horizon="T+2")
    assert result["status"] == "ok"
    assert (out_dir / "postmortem_TST.md").exists()
    saved = json.loads((out_dir / "postmortem_TST.json").read_text(encoding="utf-8"))
    assert saved["edas_id"] == "TST"
