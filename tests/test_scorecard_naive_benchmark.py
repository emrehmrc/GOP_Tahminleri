"""Faz 2 (2026-07-13) — daily_scorecard naive lag168 (geçen hafta aynı saat) benchmark testleri.

Kullanıcının 12 Temmuz Pazar post-mortem'inde sorduğu soru: "model geçen
haftanın aynı gününü kopyalamaktan iyi mi?" — artık her (edas_id, target_date,
horizon_day) satırında mape_naive_lag168 + beats_naive_lag168 olarak otomatik
cevaplanıyor. Bu dosya hem mekanizmayı hem 12 Temmuz'un sentetik bir tekrarını
(model düz devam ediyor, gerçek düşüyor, geçen hafta aynı düşüşü zaten
gösteriyor) test eder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

# monitoring/forecast_logger.py bare `from holiday_calendar import ...` yapar --
# src/ acikca sys.path'te olmali (diger test dosyalarinin yan etkisine guvenme).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monitoring.forecast_logger import _write_typed_parquet, rebuild_duckdb_views
from monitoring.schema import ACTUALS_LOG_SCHEMA, FORECAST_LOG_SCHEMA
from monitoring.scorecard import build_daily_scorecard, window_report
from monitoring.tenant_config import TenantConfig


def _config(tmp_path: Path) -> TenantConfig:
    return TenantConfig(
        edas_id="TST", logger_name="test_adm_live", log_root=tmp_path,
        archive_dir=tmp_path / "archive", weather_history_parquet=tmp_path / "wh.parquet",
        known_events_csv=tmp_path / "known_events.csv", alerts_dir=tmp_path / "alerts",
        log_backup_dir=tmp_path / "backup",
    )


def _forecast_rows(target_date: str, horizon_day: str, y_pred_final: list[float]) -> pd.DataFrame:
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
        "y_pred_final": y_pred_final, "wx_temp_fcst": [25.0] * n, "wx_ghi_fcst": [500.0] * n,
        "day_type": ["pazar"] * n, "flag_holiday": [False] * n,
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


def _write_all(config: TenantConfig, fc: pd.DataFrame, target_date: str, run_id: str = "r1") -> None:
    path = config.forecast_log_dir / "edas_id=TST" / f"target_date={target_date}" / f"run_{run_id}.parquet"
    _write_typed_parquet(fc, FORECAST_LOG_SCHEMA, path)


def _write_actuals(config: TenantConfig, ac: pd.DataFrame, target_date: str) -> None:
    path = config.actuals_log_dir / "edas_id=TST" / f"{target_date}.parquet"
    _write_typed_parquet(ac, ACTUALS_LOG_SCHEMA, path)


def test_naive_lag168_computed_when_history_available(tmp_path):
    config = _config(tmp_path)

    # Gecen hafta Pazar (referans): actual duz 1400.
    _write_actuals(config, _actuals_rows("2026-07-05", [1400.0] * 24), "2026-07-05")
    # Bu hafta Pazar: model 1400 diyor, gercek de 1400 -> mukemmel tahmin.
    _write_actuals(config, _actuals_rows("2026-07-12", [1400.0] * 24), "2026-07-12")
    _write_all(config, _forecast_rows("2026-07-12", "T+2", [1400.0] * 24), "2026-07-12")

    rebuild_duckdb_views(config)
    result = build_daily_scorecard(config)
    assert result["status"] == "ok"

    con = duckdb.connect(str(config.monitoring_db), read_only=True)
    try:
        row = con.execute(
            "SELECT mape_final, mape_naive_lag168, beats_naive_lag168 FROM daily_scorecard "
            "WHERE target_date = '2026-07-12'"
        ).df().iloc[0]
    finally:
        con.close()

    assert row["mape_final"] == 0.0
    assert row["mape_naive_lag168"] == 0.0  # gecen hafta da ayni deger -> naive de mukemmel


def test_model_worse_than_naive_lag168_12_temmuz_style(tmp_path):
    """12 Temmuz senaryosunun sentetik tekrari: model Pazar dususunu KACIRIYOR
    (duz devam tahmini), ama gecen haftanin ayni gunu zaten o dususu gosteriyor.
    beats_naive_lag168 = False olmali -- tam olarak kacirilan sinyal."""
    config = _config(tmp_path)

    # Gecen hafta Pazar: gercek zaten dusuk (1420) -- dogru referans.
    _write_actuals(config, _actuals_rows("2026-07-05", [1420.0] * 24), "2026-07-05")
    # Bu hafta Pazar: GERCEK de dusuk (1400) ama MODEL duz hafta-ici seviyesinde
    # kalmis (1600) -- Pazar dususunu kacirmis.
    _write_actuals(config, _actuals_rows("2026-07-12", [1400.0] * 24), "2026-07-12")
    _write_all(config, _forecast_rows("2026-07-12", "T+2", [1600.0] * 24), "2026-07-12")

    rebuild_duckdb_views(config)
    build_daily_scorecard(config)

    con = duckdb.connect(str(config.monitoring_db), read_only=True)
    try:
        row = con.execute(
            "SELECT mape_final, mape_naive_lag168, beats_naive_lag168 FROM daily_scorecard "
            "WHERE target_date = '2026-07-12'"
        ).df().iloc[0]
    finally:
        con.close()

    assert row["mape_final"] > row["mape_naive_lag168"]
    assert bool(row["beats_naive_lag168"]) is False


def test_naive_lag168_nan_when_no_history_first_week(tmp_path):
    """Sistemin ilk haftasinda (henuz 7 gun once actual yok) mape_naive_lag168
    NaN kalir -- crash etmez, sessizce eksik gorunur."""
    config = _config(tmp_path)
    _write_actuals(config, _actuals_rows("2026-07-12", [1400.0] * 24), "2026-07-12")
    _write_all(config, _forecast_rows("2026-07-12", "T+2", [1400.0] * 24), "2026-07-12")

    rebuild_duckdb_views(config)
    build_daily_scorecard(config)

    con = duckdb.connect(str(config.monitoring_db), read_only=True)
    try:
        row = con.execute(
            "SELECT mape_naive_lag168, beats_naive_lag168 FROM daily_scorecard "
            "WHERE target_date = '2026-07-12'"
        ).df().iloc[0]
    finally:
        con.close()

    assert pd.isna(row["mape_naive_lag168"])
    assert row["beats_naive_lag168"] is None or pd.isna(row["beats_naive_lag168"])


def test_window_report_includes_vs_naive_bps(tmp_path):
    config = _config(tmp_path)
    _write_actuals(config, _actuals_rows("2026-07-05", [1420.0] * 24), "2026-07-05")
    _write_actuals(config, _actuals_rows("2026-07-12", [1400.0] * 24), "2026-07-12")
    _write_all(config, _forecast_rows("2026-07-12", "T+2", [1600.0] * 24), "2026-07-12")

    rebuild_duckdb_views(config)
    build_daily_scorecard(config)

    report = window_report(config, windows=(7,), edas_id="TST", horizon="T+2")
    agg7 = report[7]
    assert agg7["mape_naive_lag168"] is not None
    assert agg7["vs_naive_lag168_bps"] is not None
    assert agg7["vs_naive_lag168_bps"] < 0  # model naive'den kotu -> negatif bps
