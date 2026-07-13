"""Faz 1 (2026-07-13) — forecast_log_v.run_count görünürlüğü.

Aynı (edas_id, target_ts, horizon_day) hücresine birden fazla run yazdıysa
dedup view'da tek satır kalır ama o satırın run_count'u kaç run olduğunu
söyler (eskiden bu bilgi tamamen kaybolurdu — "aynı gün iki kez koşturuldu"
durumu İzleme'de sessizce görünmez kalıyordu).
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

# monitoring/forecast_logger.py bare `from holiday_calendar import ...` yapar --
# src/ sys.path'te olmadan bu dosya TEK BASINA calistirilirsa (pytest tests/
# yerine pytest tests/test_forecast_log_run_count.py) import hatasi verir.
# Diger test dosyalarinin (config_live import eden) sys.path yan etkisine
# guvenmek yerine burada da acikca ekleniyor.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monitoring.forecast_logger import _write_typed_parquet, rebuild_duckdb_views
from monitoring.schema import FORECAST_LOG_SCHEMA
from monitoring.tenant_config import TenantConfig


def _config(tmp_path: Path) -> TenantConfig:
    return TenantConfig(
        edas_id="TST", logger_name="test_adm_live", log_root=tmp_path,
        archive_dir=tmp_path / "archive", weather_history_parquet=tmp_path / "wh.parquet",
        known_events_csv=tmp_path / "known_events.csv", alerts_dir=tmp_path / "alerts",
        log_backup_dir=tmp_path / "backup",
    )


def _make_rows(run_id: str, issue_ts: str, target_date: str, horizon_day: str, n: int = 24) -> pd.DataFrame:
    return pd.DataFrame({
        "edas_id": ["TST"] * n,
        "run_id": [run_id] * n,
        "config_hash": ["h"] * n,
        "issue_ts": [pd.Timestamp(issue_ts)] * n,
        "target_ts": pd.date_range(f"{target_date} 00:00", periods=n, freq="h"),
        "target_date": [target_date] * n,
        "horizon_day": [horizon_day] * n,
        "issue_date": [pd.Timestamp(issue_ts).date()] * n,
        "horizon_days": [1] * n,
        "source": ["live"] * n,
        "lead_time_h": list(range(n)),
        "y_pred_xgb": [1000.0] * n, "y_pred_lgbm": [1000.0] * n,
        "y_pred_cat": [1000.0] * n, "y_pred_chronos": [1000.0] * n,
        "cat_present": [True] * n, "chronos_ok": [True] * n,
        "y_pred_ens_raw": [1000.0] * n, "meta_method": ["static"] * n,
        "meta_w_xgb": [0.25] * n, "meta_w_lgbm": [0.25] * n, "meta_w_cat": [0.25] * n,
        "meta_w_chronos": [0.25] * n, "meta_intercept": [0.0] * n,
        "override_active": [False] * n, "override_delta": [0.0] * n,
        "subst_active": [False] * n, "subst_delta": [0.0] * n, "pv_bias_delta": [0.0] * n,
        "y_pred_final": [1000.0] * n, "wx_temp_fcst": [25.0] * n, "wx_ghi_fcst": [500.0] * n,
        "day_type": ["hafta_ici"] * n, "flag_holiday": [False] * n,
        "flag_bridge": [False] * n, "flag_ramadan": [False] * n,
        "model_versions": ["{}"] * n, "feature_snapshot_ref": [None] * n,
    })


def test_run_count_reflects_shadowed_runs(tmp_path):
    config = _config(tmp_path)

    # Hücre A: 2026-07-14/T+1 icin İKİ run (aynı gün iki kez koşturulmuş gibi).
    df1 = _make_rows("2026-07-13_run1", "2026-07-13T09:00:00", "2026-07-14", "T+1")
    df2 = _make_rows("2026-07-13_run2", "2026-07-13T11:00:00", "2026-07-14", "T+1")
    # Hücre B: 2026-07-15/T+1 icin TEK run.
    df3 = _make_rows("2026-07-13_run3", "2026-07-13T09:00:00", "2026-07-15", "T+1")

    for df, run_id, tdate in [
        (df1, "2026-07-13_run1", "2026-07-14"),
        (df2, "2026-07-13_run2", "2026-07-14"),
        (df3, "2026-07-13_run3", "2026-07-15"),
    ]:
        path = config.forecast_log_dir / f"edas_id=TST" / f"target_date={tdate}" / f"run_{run_id}.parquet"
        _write_typed_parquet(df, FORECAST_LOG_SCHEMA, path)

    result = rebuild_duckdb_views(config)
    assert result["status"] == "ok"

    con = duckdb.connect(str(config.monitoring_db), read_only=True)
    try:
        out = con.execute(
            "SELECT target_date, run_id, run_count FROM forecast_log_v GROUP BY 1,2,3 ORDER BY 1"
        ).df()
    finally:
        con.close()

    # Dedup: her hedef gun icin TEK satir kalmali (kazanan run).
    assert len(out) == 2

    cell_a = out[out.target_date == "2026-07-14"].iloc[0]
    assert cell_a.run_id == "2026-07-13_run2"  # en son issue_ts kazanir
    assert cell_a.run_count == 2               # ama 2 run oldugu goruniyor

    cell_b = out[out.target_date == "2026-07-15"].iloc[0]
    assert cell_b.run_id == "2026-07-13_run3"
    assert cell_b.run_count == 1
