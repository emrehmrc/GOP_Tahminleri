from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import diagnostic_core as core
from diagnostic_support import load_hourly_performance, load_model_signals


def _history(target: date) -> pd.DataFrame:
    timestamps = pd.date_range(pd.Timestamp(target) - pd.Timedelta(days=900),
                               pd.Timestamp(target) - pd.Timedelta(hours=1), freq="h")
    doy = timestamps.dayofyear.to_numpy()
    hour = timestamps.hour.to_numpy()
    dow = timestamps.dayofweek.to_numpy()
    temp = 21 + 9 * np.sin(2 * np.pi * (doy - 170) / 365) + 4 * np.sin(2 * np.pi * (hour - 8) / 24)
    ghi = np.maximum(0, 750 * np.sin(np.pi * (hour - 6) / 13))
    cloud = 40 + 20 * np.sin(2 * np.pi * doy / 17)
    humidity = 60 - 0.8 * (temp - 20)
    wind = 3 + 1.5 * np.cos(2 * np.pi * hour / 24) + 0.5 * np.sin(2 * np.pi * doy / 11)
    precip = np.where(cloud > 55, 0.5, 0.0)
    trend = np.arange(len(timestamps)) / (24 * 365)
    load = (900 + 18 * hour + 13 * temp - 0.08 * ghi + 2.5 * cloud
            + np.where(dow == 5, -70, 0) + np.where(dow == 6, -110, 0) + 12 * trend)
    return pd.DataFrame({
        "dt": timestamps.normalize(), "h": hour, "load": load,
        "temp": temp, "ghi": ghi, "cloud": cloud, "humidity": humidity,
        "wind": wind, "precip": precip, "special": None,
    })


def test_consolidated_expected_uses_calendar_load_weather_models_and_performance():
    target = date(2026, 7, 14)
    hist = _history(target)
    hours = np.arange(24)
    fc_wx = {
        "temp": (28 + 4 * np.sin(2 * np.pi * (hours - 8) / 24)).tolist(),
        "ghi": np.maximum(0, 780 * np.sin(np.pi * (hours - 6) / 13)).tolist(),
        "cloud": [35.0] * 24, "humidity": [48.0] * 24,
        "wind": [4.0] * 24, "precip": [0.0] * 24,
    }
    forecast = (1250 + 18 * hours).astype(float).tolist()
    signals = {h: {
        "xgb": forecast[h] - 20, "lgbm": forecast[h] + 10,
        "cat": forecast[h] + 25, "chronos": forecast[h] - 5,
        "ensemble": forecast[h], "flag_holiday": False,
        "flag_bridge": False, "flag_ramadan": False,
    } for h in range(24)}
    performance = {h: {
        "bias7": 20.0, "mape7": 3.0, "n7": 7,
        "bias30": 15.0, "mape30": 3.5, "n30": 20,
    } for h in range(24)}

    result = core.compute(hist, forecast, fc_wx, str(target), "ADM", signals, performance)
    assert len(result["REC"]) == 24
    for row in result["REC"]:
        assert row["exp"] is not None and row["lo"] < row["exp"] < row["hi"]
        assert {"temp", "cloud", "humidity", "wind", "lag7", "asof_d", "asof_d1",
                "asof_d7", "roll7", "roll14"}.issubset(row["features_used"])
        assert row["model_center"] is not None
        assert row["bias_corrected"] is not None
        assert row["confidence"] in {"Yüksek", "Orta", "Düşük"}
        assert 0 <= row["confidence_score"] <= 100
        assert row["drivers"]
        assert {"ridge", "analog", "bias", "models"}.issubset(row["expert_weights"])
    json.dumps(result, ensure_ascii=False)


def _write_log_pair(fc_root: Path, actual_root: Path, edas: str, target: date, issue: date, bias: float):
    ts = pd.date_range(target, periods=24, freq="h")
    folder = fc_root / f"edas_id={edas}" / f"target_date={target}"
    folder.mkdir(parents=True, exist_ok=True)
    forecast = pd.DataFrame({
        "target_ts": ts, "issue_ts": [pd.Timestamp(issue)] * 24,
        "horizon_day": ["T+2"] * 24,
        "y_pred_xgb": np.arange(24) + 1000,
        "y_pred_lgbm": np.arange(24) + 1010,
        "y_pred_cat": np.arange(24) + 1020,
        "y_pred_chronos": np.arange(24) + 1030,
        "y_pred_ens_raw": np.arange(24) + 1015,
        "y_pred_final": np.arange(24) + 1020,
        "flag_holiday": [False] * 24, "flag_bridge": [False] * 24,
        "flag_ramadan": [False] * 24, "day_type": ["hafta_ici"] * 24,
    })
    forecast.to_parquet(folder / f"run_{issue}.parquet", index=False)
    actual_folder = actual_root / f"edas_id={edas}"
    actual_folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"target_ts": ts, "y_actual": forecast["y_pred_final"] - bias}).to_parquet(
        actual_folder / f"target_date={target}.parquet", index=False,
    )


def test_diagnostic_support_reads_model_components_and_hourly_7_30_metrics(tmp_path):
    fc_root, actual_root = tmp_path / "forecast", tmp_path / "actual"
    target = date(2026, 7, 14)
    _write_log_pair(fc_root, actual_root, "ADM", target, target - timedelta(days=2), 10.0)
    for offset in range(1, 6):
        _write_log_pair(fc_root, actual_root, "ADM", target - timedelta(days=offset),
                        target - timedelta(days=offset + 2), 10.0)

    signals = load_model_signals(fc_root, "ADM", target)
    performance = load_hourly_performance(fc_root, actual_root, "ADM", target)
    assert len(signals) == 24
    assert signals[12]["xgb"] == 1012
    assert signals[12]["chronos"] == 1042
    assert performance[12]["n7"] == 5
    assert performance[12]["n30"] == 5
    assert performance[12]["bias30"] == 10.0
    assert performance[12]["mape30"] > 0
