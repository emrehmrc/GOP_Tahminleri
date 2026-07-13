"""Diagnostic icin model-bileseni ve gerceklesen performans girdileri."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_COLUMNS = {
    "xgb": "y_pred_xgb",
    "lgbm": "y_pred_lgbm",
    "cat": "y_pred_cat",
    "chronos": "y_pred_chronos",
    "ensemble": "y_pred_ens_raw",
    "final": "y_pred_final",
}


def _read_parquets(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_parquet(path))
        except (OSError, ValueError):
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _latest_hourly_run(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "target_ts" not in frame:
        return pd.DataFrame()
    work = frame.copy()
    work["target_ts"] = pd.to_datetime(work["target_ts"], errors="coerce")
    if "horizon_day" in work and (work["horizon_day"] == "T+2").any():
        work = work[work["horizon_day"] == "T+2"]
    if "issue_ts" in work:
        work["issue_ts"] = pd.to_datetime(work["issue_ts"], errors="coerce")
        work = work.sort_values(["target_ts", "issue_ts"])
    return work.dropna(subset=["target_ts"]).drop_duplicates("target_ts", keep="last")


def load_model_signals(forecast_log_root: Path, edas: str, target_date: str | date) -> dict[int, dict]:
    """Hedef gune ait son run'in model bilesenlerini saat bazinda dondur."""
    target = date.fromisoformat(target_date) if isinstance(target_date, str) else target_date
    folder = Path(forecast_log_root) / f"edas_id={edas}" / f"target_date={target}"
    frame = _latest_hourly_run(_read_parquets(sorted(folder.glob("*.parquet"))))
    result: dict[int, dict] = {}
    for _, row in frame.iterrows():
        hour = int(row["target_ts"].hour)
        item = {}
        for label, column in MODEL_COLUMNS.items():
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            item[label] = float(value) if pd.notna(value) and np.isfinite(value) else None
        for flag in ("flag_holiday", "flag_bridge", "flag_ramadan"):
            value = row.get(flag)
            item[flag] = bool(value) if pd.notna(value) else False
        item["day_type"] = None if pd.isna(row.get("day_type")) else str(row.get("day_type"))
        result[hour] = item
    return result


def _actual_frame(actuals_root: Path, edas: str, target: date) -> pd.DataFrame:
    path = Path(actuals_root) / f"edas_id={edas}" / f"target_date={target}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError):
        return pd.DataFrame()
    if "target_ts" not in frame or "y_actual" not in frame:
        return pd.DataFrame()
    frame = frame[["target_ts", "y_actual"]].copy()
    frame["target_ts"] = pd.to_datetime(frame["target_ts"], errors="coerce")
    return frame.dropna(subset=["target_ts", "y_actual"])


def load_hourly_performance(
    forecast_log_root: Path,
    actuals_log_root: Path,
    edas: str,
    target_date: str | date,
    max_days: int = 30,
) -> dict[int, dict]:
    """Son 7/30 gunde final tahmin icin saatlik bias, MAPE ve ornek sayisi."""
    target = date.fromisoformat(target_date) if isinstance(target_date, str) else target_date
    rows = []
    for offset in range(1, max_days + 1):
        day = target - timedelta(days=offset)
        fc_folder = Path(forecast_log_root) / f"edas_id={edas}" / f"target_date={day}"
        forecast = _latest_hourly_run(_read_parquets(sorted(fc_folder.glob("*.parquet"))))
        actual = _actual_frame(actuals_log_root, edas, day)
        if forecast.empty or actual.empty or "y_pred_final" not in forecast:
            continue
        joined = forecast[["target_ts", "y_pred_final"]].merge(actual, on="target_ts", how="inner")
        joined["target_date"] = day
        rows.append(joined)
    history = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    result = {hour: {
        "bias7": None, "mape7": None, "n7": 0,
        "bias30": None, "mape30": None, "n30": 0,
    } for hour in range(24)}
    if history.empty:
        return result
    history["hour"] = pd.to_datetime(history["target_ts"]).dt.hour
    history["error"] = pd.to_numeric(history["y_pred_final"], errors="coerce") - pd.to_numeric(history["y_actual"], errors="coerce")
    actual = pd.to_numeric(history["y_actual"], errors="coerce")
    history["ape"] = np.where(actual > 0, history["error"].abs() / actual * 100.0, np.nan)
    for hour in range(24):
        hourly = history[history["hour"] == hour].sort_values("target_date", ascending=False)
        for window in (7, 30):
            sample = hourly[hourly["target_date"] >= target - timedelta(days=window)]
            sample = sample.dropna(subset=["error", "ape"])
            if not sample.empty:
                result[hour][f"bias{window}"] = float(sample["error"].mean())
                result[hour][f"mape{window}"] = float(sample["ape"].mean())
                result[hour][f"n{window}"] = int(len(sample))
    return result
