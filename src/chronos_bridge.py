"""
Chronos-2 predict_df formatina, mevcut DataManager panelinden kopru.
VOLTRON (KAGGLE_MEGA_MASTER) ile ayni kovariat isimleri kullanilir; eksik sutunlar 0 ile doldurulur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CHRONOS_SERIES_ID = "AYDEM"

# VOLTRON ile hizali isimler
PAST_ONLY = ["y_Lag24h", "y_Lag168h", "Mean_Last_3Days_Same_Hour"]

FUTURE_KNOWN = [
    "hour_of_day",
    "day_of_week",
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
    "sin_doy",
    "cos_doy",
    "is_sunset_flag",
    "is_monday",
    "is_weekend",
    "Temp_Avg",
    "HDD_Stress",
    "CDD_Stress",
    "GHI",
    "days_until_next_holiday",
    "days_since_last_holiday",
    "holiday_type",
    "holiday_duration_remaining",
    "is_de_facto_bridge",
    "Is_Semester",
    "Is_Summer_Break",
    "Is_lockdown",
    "Is_Sahur",
    "Is_Ramadan",
    "weekday_after_bayram",
    "is_t2_position",
]

ALL_COV_COLS = list(dict.fromkeys(PAST_ONLY + FUTURE_KNOWN))


_HOLIDAY_TYPE_MAP = {
    "normal": 0,
    "official_pre_1": 1,
    "official_day": 1,
    "official_post_1": 1,
    "religious_pre_2": 2,
    "religious_pre_1": 2,
    "religious_day_1": 3,
    "religious_day_2": 3,
    "religious_day_3": 3,
    "religious_post_1": 4,
    "religious_post_2_3": 4,
}


def _compute_holiday_distance(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    from datetime import timedelta
    from src.holiday_calendar import build_holiday_calendar

    years = range(index.year.min() - 1, index.year.max() + 2)
    hcal = build_holiday_calendar(years=list(years))
    hard_dates = {
        d for d, m in hcal.items()
        if m["holiday_type"] in ("religious", "official", "bridge")
    }

    unique_dates = sorted({ts.date() for ts in index})
    date_to_until: dict = {}
    date_to_since: dict = {}

    for d in unique_dates:
        n = 0
        while n <= 365 and (d + timedelta(days=n)) not in hard_dates:
            n += 1
        date_to_until[d] = min(n, 365)

        s = 0
        while s <= 365 and (d - timedelta(days=s)) not in hard_dates:
            s += 1
        date_to_since[d] = min(s, 365)

    dates = [ts.date() for ts in index]
    days_until = np.array([date_to_until[d] for d in dates], dtype="float32")
    days_since = np.array([date_to_since[d] for d in dates], dtype="float32")
    return days_until, days_since


def _compute_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add holiday_duration_remaining, holiday_type, is_t2_position to the panel.
    """
    from src.holiday_calendar import build_event_window_map

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return df

    years = range(idx.year.min() - 1, idx.year.max() + 2)
    event_map = build_event_window_map(years=years)

    HOLIDAY_BLOCK = {
        "religious_pre_2", "religious_pre_1",
        "religious_day_1", "religious_day_2", "religious_day_3",
        "official_pre_1", "official_day",
    }

    dates = [ts.date() for ts in idx]
    labels = [event_map.get(d, "normal") for d in dates]

    # holiday_duration_remaining: days left in current holiday block
    duration = np.zeros(len(df), dtype="float32")
    for i in range(len(df)):
        lbl = labels[i]
        if lbl in HOLIDAY_BLOCK:
            # Count forward within same holiday block
            remaining = 0
            j = i
            while j + 24 < len(df) and idx[j + 24].hour == idx[j].hour:
                j += 24
                jlbl = event_map.get(idx[j].date(), "normal")
                if jlbl in HOLIDAY_BLOCK:
                    remaining += 1
                else:
                    break
            duration[i] = float(remaining + 1)  # including today

    # holiday_type: categorical numeric encoding
    htype = np.zeros(len(df), dtype="float32")
    for i in range(len(df)):
        htype[i] = float(_HOLIDAY_TYPE_MAP.get(labels[i], 0))

    # is_t2_position: 1 for hours 25-48 in each 48h block
    is_t2 = np.zeros(len(df), dtype="float32")
    for i in range(len(df)):
        if i % 48 >= 24:
            is_t2[i] = 1.0

    out = df.copy()
    out["holiday_duration_remaining"] = duration
    out["holiday_type"] = htype
    out["is_t2_position"] = is_t2
    return out


def _pick_temp_series(df: pd.DataFrame) -> pd.Series:
    for c in [
        "Hissedilen_Sicaklik_Mean_MUGLA",
        "Hissedilen_Sicaklik_Mean_DNZ",
        "Hissedilen_Sicaklik_Mean_AYD",
    ]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").astype("float64")
    tcols = [c for c in df.columns if "app_temp_actual" in c]
    if tcols:
        return df[tcols].mean(axis=1).astype("float64")
    return pd.Series(20.0, index=df.index, dtype="float64")


def prepare_panel_for_chronos(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    dm.data kopyasi uzerinde Chronos icin gerekli sutunlari uretir (y, kovaryatlar, sin/cos).
    Orijinal DataManager ciktisini bozmaz; kopya doner.
    """
    out = df.copy()
    out["y"] = pd.to_numeric(out[target_col], errors="coerce").astype("float32")
    out["y_Lag24h"] = out["y"].shift(24)
    if "Mean_Last_3_Days_Same_Hour" in out.columns:
        out["Mean_Last_3Days_Same_Hour"] = pd.to_numeric(
            out["Mean_Last_3_Days_Same_Hour"], errors="coerce"
        ).astype("float32")
    else:
        out["Mean_Last_3Days_Same_Hour"] = (
            (out["y"].shift(24) + out["y"].shift(48) + out["y"].shift(72)) / 3.0
        ).astype("float32")

    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Panel DatetimeIndex bekleniyor.")

    out["is_monday"] = (idx.dayofweek == 0).astype("float32")
    out["is_weekend"] = (idx.dayofweek >= 5).astype("float32")

    if "is_de_facto_bridge" in out.columns:
        out["is_de_facto_bridge"] = (
            pd.to_numeric(out["is_de_facto_bridge"], errors="coerce").fillna(0).astype("float32")
        )
    else:
        from src.holiday_calendar import is_de_facto_bridge_day

        out["is_de_facto_bridge"] = np.array(
            [float(is_de_facto_bridge_day(ts.date())) for ts in idx], dtype="float32"
        )

    for c in [
        "Is_Eve",
        "Is_Sahur",
        "Is_Ramadan",
        "Ramazan_Bayram",
        "Kurban_Bayram",
        "After_Bayram",
        "weekday_after_bayram",
        "is_post_holiday_day1",
        "is_post_holiday_day2",
        "Yilbasi",
        "before_yilbasi",
        "Secim_Gunu",
        "Milli_Bayram",
        "Is_lockdown",
        "Is_Semester",
        "Is_Summer_Break",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("float32")
        else:
            out[c] = np.float32(0)

    temp = _pick_temp_series(out).ffill().bfill()
    out["Temp_Avg"] = temp.astype("float32")
    out["HDD_Stress"] = np.maximum(0.0, 16.0 - temp).astype("float32")
    out["CDD_Stress"] = np.maximum(0.0, temp - 24.0).astype("float32")
    out["Temp_Diff_3h"] = (temp - temp.shift(3)).fillna(0).astype("float32")
    out["Temp_Diff_24h"] = (temp - temp.shift(24)).fillna(0).astype("float32")
    out["temp_squared"] = (temp ** 2).astype("float32")

    if "HDD_Heating_Stress" in out.columns and "HDD_Stress" not in out.columns:
        out["HDD_Stress"] = pd.to_numeric(out["HDD_Heating_Stress"], errors="coerce").fillna(0).astype(
            "float32"
        )
    if "CDD_Cooling_Stress" in out.columns:
        out["CDD_Stress"] = pd.to_numeric(out["CDD_Cooling_Stress"], errors="coerce").fillna(0).astype(
            "float32"
        )

    if "GHI_ADM_Weighted" in out.columns:
        out["GHI"] = pd.to_numeric(out["GHI_ADM_Weighted"], errors="coerce").fillna(0).astype("float32")
    else:
        out["GHI"] = np.float32(0)

    out["hour_of_day"] = idx.hour.astype("float32")
    out["day_of_week"] = idx.dayofweek.astype("float32")
    out["sin_hour"] = np.sin(2 * np.pi * out["hour_of_day"] / 24.0).astype("float32")
    out["cos_hour"] = np.cos(2 * np.pi * out["hour_of_day"] / 24.0).astype("float32")
    out["sin_dow"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0).astype("float32")
    out["cos_dow"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0).astype("float32")
    out["sin_doy"] = np.sin(2 * np.pi * idx.dayofyear / 365.0).astype("float32")
    out["cos_doy"] = np.cos(2 * np.pi * idx.dayofyear / 365.0).astype("float32")
    out["is_sunset_flag"] = ((out["hour_of_day"] >= 15) & (out["hour_of_day"] <= 20)).astype("float32")

    # Lag168h as past-only covariate for T+2 level calibration
    out["y_Lag168h"] = out["y"].shift(168).fillna(0).astype("float32")

    _days_until, _days_since = _compute_holiday_distance(out.index)
    out["days_until_next_holiday"] = _days_until.astype("float32")
    out["days_since_last_holiday"] = _days_since.astype("float32")

    out = _compute_holiday_features(out)

    out = out.fillna(0)
    return out


def panel_slice_to_predict_frames(
    panel: pd.DataFrame,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    context_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    last_train_pos = int(train_positions[-1])
    ctx = panel.iloc[: last_train_pos + 1]
    if len(ctx) > context_length:
        ctx = ctx.iloc[-context_length:]
    fut = panel.iloc[test_positions]

    ctx_df = _rows_to_context_df(ctx)
    fut_df = _rows_to_future_df(fut)
    return ctx_df, fut_df


def _rows_to_context_df(rows: pd.DataFrame) -> pd.DataFrame:
    ds = rows.index
    if isinstance(ds, pd.DatetimeIndex):
        if ds.freq is None or ds.freqstr is None:
            ds = ds.copy()
            try:
                ds.freq = pd.infer_freq(ds) or "h"
            except Exception:
                ds.freq = "h"
    data = {
        "unique_id": CHRONOS_SERIES_ID,
        "ds": ds,
        "y": rows["y"].values.astype("float32"),
    }
    for c in ALL_COV_COLS:
        data[c] = rows[c].values.astype("float32") if c in rows.columns else np.zeros(len(rows), np.float32)
    return pd.DataFrame(data)


def _rows_to_future_df(rows: pd.DataFrame) -> pd.DataFrame:
    ds = rows.index
    if isinstance(ds, pd.DatetimeIndex):
        if ds.freq is None or ds.freqstr is None:
            ds = ds.copy()
            try:
                ds.freq = pd.infer_freq(ds) or "h"
            except Exception:
                ds.freq = "h"
    data = {"unique_id": CHRONOS_SERIES_ID, "ds": ds}
    for c in FUTURE_KNOWN:
        data[c] = rows[c].values.astype("float32") if c in rows.columns else np.zeros(len(rows), np.float32)
    return pd.DataFrame(data)
