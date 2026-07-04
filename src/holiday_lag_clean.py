# -*- coding: utf-8 -*-
"""
Holiday-aware lag cleaning + bayram sonrası feature'ları.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.holiday_calendar import build_holiday_calendar, is_ramadan_special_hour, lookup_calendar


def _holiday_mask_for_lag(cal: dict, index: pd.DatetimeIndex) -> np.ndarray:
    """Dini + resmi + köprü günleri (Ramazan 'dönemi' hariç — lag kirlenmesi bayram/resmi için)."""

    def _d(d):
        m = cal.get(d)
        if m is None:
            return False
        return m["holiday_type"] in ("religious", "official", "bridge")

    days = index.normalize().date
    return np.array([_d(pd.Timestamp(d).date()) for d in days], dtype=bool)


def get_lag_clean(
    df: pd.DataFrame,
    target_col: str,
    holiday_calendar: dict | None = None,
    lag_hours: tuple[int, ...] = (24, 48, 72, 168, 336),
) -> pd.DataFrame:
    """
    lag_X_raw = shift(X); eğer (t-X) tatil saatine denk geliyorsa,
    en yakın 2 non-holiday haftanın aynı saat ortalaması → lag_X_clean
    """
    if holiday_calendar is None:
        holiday_calendar = build_holiday_calendar()

    idx = df.index
    n = len(df)
    series = df[target_col].astype(float)
    hm = _holiday_mask_for_lag(holiday_calendar, idx)

    out = df.copy()
    vals = series.values

    for lag in lag_hours:
        raw = series.shift(lag).values
        clean = raw.copy()
        dirty = np.zeros(n, dtype=bool)
        dirty[lag:] = hm[:-lag]

        # Sadece kirli satırlarda pahalı döngü
        bad_idx = np.where(dirty & np.isfinite(raw))[0]
        for i in bad_idx:
            acc: list[float] = []
            for w in range(1, 12):
                j = i - lag - 168 * w
                if j < 0:
                    break
                if not hm[j] and np.isfinite(vals[j]):
                    acc.append(float(vals[j]))
                    if len(acc) >= 2:
                        break
            if len(acc) >= 2:
                clean[i] = (acc[0] + acc[1]) / 2.0
            elif len(acc) == 1:
                clean[i] = acc[0]
            # else: raw[i] kalır (NaN olabilir)

        out[f"lag_{lag}_clean"] = clean

    return out


def compute_post_holiday_features(df: pd.DataFrame, cal: dict | None = None) -> pd.DataFrame:
    """
    days_since_holiday_end: tatil günü 0; tatil bitişinden 1., 2., ... gün; uzun süre tatil yoksa -1.
    is_post_holiday_day1 / day2: bayram sonrası 1. ve 2. gün.
    recovery_blend_weight: day1→0.3, day2→0.1, else 0
    """
    from datetime import timedelta

    if cal is None:
        cal = build_holiday_calendar()

    idx = df.index
    day_series = pd.Series(idx.normalize().date, index=idx)

    def is_holiday_date(d) -> bool:
        m = cal.get(d)
        if m is None:
            return False
        return m["holiday_type"] in ("religious", "official", "bridge")

    unique_days = sorted(set(day_series.values))
    day_to_since: dict = {}
    is_workday = lambda d: d.weekday() < 5  # Mon=0..Fri=4
    prev_hol = False
    for d in unique_days:
        h = is_holiday_date(d)
        if h:
            day_to_since[d] = 0
            prev_hol = True
        else:
            if not is_workday(d):
                # Weekend after holiday: don't count as workday, but track the pending state
                day_to_since[d] = -1
                # keep prev_hol=True so Monday becomes day 1
            elif prev_hol:
                day_to_since[d] = 1
                prev_hol = False
            else:
                p = d - timedelta(days=1)
                ps = day_to_since.get(p, -1)
                if ps > 0:
                    day_to_since[d] = ps + 1
                elif ps == 0:
                    day_to_since[d] = 1
                else:
                    day_to_since[d] = -1

    days_since = np.array([day_to_since.get(d, -1) for d in day_series.values], dtype=np.int16)
    is_p1 = (days_since == 1).astype(np.int8)
    is_p2 = (days_since == 2).astype(np.int8)
    rec_w = np.zeros(len(df), dtype=np.float32)
    rec_w[days_since == 1] = 0.3
    rec_w[days_since == 2] = 0.1

    out = df.copy()
    out["days_since_holiday_end"] = days_since
    out["is_post_holiday_day1"] = is_p1
    out["is_post_holiday_day2"] = is_p2
    out["recovery_blend_weight"] = rec_w
    return out


def compute_recovery_lags(
    df: pd.DataFrame,
    target_col: str,
    cal: dict | None = None,
) -> pd.DataFrame:
    """
    religious_post_1 günleri için, tatil bloğu öncesindeki temiz aynı-saat değerlerini
    post_holiday_recovery_lag_24 ve post_holiday_recovery_lag_168 olarak üretir.

    Diğer günler: mevcut lag_24_clean / lag_168_clean değerlerini kopyalar.
    Mantık: tatil sonrası 1. günde lag_24_clean zaten tatil dışına yönlenir ama
    "tatilden hemen önce nasıldı" sinyali modele verilmemiş olur.
    Bu feature tatil başlamadan önceki son temiz günün aynı saatini doğrudan sağlar.
    """
    from src.holiday_calendar import build_event_window_map

    if cal is None:
        cal = build_holiday_calendar()

    years_in_data = range(df.index.year.min() - 1, df.index.year.max() + 2)
    event_map = build_event_window_map(years=years_in_data)

    HOLIDAY_BLOCK = {
        "religious_pre_2", "religious_pre_1",
        "religious_day_1", "religious_day_2",
        "religious_day_3",
        "religious_post_1",
    }

    series = df[target_col].astype(float)
    vals = series.values
    idx = df.index
    n = len(df)

    event_days = [event_map.get(ts.date(), "normal") for ts in idx]

    # Default: mevcut lag_clean değerleri
    rec_lag24 = (
        df["lag_24_clean"].values.copy()
        if "lag_24_clean" in df.columns
        else series.shift(24).values
    )
    rec_lag168 = (
        df["lag_168_clean"].values.copy()
        if "lag_168_clean" in df.columns
        else series.shift(168).values
    )

    for i in range(n):
        if event_days[i] != "religious_post_1":
            continue

        # Tatil bloğunun başlamadan önceki son temiz günü bul (24h adımlarla aynı saat)
        j = i - 24  # aynı saatte bir önceki gün
        while j >= 0:
            if event_map.get(idx[j].date(), "normal") not in HOLIDAY_BLOCK:
                break
            j -= 24

        if j < 0:
            continue  # yeterli geçmiş yok, default değer kalsın

        # j: tatil öncesi son temiz aynı-saat pozisyonu
        # post_holiday_recovery_lag_24: tatil öncesi son temiz gün (≈ 24h before holiday_start)
        if np.isfinite(vals[j]):
            rec_lag24[i] = vals[j]

        # post_holiday_recovery_lag_168: tatil öncesi o günden 1 hafta önce
        j168 = j - 168
        if j168 >= 0 and np.isfinite(vals[j168]):
            rec_lag168[i] = vals[j168]

    out = df.copy()
    out["post_holiday_recovery_lag_24"] = rec_lag24
    out["post_holiday_recovery_lag_168"] = rec_lag168
    print("[HolidayLagClean] post_holiday_recovery_lag_24/168 eklendi.")
    return out


def compute_t2_lag_clean(
    df: pd.DataFrame,
    target_col: str,
    cal: dict | None = None,
    lag_hours: tuple[int, ...] = (24,),
) -> pd.DataFrame:
    """
    Fix 2: T+2 (25-48. saat) için lag_24 proxy'si.

    T+2'nin lag_24 değeri bir önceki günün (T+1) saatinden gelir.
    T+1 tatil ise lag_24 kirli olur ve model yanılır.
    Bu proxy, lag_24'ün düştüğü tarih tatil bloğundayken lag_48 veya lag_168 kullanır.

    Her lag_hour için 'lag_{X}_t2_proxy' sütunu üretir.
    """
    if cal is None:
        cal = build_holiday_calendar()

    from src.holiday_calendar import build_event_window_map

    years_in_data = range(df.index.year.min() - 1, df.index.year.max() + 2)
    event_map = build_event_window_map(years=years_in_data)

    HOLIDAY_BLOCK = {
        "religious_pre_2", "religious_pre_1",
        "religious_day_1", "religious_day_2",
        "religious_day_3",
        "religious_post_1",
        "official_pre_1", "official_day", "official_post_1",
    }

    series = df[target_col].astype(float)
    vals = series.values
    idx = df.index
    n = len(df)

    out = df.copy()

    for lag in lag_hours:
        raw = series.shift(lag).values
        proxy = raw.copy()

        lag_date = idx - pd.Timedelta(hours=lag)
        lag_dates = [pd.Timestamp(ts).date() for ts in lag_date]

        for i in range(lag, n):
            d = lag_dates[i]
            lbl = event_map.get(d, "normal")
            if lbl in HOLIDAY_BLOCK:
                # lag_24 kirli -> daha uzun lag'leri dene (lag*2, lag*3, lag*7)
                for multiplier in (2, 3, 7):
                    j = i - lag * multiplier
                    if j < 0:
                        continue
                    jd = idx[j].date()
                    jlbl = event_map.get(jd, "normal")
                    if jlbl not in HOLIDAY_BLOCK and np.isfinite(vals[j]):
                        proxy[i] = vals[j]
                        break
                # En kötü durum: raw değeri koru (kirli de olsa)

        out[f"lag_{lag}_t2_proxy"] = proxy

    print(f"[HolidayLagClean] lag_{list(lag_hours)}_t2_proxy eklendi (Fix 2).")
    return out


def compute_chain_aware_lag_clean(
    df: pd.DataFrame,
    target_col: str,
    holiday_dates_set: set,
) -> pd.DataFrame:
    """
    Tatil zincirleri (4 günlük Kurban vb.) için lag_24 zincirleme temizliği.

    Mevcut get_lag_clean `lag_24_clean` üretir ama uzun zincirlerde lag_24'un
    düştüğü tarih de tatil olduğundan 168h atlamalar yetersiz kalabilir.
    Bu fonksiyon lag_24 hedef tarihi tatil ise 24h adımlarla geriye giderek
    en yakın non-holiday aynı-saat değerini bulur.

    Zincir senaryosu (4 günlük bayram):
      - Day 2: lag_24 → Day 1 (holiday) → lag_48 → clean ✓
      - Day 3: lag_24 → Day 2 (holiday) → lag_48 → Day 1 (holiday) → lag_72 → clean ✓
      - Day 4: lag_24 → Day 3 (holiday) → ... → lag_96 → clean ✓
    """
    out = df.copy()
    series = df[target_col].astype(float)
    vals = series.values
    idx = df.index
    n = len(df)

    lag_24_raw = series.shift(24).values
    chain_clean = lag_24_raw.copy()

    for i in range(24, n):
        lag_date = (idx[i] - pd.Timedelta(hours=24)).date()
        if lag_date not in holiday_dates_set:
            continue

        for step in range(2, 15):
            j = i - 24 * step
            if j < 0:
                break
            j_date = idx[j].date()
            if j_date not in holiday_dates_set and np.isfinite(vals[j]):
                chain_clean[i] = vals[j]
                break

    out["lag_24_chain_clean"] = chain_clean
    print("[HolidayLagClean] lag_24_chain_clean eklendi (chain-aware tatil zinciri temizliği).")
    return out


def attach_lag_clean_and_post_holiday(
    df: pd.DataFrame,
    target_col: str,
    cal: dict | None = None,
) -> pd.DataFrame:
    """DataManager içinden tek çağrı."""
    if cal is None:
        cal = build_holiday_calendar()
    df2 = get_lag_clean(df, target_col, cal)
    df2 = compute_post_holiday_features(df2, cal)
    df2 = compute_recovery_lags(df2, target_col, cal)
    df2 = compute_t2_lag_clean(df2, target_col, cal)
    
    # Chain-aware lag cleaning ekleme
    holiday_dates_set = {pd.Timestamp(d).date() for d, info in cal.items() if info["holiday_type"] in ("religious", "official", "bridge")}
    df2 = compute_chain_aware_lag_clean(df2, target_col, holiday_dates_set)
    
    print("[HolidayLagClean] lag_24/48/72/168/336_clean + post-holiday feature'ları + t2_proxy + chain_clean eklendi.")
    return df2

