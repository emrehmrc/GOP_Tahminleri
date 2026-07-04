# -*- coding: utf-8 -*-
"""
Tatil günleri için geçmiş yıl profil substitution + bayram sonu blend.
Stacking çıktısından sonra apply_substitution() ile post-process.

Blend: final = alpha * lag_sub + (1 - alpha) * base (kategori bazlı alpha).
alpha=0 -> o kategoride substitution kapalı.

12-kategori sistemi (EVENT_WINDOW_LABELS):
    religious_pre_2    → dini tatilden 2 takvim günü önce
    religious_pre_1    → arefe günü
    religious_day_1    → dini tatil 1. günü
    religious_day_2    → dini tatil 2. günü
    religious_day_3    → dini tatil 3+ günü
    religious_post_1   → dini tatil sonrası 1. iş günü
    religious_post_2_3 → dini tatil sonrası 2-3. iş günü
    official_pre_1     → resmi tatilden 1 gün önce
    official_day       → resmi tatil (ve köprü günleri)
    de_facto_bridge    → türetilmiş (resmi olmayan) köprü adayı (A/B/C)
    official_post_1    → resmi tatil sonrası 1. iş günü
    normal             → diğer her şey

Türetilmiş köprü günleri (`de_facto_bridge`) artık ayrı bir substitution kategorisidir.
Alpha üst sınırı 0.60 ile regularize edilir (yılda 2-3 gün → overfit riski).
`classify_de_facto_bridge_day` / `is_de_facto_bridge` sütunu (DataManager) ile taban + Chronos kovaryatına girer.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.holiday_calendar import (
    EVENT_WINDOW_LABELS,
    build_event_window_map,
    build_holiday_calendar,
    is_ramadan_special_hour,
    lookup_calendar,
)

logger = logging.getLogger(__name__)

COVID_YEARS = {2020, 2021}
ELASTICITY_TEMP = 0.008

# Yeni 11-kategori anahtarları (JSON ile aynı)
SUBSTITUTION_BLEND_KEYS = tuple(EVENT_WINDOW_LABELS)

# Geriye dönük uyumluluk: eski anahtar -> yeni anahtar eşlemesi
_LEGACY_KEY_MAP: dict[str, str] = {
    "religious":        "religious_day_1",   # eski "religious" → yeni day_1 (en yaygın)
    "official":         "official_day",
    "bridge":           "official_day",
    "post_holiday_d1":  "religious_post_1",
    "post_holiday_d2":  "religious_post_2_3",
    "post_holiday_d3":  "religious_post_2_3",
    "ramadan_special":  "normal",            # artık yok, normal'e düşer
    "religious_day_2_3": "religious_day_2",  # eski tek kategori → yeni day_2'ye düşer
}

# Güvenli varsayılan alpha (JSON yoksa veya anahtar eksikse)
DEFAULT_BLEND_ALPHAS: dict[str, float] = {k: 0.0 for k in SUBSTITUTION_BLEND_KEYS}
DEFAULT_BLEND_ALPHAS["yilbasi_eve"] = 0.0
DEFAULT_BLEND_ALPHAS["yilbasi_day"] = 0.3

# Fix 4: T+2 için başlangıç override alpha'ları (holiday_walkforward_eval.py --split-t1-t2 ile tune edilmeli)
# Şu an sabit: sadece T+2 alphas JSON yoksa kullanılır
T2_ALPHA_STARTING_POINTS: dict[str, float] = {
    "official_day": 0.1,        # T+1'de 0.1 ile aynı
    "official_post_1": 0.3,     # T+2'de substitution aç (şu an 0.0)
    "religious_post_2_3": 0.2,  # T+2'de substitution aç (şu an 0.0)
    "religious_day_3": 0.3,     # T+2'de biraz daha yüksek (şu an 0.5 birleşik)
}

_PROFILE_CACHE: dict[tuple, np.ndarray | None] = {}


def _normal_day_mean_before(
    df_hist: pd.DataFrame,
    target_col: str,
    end_ts: pd.Timestamp,
    cal: dict,
    n_days: int = 30,
) -> float:
    start = end_ts - pd.Timedelta(days=n_days + 20)
    sl = df_hist.loc[(df_hist.index >= start) & (df_hist.index < end_ts), target_col].astype(float)
    if sl.empty:
        return 1.0
    mask = []
    for ts in sl.index:
        m = lookup_calendar(cal, ts)
        ok = m is None or m["holiday_type"] not in ("religious", "official", "bridge")
        mask.append(ok)
    sub = sl.values[np.array(mask)]
    v = float(np.nanmean(sub)) if len(sub) else float(np.nanmean(sl.values))
    return max(v, 1.0)


def _value_at_hour_on_date(df_hist: pd.DataFrame, target_col: str, d: date, hour: int) -> float | None:
    ts = pd.Timestamp(d) + pd.Timedelta(hours=hour)
    if ts in df_hist.index:
        v = df_hist.loc[ts, target_col]
        if np.isfinite(v):
            return float(v)
    for delta in (-1, 1, -2, 2):
        ts2 = ts + pd.Timedelta(hours=delta)
        if ts2 in df_hist.index:
            v = df_hist.loc[ts2, target_col]
            if np.isfinite(v):
                return float(v)
    return None


def get_holiday_profile(
    target_ts: pd.Timestamp,
    holiday_calendar: dict,
    df_hist: pd.DataFrame,
    target_col: str,
    base_temp_col: str | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    meta = lookup_calendar(holiday_calendar, target_ts)
    info: dict[str, Any] = {"matched": False}
    if meta is None:
        return None, info
    if meta["holiday_type"] not in ("religious", "official"):
        return None, info

    ty = target_ts.year
    cat = meta.get("category", "")
    hnum = meta.get("holiday_day_number", 0)
    htype = meta["holiday_type"]

    is_midweek_official = (htype == "official" and target_ts.weekday() in (1, 2, 3))
    if is_midweek_official:
        ckey = (ty, htype, "midweek_official")
    else:
        ckey = (ty, htype, cat, hnum, meta.get("holiday_name"))

    if ckey in _PROFILE_CACHE:
        return _PROFILE_CACHE[ckey], info

    matches: list[tuple[date, float]] = []
    for d, m in holiday_calendar.items():
        if d.year >= ty:
            continue
        if htype == "religious":
            if m.get("category") != cat or m.get("holiday_day_number") != hnum:
                continue
        else:
            if is_midweek_official:
                # Match any official holiday that fell on a mid-week day
                if m.get("holiday_type") != "official" or d.weekday() not in (1, 2, 3):
                    continue
            else:
                if m.get("holiday_type") != "official" or m.get("holiday_name") != meta.get("holiday_name"):
                    continue
        ydiff = ty - d.year
        w = float(np.exp(-0.5 * ydiff))
        if d.year in COVID_YEARS:
            w *= 0.3
        matches.append((d, w))

    if not matches:
        match_name = "midweek_official" if is_midweek_official else meta.get("holiday_name", cat)
        logger.warning("No historical match for %s y=%s", match_name, ty)
        _PROFILE_CACHE[ckey] = None
        return None, info

    ws = np.array([x[1] for x in matches], dtype=float)
    ws /= ws.sum()

    m_tgt = _normal_day_mean_before(df_hist, target_col, target_ts, holiday_calendar)

    tday = np.nan
    if base_temp_col and base_temp_col in df_hist.columns:
        day0 = target_ts.normalize()
        m1 = (df_hist.index >= day0) & (df_hist.index < day0 + pd.Timedelta(days=1))
        tday = float(df_hist.loc[m1, base_temp_col].mean()) if m1.any() else np.nan

    prof_scaled = np.zeros(24, dtype=float)
    total_w = 0.0

    for (d, _w), nw in zip(matches, ws):
        prof_d = np.zeros(24, dtype=float)
        valid_hours = 0
        for h in range(24):
            val = _value_at_hour_on_date(df_hist, target_col, d, h)
            if val is not None:
                prof_d[h] = val
                valid_hours += 1
        if valid_hours == 0:
            continue

        ref_d = d + timedelta(days=1)
        for _ in range(10):
            m_ref_day = holiday_calendar.get(ref_d)
            if m_ref_day is None or m_ref_day["holiday_type"] not in ("religious", "official", "bridge"):
                break
            ref_d += timedelta(days=1)

        m_ref_d = _normal_day_mean_before(df_hist, target_col, pd.Timestamp(ref_d), holiday_calendar)
        scale_d = m_tgt / max(m_ref_d, 1.0)

        if np.isfinite(tday) and base_temp_col and base_temp_col in df_hist.columns:
            bd0 = pd.Timestamp(d).normalize()
            m2 = (df_hist.index >= bd0) & (df_hist.index < bd0 + pd.Timedelta(days=1))
            tref_d = float(df_hist.loc[m2, base_temp_col].mean()) if m2.any() else np.nan
            if np.isfinite(tref_d):
                scale_d = scale_d * (1.0 + ELASTICITY_TEMP * (tday - tref_d))

        prof_scaled += nw * (prof_d * scale_d)
        total_w += nw

    if total_w <= 0.0 or np.nanmax(prof_scaled) <= 0:
        _PROFILE_CACHE[ckey] = None
        return None, info

    prof_scaled /= total_w

    info["matched"] = True
    info["n_years"] = len(matches)
    _PROFILE_CACHE[ckey] = prof_scaled
    return prof_scaled, info


def _get_post_holiday_profile(
    target_ts: pd.Timestamp,
    event_label: str,
    df_hist: pd.DataFrame,
    target_col: str,
    event_map: dict[date, str],
    holiday_calendar: dict,
    base_temp_col: str | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Post-holiday event-window'lar (religious_post_1, religious_post_2_3, official_post_1)
    için geçmişte aynı event-window kategorisine sahip günlerin saatlik profilini çıkarır.
    """
    ty = target_ts.year
    ckey = ("post", ty, event_label)
    if ckey in _PROFILE_CACHE:
        return _PROFILE_CACHE[ckey], {"matched": True, "cached": True}

    matches: list[tuple[date, float]] = []
    for d, lbl in event_map.items():
        if lbl != event_label:
            continue
        if d.year >= ty:
            continue
        ydiff = ty - d.year
        w = float(np.exp(-0.5 * ydiff))
        if d.year in COVID_YEARS:
            w *= 0.3
        matches.append((d, w))

    if not matches:
        logger.warning("No historical post-holiday match for %s y=%s", event_label, ty)
        _PROFILE_CACHE[ckey] = None
        return None, {"matched": False}

    ws = np.array([x[1] for x in matches], dtype=float)
    ws /= ws.sum()

    m_tgt = _normal_day_mean_before(df_hist, target_col, target_ts, holiday_calendar)

    tday = np.nan
    if base_temp_col and base_temp_col in df_hist.columns:
        day0 = target_ts.normalize()
        m1 = (df_hist.index >= day0) & (df_hist.index < day0 + pd.Timedelta(days=1))
        tday = float(df_hist.loc[m1, base_temp_col].mean()) if m1.any() else np.nan

    prof_scaled = np.zeros(24, dtype=float)
    total_w = 0.0

    for (d, _w), nw in zip(matches, ws):
        prof_d = np.zeros(24, dtype=float)
        valid_hours = 0
        for h in range(24):
            val = _value_at_hour_on_date(df_hist, target_col, d, h)
            if val is not None:
                prof_d[h] = val
                valid_hours += 1
        if valid_hours == 0:
            continue

        ref_d = d + timedelta(days=1)
        for _ in range(10):
            m_ref_day = holiday_calendar.get(ref_d)
            if m_ref_day is None or m_ref_day["holiday_type"] not in ("religious", "official", "bridge"):
                break
            ref_d += timedelta(days=1)

        m_ref_d = _normal_day_mean_before(df_hist, target_col, pd.Timestamp(ref_d), holiday_calendar)
        scale_d = m_tgt / max(m_ref_d, 1.0)

        if np.isfinite(tday) and base_temp_col and base_temp_col in df_hist.columns:
            bd0 = pd.Timestamp(d).normalize()
            m2 = (df_hist.index >= bd0) & (df_hist.index < bd0 + pd.Timedelta(days=1))
            tref_d = float(df_hist.loc[m2, base_temp_col].mean()) if m2.any() else np.nan
            if np.isfinite(tref_d):
                scale_d = scale_d * (1.0 + ELASTICITY_TEMP * (tday - tref_d))

        prof_scaled += nw * (prof_d * scale_d)
        total_w += nw

    if total_w <= 0.0 or np.nanmax(prof_scaled) <= 0:
        _PROFILE_CACHE[ckey] = None
        return None, {"matched": False}

    prof_scaled /= total_w

    _PROFILE_CACHE[ckey] = prof_scaled
    return prof_scaled, {"matched": True, "n_years": len(matches)}


def _normalize_alpha_val(val: Any) -> float | dict:
    if isinstance(val, dict):
        return val
    return float(val)


def load_blend_alphas_json(path: str | None) -> dict[str, Any] | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        out: dict[str, Any] = {}
        for k in SUBSTITUTION_BLEND_KEYS:
            if k in raw:
                out[k] = _normalize_alpha_val(raw[k])
        for old_k, new_k in _LEGACY_KEY_MAP.items():
            if old_k in raw and new_k not in out:
                out[new_k] = _normalize_alpha_val(raw[old_k])
        return out if out else None
    except Exception as e:
        logger.warning("Blend JSON okunamadi %s: %s", path, e)
        return None


def save_blend_alphas_json(path: str, alphas: dict[str, Any]) -> None:
    payload: dict[str, Any] = {}
    for k in SUBSTITUTION_BLEND_KEYS:
        v = alphas.get(k, 0.0)
        if isinstance(v, dict):
            payload[k] = v
        else:
            payload[k] = round(float(v), 4)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Blend alpha'lar kaydedildi: %s", path)


def get_substitution_components(
    index: pd.DatetimeIndex,
    base_predictions: pd.Series | np.ndarray,
    df_history: pd.DataFrame,
    target_col: str,
    holiday_calendar: dict | None = None,
    base_temp_col: str | None = None,
    days_since_series: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    11-kategori event-window sistemi ile lag_sub ve category dizileri üretir.

    Her satır için:
      - lag_sub[i]: historical profilden gelen değer (NaN = profil yok)
      - cat[i]:     event-window etiketi (boş string = normal)
      - base[i]:    base_predictions'dan alınan değer

    Kategoriler (priority: build_event_window_map() ile belirlenir):
      religious_pre_2    → profil yok (lag_sub=NaN, alpha etkisiz)
      religious_pre_1    → arefe profili (get_holiday_profile, day_number=0)
      religious_day_1    → bayram 1. gün profili
      religious_day_2    → bayram 2. gün profili
      religious_day_3    → bayram 3+ gün profili
      religious_post_1   → geçmiş religious_post_1 günleri profili
      religious_post_2_3 → geçmiş religious_post_2_3 günleri profili
      official_pre_1     → profil yok (lag_sub=NaN, alpha etkisiz)
      official_day       → resmi tatil profili (get_holiday_profile)
      official_post_1    → geçmiş official_post_1 günleri profili
    """
    global _PROFILE_CACHE
    _PROFILE_CACHE.clear()

    if holiday_calendar is None:
        holiday_calendar = build_holiday_calendar(
            years=list(range(index.year.min() - 1, index.year.max() + 2))
        )

    # Event-window map: pre-compute for speed
    ew_years = range(index.year.min() - 1, index.year.max() + 2)
    event_map = build_event_window_map(years=ew_years)

    base = np.asarray(base_predictions, dtype=float).ravel()
    n = len(index)
    lag_sub = np.full(n, np.nan, dtype=float)
    cat = np.array([""] * n, dtype=object)

    if len(base) != n:
        raise ValueError("base_predictions uzunlugu index ile uyumsuz")

    # Loglama için sayaçlar
    _applied: dict[str, int] = {k: 0 for k in SUBSTITUTION_BLEND_KEYS}

    for i, ts in enumerate(index):
        ts = pd.Timestamp(ts)
        h = int(ts.hour)
        d = ts.date()

        window_label = event_map.get(d, "normal")

        # ---- Post-holiday: historical same-event-window profile ----
        if window_label == "religious_post_1":
            prof, _ = _get_post_holiday_profile(
                ts, "religious_post_1", df_history, target_col,
                event_map, holiday_calendar, base_temp_col,
            )
            if prof is not None:
                lag_sub[i] = float(prof[h])
                cat[i] = "religious_post_1"
                _applied["religious_post_1"] += 1
            continue

        if window_label == "religious_post_2_3":
            prof, _ = _get_post_holiday_profile(
                ts, "religious_post_2_3", df_history, target_col,
                event_map, holiday_calendar, base_temp_col,
            )
            if prof is not None:
                lag_sub[i] = float(prof[h])
                cat[i] = "religious_post_2_3"
                _applied["religious_post_2_3"] += 1
            continue

        if window_label == "official_post_1":
            prof, _ = _get_post_holiday_profile(
                ts, "official_post_1", df_history, target_col,
                event_map, holiday_calendar, base_temp_col,
            )
            if prof is not None:
                lag_sub[i] = float(prof[h])
                cat[i] = "official_post_1"
                _applied["official_post_1"] += 1
            continue

        if window_label == "official_midweek_post_1":
            prof, _ = _get_post_holiday_profile(
                ts, "official_midweek_post_1", df_history, target_col,
                event_map, holiday_calendar, base_temp_col,
            )
            if prof is not None:
                lag_sub[i] = float(prof[h])
                cat[i] = "official_midweek_post_1"
                _applied["official_midweek_post_1"] += 1
            continue

        if window_label == "de_facto_bridge":
            prof, _ = _get_post_holiday_profile(
                ts, "de_facto_bridge", df_history, target_col,
                event_map, holiday_calendar, base_temp_col,
            )
            if prof is not None:
                lag_sub[i] = float(prof[h])
                cat[i] = "de_facto_bridge"
                _applied["de_facto_bridge"] += 1
            continue

        # ---- Pre-holiday: label set, but no profile → lag_sub stays NaN ----
        if window_label in ("religious_pre_2", "official_pre_1", "official_midweek_pre_1"):
            cat[i] = window_label  # alpha may be set but blend → base (NaN lag_sub)
            continue

        # ---- Calendar-based days (pre_1, day_1, day_2_3, official_day) ----
        meta = lookup_calendar(holiday_calendar, ts)
        if meta is None:
            continue

        ht = meta["holiday_type"]

        if ht in ("religious", "official", "bridge"):
            prof, _ = get_holiday_profile(
                ts, holiday_calendar, df_history, target_col, base_temp_col
            )
            if prof is None:
                continue
            pv = float(prof[h])

            if ht == "religious":
                hnum = meta.get("holiday_day_number", 0)
                if hnum == 0:
                    effective_label = "religious_pre_1"
                elif hnum == 1:
                    effective_label = "religious_day_1"
                elif hnum == 2:
                    effective_label = "religious_day_2"
                else:
                    effective_label = "religious_day_3"
                lag_sub[i] = pv
                cat[i] = effective_label
                _applied[effective_label] += 1

            else:  # official or bridge
                lag_sub[i] = pv
                if window_label in ("yilbasi_eve", "yilbasi_day"):
                    eff_cat = window_label
                else:
                    is_midweek = ts.weekday() in (1, 2, 3) and ht == "official"
                    eff_cat = "official_midweek_day" if is_midweek else "official_day"
                cat[i] = eff_cat
                _applied[eff_cat] += 1

    # Log which windows received substitution values
    active = {k: v for k, v in _applied.items() if v > 0}
    if active:
        logger.info(
            "[HolidaySub] Profil atanan satırlar (lag_sub set): %s",
            ", ".join(f"{k}={v}" for k, v in active.items()),
        )

    return lag_sub, cat, base


def _get_effective_alpha(alpha_spec: Any, hour: int) -> float:
    if isinstance(alpha_spec, dict):
        default = float(alpha_spec.get("default", 0.0))
        overrides = alpha_spec.get("hourly_overrides", {})
        key = f"H{int(hour):02d}"
        return float(overrides.get(key, default))
    return float(alpha_spec)


def blend_with_alphas(
    base: np.ndarray,
    lag_sub: np.ndarray,
    cat: np.ndarray,
    blend_alphas: dict[str, Any],
    is_t2: np.ndarray | None = None,
    blend_alphas_t2: dict[str, float] | None = None,
    hours: np.ndarray | None = None,
) -> np.ndarray:
    """
    final[i] = alpha[cat]*lag + (1-alpha)*base; alpha=0 veya bos kategori -> base.

    is_t2 verilmisse T+2 satirlari icin blend_alphas_t2 (veya blend_alphas) kullanilir.
    hours: saat dizisi (len=n), alpha dict tabanlı ise hourly override için gerekli.
    """
    out = base.copy()
    for i in range(len(base)):
        k = cat[i]
        if k == "" or k not in blend_alphas:
            continue
        h = int(hours[i]) if hours is not None else 12
        if is_t2 is not None and is_t2[i] and blend_alphas_t2 is not None:
            a_spec = blend_alphas_t2.get(k, blend_alphas.get(k, 0.0))
        else:
            a_spec = blend_alphas[k]
        a = _get_effective_alpha(a_spec, h)
        if a <= 0.0:
            continue
        if not np.isfinite(lag_sub[i]):
            continue
        # Quality‑check: if historical profile is too far from base, reduce blend strength
        base_i = float(base[i])
        if base_i > 1.0:
            dev = abs(float(lag_sub[i]) - base_i) / base_i
            if dev > 0.4:
                a *= 0.5
        out[i] = a * float(lag_sub[i]) + (1.0 - a) * base_i
    return out


def optimize_blend_alphas_grid(
    index: pd.DatetimeIndex,
    base: np.ndarray,
    y_true: np.ndarray,
    lag_sub: np.ndarray,
    cat: np.ndarray,
    alpha_step: float = 0.1,
    alpha_max: float = 0.8,
    min_hours: int = 48,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Kategori bazında alpha ∈ [0, alpha_max] grid search; MAPE minimize.
    min_hours'tan az veri olan kategoriler: alpha=0.0 sabit (yetersiz veri uyarısı).
    """
    n_steps = int(round(alpha_max / alpha_step)) + 1
    grid = np.linspace(0.0, alpha_max, n_steps)

    y_true = np.asarray(y_true, dtype=float)
    base = np.asarray(base, dtype=float)

    per_key: dict[str, dict[str, Any]] = {}
    best: dict[str, float] = {}

    for key in SUBSTITUTION_BLEND_KEYS:
        m = (cat == key) & np.isfinite(base) & (y_true > 1.0)
        n_rows = int(m.sum())

        if n_rows < min_hours:
            if n_rows > 0:
                logger.warning(
                    "[AlphaTune] '%s': sadece %d saat verisi var (<%d). alpha=0.0 sabitlendi.",
                    key, n_rows, min_hours,
                )
            per_key[key] = {
                "kategori": key,
                "optimal_alpha": 0.0,
                "baseline_mape": float("nan"),
                "optimized_mape": float("nan"),
                "kazanc_pp": float("nan"),
                "n_satir": n_rows,
                "kilitli": True,
            }
            best[key] = 0.0
            continue

        yt = y_true[m]
        bs = base[m]
        lg = lag_sub[m]

        def _mape(yp: np.ndarray) -> float:
            ok = np.isfinite(yp) & np.isfinite(yt) & (yt > 1.0)
            if not ok.any():
                return float("nan")
            return float(np.mean(np.abs((yt[ok] - yp[ok]) / yt[ok])) * 100.0)

        base_mape = _mape(bs)
        best_a = 0.0
        best_m = base_mape
        for a in grid:
            blended = np.where(
                np.isfinite(lg),
                a * lg + (1.0 - a) * bs,
                bs,
            )
            mm = _mape(blended)
            if not np.isfinite(mm):
                continue
            if mm < best_m - 1e-12:
                best_m = mm
                best_a = float(a)
            elif np.isfinite(best_m) and abs(mm - best_m) <= 1e-9 and float(a) < best_a:
                best_a = float(a)

        gain = (base_mape - best_m) if (np.isfinite(base_mape) and np.isfinite(best_m)) else float("nan")

        per_key[key] = {
            "kategori": key,
            "optimal_alpha": best_a,
            "baseline_mape": base_mape,
            "optimized_mape": best_m,
            "kazanc_pp": gain,
            "n_satir": n_rows,
            "kilitli": False,
        }
        best[key] = best_a

    df = pd.DataFrame([per_key[k] for k in SUBSTITUTION_BLEND_KEYS])
    return df, best


def apply_substitution(
    index: pd.DatetimeIndex,
    base_predictions: pd.Series | np.ndarray,
    df_history: pd.DataFrame,
    target_col: str,
    holiday_calendar: dict | None = None,
    base_temp_col: str | None = None,
    days_since_series: pd.Series | None = None,
    blend_alphas: dict[str, float] | None = None,
    blend_alphas_path: str | None = None,
    is_t2: np.ndarray | None = None,
    blend_alphas_t2_path: str | None = None,
    chronos_predictions: pd.Series | np.ndarray | None = None,
    post_holiday_multipliers: dict[str, float] | None = None,
    post_holiday_multipliers_t2: dict[str, float] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    blend_alphas: kategori -> alpha. alpha=0 o kategoride substitution yok.
    blend_alphas_path: JSON (config HOLIDAY_BLEND_ALPHAS_JSON). Once blend_alphas argumani.
    Ikisi de yoksa DEFAULT_BLEND_ALPHAS (tumu 0.0).

    is_t2: T+2 satirlarini isaretleyen boolean array. Verilirse T+2 icin
    blend_alphas_t2_path'ten okunan ayri alpha'lar kullanilir.
    """
    merged = dict(DEFAULT_BLEND_ALPHAS)
    if blend_alphas_path:
        loaded = load_blend_alphas_json(blend_alphas_path)
        if loaded:
            merged.update(loaded)
    if blend_alphas:
        merged.update(blend_alphas)

    merged_t2: dict[str, float] | None = None
    if is_t2 is not None:
        if blend_alphas_t2_path:
            loaded_t2 = load_blend_alphas_json(blend_alphas_t2_path)
            if loaded_t2:
                def _cast_or_keep(v):
                    if isinstance(v, dict):
                        return v
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return 0.0
                merged_t2 = {k: _cast_or_keep(loaded_t2.get(k, 0.0)) for k in SUBSTITUTION_BLEND_KEYS}
            else:
                merged_t2 = dict(merged)
        else:
            # Fix 4: T+2 override JSON yoksa starting defaults ile doldur
            merged_t2 = dict(merged)
            for k, v in T2_ALPHA_STARTING_POINTS.items():
                merged_t2[k] = float(v)
        # T+2'de tanımlı olmayanlar için T+1 alphası ile doldur
        for k in SUBSTITUTION_BLEND_KEYS:
            def _scalar(v):
                if isinstance(v, dict):
                    return float(v.get("default", 0.0))
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return 0.0
            if _scalar(merged_t2[k]) == 0.0 and _scalar(merged.get(k, 0.0)) > 0:
                merged_t2[k] = merged[k]

    lag_sub, cat, base_raw = get_substitution_components(
        index,
        base_predictions,
        df_history,
        target_col,
        holiday_calendar=holiday_calendar,
        base_temp_col=base_temp_col,
        days_since_series=days_since_series,
    )
    
    base = base_raw.copy()
    
    # 1. Apply New Year (Yılbaşı) blending override: 31 Dec 18:00 to 1 Jan 23:00
    if chronos_predictions is not None:
        chronos_arr = np.asarray(chronos_predictions, dtype=float).ravel()
        if len(chronos_arr) == len(base):
            for i, ts in enumerate(index):
                is_ny_override = (ts.month == 12 and ts.day == 31 and ts.hour >= 18) or (ts.month == 1 and ts.day == 1)
                if is_ny_override:
                    base[i] = chronos_arr[i]
                    
    # 2. Apply post-holiday recovery multipliers to the base predictions before blending
    for i, category_label in enumerate(cat):
        is_hour_t2 = is_t2 is not None and is_t2[i]
        multipliers = post_holiday_multipliers_t2 if is_hour_t2 else post_holiday_multipliers
        if multipliers and category_label in multipliers:
            base[i] = base[i] * multipliers[category_label]
            
    hours_arr = index.hour.values
    out = blend_with_alphas(base, lag_sub, cat, merged, is_t2=is_t2, blend_alphas_t2=merged_t2, hours=hours_arr)

    # Hangi kategorilerde blend uygulandığını logla
    blended_cats = {}
    for k in SUBSTITUTION_BLEND_KEYS:
        a_spec = merged.get(k, 0.0)
        a = a_spec.get("default", a_spec) if isinstance(a_spec, dict) else a_spec
        a_t2_spec = merged_t2.get(k, 0.0) if merged_t2 else 0.0
        a_t2 = a_t2_spec.get("default", a_t2_spec) if isinstance(a_t2_spec, dict) else a_t2_spec
        if float(a) > 0 or float(a_t2) > 0:
            n_blended = int(((cat == k) & np.isfinite(lag_sub)).sum())
            if n_blended > 0:
                label = f"α={a}"
                if float(a_t2) > 0 and abs(float(a_t2) - float(a)) > 0.001:
                    label += f"/T2α={a_t2}"
                blended_cats[k] = {"alpha": a, "alpha_t2": a_t2, "n": n_blended}
    if blended_cats:
        logger.info(
            "[HolidaySub] Blend uygulanan kategoriler: %s",
            ", ".join(f"{k}(α={v['alpha']},n={v['n']})" for k, v in blended_cats.items()),
        )

    meta_stats: dict[str, Any] = {
        "blend_alphas": {k: merged.get(k, 0.0) for k in SUBSTITUTION_BLEND_KEYS},
        "n_by_category": {k: int((cat == k).sum()) for k in SUBSTITUTION_BLEND_KEYS},
    }

    ser = pd.Series(out, index=index, name="substituted")
    return ser, meta_stats
