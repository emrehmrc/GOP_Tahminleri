# -*- coding: utf-8 -*-
"""
Türkiye tatil takvimi (2018–2025) — dini + resmi + köprü + Ramazan dönemleri.
build_holiday_calendar(years) → {date: metadata dict}
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Literal

import pandas as pd

# --- Resmi tatiller (sabit gün/ay; yıl bazlı 29 Şubat yok) ---
OFFICIAL_RULES: list[tuple[int, int, str]] = [
    (1, 1, "Yilbasi"),
    (4, 23, "Ulusal_Egemenlik"),
    (5, 1, "Emek_ve_Dayanisma"),
    (5, 19, "Ataturk_Genclik_Spor"),
    (7, 15, "Demokrasi_Birlik_Gunu"),  # 15 Temmuz
    (8, 30, "Zafer_Bayrami"),
    (10, 29, "Cumhuriyet_Bayrami"),
    (12, 31, "Yilbasi_Eve"),
]

# Dini bayramlar — Diyanet takvimine yakın tarihler (Türkiye)
# Format: (yıl, ay, gün, gün_no, isim_kök) gün_no: 0=arife, 1-4 bayram
_RELIGIOUS_ROWS: list[tuple[int, int, int, int, str]] = [
    # Ramazan Bayramı
    (2018, 6, 14, 0, "Ramazan_Bayram"),
    (2018, 6, 15, 1, "Ramazan_Bayram"),
    (2018, 6, 16, 2, "Ramazan_Bayram"),
    (2018, 6, 17, 3, "Ramazan_Bayram"),
    (2019, 6, 3, 0, "Ramazan_Bayram"),
    (2019, 6, 4, 1, "Ramazan_Bayram"),
    (2019, 6, 5, 2, "Ramazan_Bayram"),
    (2019, 6, 6, 3, "Ramazan_Bayram"),
    (2020, 5, 23, 0, "Ramazan_Bayram"),
    (2020, 5, 24, 1, "Ramazan_Bayram"),
    (2020, 5, 25, 2, "Ramazan_Bayram"),
    (2020, 5, 26, 3, "Ramazan_Bayram"),
    (2021, 5, 12, 0, "Ramazan_Bayram"),
    (2021, 5, 13, 1, "Ramazan_Bayram"),
    (2021, 5, 14, 2, "Ramazan_Bayram"),
    (2021, 5, 15, 3, "Ramazan_Bayram"),
    (2022, 5, 1, 0, "Ramazan_Bayram"),
    (2022, 5, 2, 1, "Ramazan_Bayram"),
    (2022, 5, 3, 2, "Ramazan_Bayram"),
    (2022, 5, 4, 3, "Ramazan_Bayram"),
    (2023, 4, 20, 0, "Ramazan_Bayram"),
    (2023, 4, 21, 1, "Ramazan_Bayram"),
    (2023, 4, 22, 2, "Ramazan_Bayram"),
    (2023, 4, 23, 3, "Ramazan_Bayram"),
    (2024, 4, 9, 0, "Ramazan_Bayram"),
    (2024, 4, 10, 1, "Ramazan_Bayram"),
    (2024, 4, 11, 2, "Ramazan_Bayram"),
    (2024, 4, 12, 3, "Ramazan_Bayram"),
    (2025, 3, 29, 0, "Ramazan_Bayram"),
    (2025, 3, 30, 1, "Ramazan_Bayram"),
    (2025, 3, 31, 2, "Ramazan_Bayram"),
    (2025, 4, 1, 3, "Ramazan_Bayram"),
    (2026, 3, 19, 0, "Ramazan_Bayram"),
    (2026, 3, 20, 1, "Ramazan_Bayram"),
    (2026, 3, 21, 2, "Ramazan_Bayram"),
    (2026, 3, 22, 3, "Ramazan_Bayram"),
    # Kurban Bayramı
    (2018, 8, 20, 0, "Kurban_Bayram"),
    (2018, 8, 21, 1, "Kurban_Bayram"),
    (2018, 8, 22, 2, "Kurban_Bayram"),
    (2018, 8, 23, 3, "Kurban_Bayram"),
    (2018, 8, 24, 4, "Kurban_Bayram"),
    (2019, 8, 10, 0, "Kurban_Bayram"),
    (2019, 8, 11, 1, "Kurban_Bayram"),
    (2019, 8, 12, 2, "Kurban_Bayram"),
    (2019, 8, 13, 3, "Kurban_Bayram"),
    (2019, 8, 14, 4, "Kurban_Bayram"),
    (2020, 7, 30, 0, "Kurban_Bayram"),
    (2020, 7, 31, 1, "Kurban_Bayram"),
    (2020, 8, 1, 2, "Kurban_Bayram"),
    (2020, 8, 2, 3, "Kurban_Bayram"),
    (2020, 8, 3, 4, "Kurban_Bayram"),
    (2021, 7, 19, 0, "Kurban_Bayram"),
    (2021, 7, 20, 1, "Kurban_Bayram"),
    (2021, 7, 21, 2, "Kurban_Bayram"),
    (2021, 7, 22, 3, "Kurban_Bayram"),
    (2021, 7, 23, 4, "Kurban_Bayram"),
    (2022, 7, 8, 0, "Kurban_Bayram"),
    (2022, 7, 9, 1, "Kurban_Bayram"),
    (2022, 7, 10, 2, "Kurban_Bayram"),
    (2022, 7, 11, 3, "Kurban_Bayram"),
    (2022, 7, 12, 4, "Kurban_Bayram"),
    (2023, 6, 27, 0, "Kurban_Bayram"),
    (2023, 6, 28, 1, "Kurban_Bayram"),
    (2023, 6, 29, 2, "Kurban_Bayram"),
    (2023, 6, 30, 3, "Kurban_Bayram"),
    (2023, 7, 1, 4, "Kurban_Bayram"),
    (2024, 6, 15, 0, "Kurban_Bayram"),
    (2024, 6, 16, 1, "Kurban_Bayram"),
    (2024, 6, 17, 2, "Kurban_Bayram"),
    (2024, 6, 18, 3, "Kurban_Bayram"),
    (2024, 6, 19, 4, "Kurban_Bayram"),
    (2025, 6, 5, 0, "Kurban_Bayram"),
    (2025, 6, 6, 1, "Kurban_Bayram"),
    (2025, 6, 7, 2, "Kurban_Bayram"),
    (2025, 6, 8, 3, "Kurban_Bayram"),
    (2025, 6, 9, 4, "Kurban_Bayram"),
    (2026, 5, 26, 0, "Kurban_Bayram"),
    (2026, 5, 27, 1, "Kurban_Bayram"),
    (2026, 5, 28, 2, "Kurban_Bayram"),
    (2026, 5, 29, 3, "Kurban_Bayram"),
    (2026, 5, 30, 4, "Kurban_Bayram"),
]

# Ramazan ayı (Hicri yaklaşık) — sahur/iftar etkisi için gün bazlı
# (yıl, başlangıç, bitiş) dahil
RAMADAN_PERIODS: list[tuple[int, date, date]] = [
    (2018, date(2018, 5, 16), date(2018, 6, 14)),
    (2019, date(2019, 5, 6), date(2019, 6, 3)),
    (2020, date(2020, 4, 24), date(2020, 5, 23)),
    (2021, date(2021, 4, 13), date(2021, 5, 12)),
    (2022, date(2022, 4, 2), date(2022, 5, 1)),
    (2023, date(2023, 3, 23), date(2023, 4, 20)),
    (2024, date(2024, 3, 11), date(2024, 4, 9)),
    (2025, date(2025, 3, 1), date(2025, 3, 29)),
    (2026, date(2026, 2, 19), date(2026, 3, 19)),
]

# Bilinen köprü günleri (isteğe bağlı genişletilebilir; kabine ilanları buraya eklenir).
# Öncelik: Bu kümedeki tarihler `build_holiday_calendar` içinde `holiday_type=bridge` ve
# `build_event_window_map` içinde resmi tatil (`official_day`) ile aynı substitution ailesine
# girer. Türetilmiş "de facto" köprü adayları (resmi olmayan hafta içi, toplu izin) BRIDGE_DATES
# dışındadır; `classify_de_facto_bridge_day` heuristikleri sadece **hard holiday olmayan**
# hafta içi günleri etiketler.
BRIDGE_DATES: set[date] = {
    date(2019, 8, 9),
    date(2021, 7, 16),
    date(2022, 7, 7),
    date(2024, 4, 8),
    date(2025, 6, 4),
    date(2026, 5, 25),
}


def _official_for_year(y: int) -> list[tuple[date, str]]:
    out = []
    for m, d, name in OFFICIAL_RULES:
        try:
            out.append((date(y, m, d), name))
        except ValueError:
            pass
    return out


def build_holiday_calendar(years: range | list[int] | None = None) -> dict[date, dict[str, Any]]:
    """
    {date: {
        'holiday_type': 'religious' | 'official' | 'bridge' | 'ramadan_period',
        'holiday_name': str,
        'holiday_day_number': int,  # dini: 0-4, resmi: 0
        'category': str,
    }}
    """
    if years is None:
        years = range(2018, 2027)
    years = set(years)
    cal: dict[date, dict[str, Any]] = {}

    for y, m, d, hnum, root in _RELIGIOUS_ROWS:
        if y not in years:
            continue
        dd = date(y, m, d)
        cal[dd] = {
            "holiday_type": "religious",
            "holiday_name": f"{root}_{hnum}" if hnum else f"{root}_Arife",
            "holiday_day_number": hnum,
            "category": root,
        }

    for y in years:
        for dd, name in _official_for_year(y):
            if dd in cal:
                continue
            cal[dd] = {
                "holiday_type": "official",
                "holiday_name": name,
                "holiday_day_number": 0,
                "category": name,
            }

    for bd in BRIDGE_DATES:
        if bd.year not in years:
            continue
        if bd not in cal:
            cal[bd] = {
                "holiday_type": "bridge",
                "holiday_name": "Kopru_Gunu",
                "holiday_day_number": 0,
                "category": "bridge",
            }

    for y, s, e in RAMADAN_PERIODS:
        if y not in years:
            continue
        d = s
        day_i = 1
        while d <= e:
            if d not in cal:
                cal[d] = {
                    "holiday_type": "ramadan_period",
                    "holiday_name": "Ramazan_Gunu",
                    "holiday_day_number": day_i,
                    "category": "Ramazan",
                }
            d += timedelta(days=1)
            day_i += 1

    return cal


def lookup_calendar(cal: dict[date, dict], ts: pd.Timestamp) -> dict[str, Any] | None:
    d = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
    return cal.get(d)


def is_holiday_hour_for_lag(ts: pd.Timestamp, cal: dict[date, dict]) -> bool:
    """Lag temizliği: dini+resmi+köprü (Ramazan 'tatil' değil — ayrı işlenir)."""
    meta = lookup_calendar(cal, ts)
    if meta is None:
        return False
    return meta["holiday_type"] in ("religious", "official", "bridge")


def is_ramadan_special_hour(ts: pd.Timestamp) -> bool:
    """03–05 ve 18–21 saatleri (Ramazan blend için)."""
    h = ts.hour
    return (3 <= h <= 5) or (18 <= h <= 21)


def get_calendar_series(index: pd.DatetimeIndex, cal: dict[date, dict]) -> pd.Series:
    """Her timestamp için calendar dict veya None."""
    dates = index.normalize().date
    return pd.Series([cal.get(d) for d in dates], index=index)


# ---------------------------------------------------------------------------
# 10-category event-window label map
# ---------------------------------------------------------------------------

EVENT_WINDOW_LABELS: list[str] = [
    "religious_pre_2",
    "religious_pre_1",
    "religious_day_1",
    "religious_day_2",
    "religious_day_3",
    "religious_post_1",
    "religious_post_2_3",
    "official_pre_1",
    "official_day",
    "official_midweek_pre_1",
    "official_midweek_day",
    "official_midweek_post_1",
    "de_facto_bridge",
    "official_post_1",
    "yilbasi_eve",
    "yilbasi_day",
    "normal",
]


def _is_workday_ew(d: date, holiday_dates: set) -> bool:
    return d.weekday() < 5 and d not in holiday_dates


def _next_workday_ew(d: date, holiday_dates: set, n: int = 1) -> date:
    """n-inci iş günü, d'den sonra."""
    cur = d
    count = 0
    while count < n:
        cur += timedelta(days=1)
        if _is_workday_ew(cur, holiday_dates):
            count += 1
    return cur


def build_event_window_map(
    years: range | list[int] | None = None,
) -> dict[date, str]:
    """
    {date: event_window_label} — 'normal' olmayan tüm tarihler.

    Öncelik sırası (EVENT_WINDOW_LABELS ile aynı): ilk eşleşen kazanır.
    Kullanım:
        label_map = build_event_window_map()
        label = label_map.get(some_date, 'normal')
    """
    if years is None:
        years = range(2017, date.today().year + 2)
    years_set = set(years)

    # Tüm 'sert' tatil tarihleri (religious + official + bridge)
    hard_holiday_dates: set[date] = set()
    for y, m, d, hnum, root in _RELIGIOUS_ROWS:
        if y in years_set:
            hard_holiday_dates.add(date(y, m, d))
    for y in years_set:
        for mo, dd, _name in OFFICIAL_RULES:
            try:
                hard_holiday_dates.add(date(y, mo, dd))
            except ValueError:
                pass
    for bd in BRIDGE_DATES:
        if bd.year in years_set:
            hard_holiday_dates.add(bd)

    # candidates: date -> list of (priority_index, label)
    candidates: dict[date, list[tuple[int, str]]] = {}

    def _add(d: date, label: str) -> None:
        idx = EVENT_WINDOW_LABELS.index(label)
        candidates.setdefault(d, []).append((idx, label))

    # ---- Dini tatiller ----
    religious_periods: dict[tuple, list[tuple[date, int]]] = {}
    for y, m, d, hnum, root in _RELIGIOUS_ROWS:
        if y not in years_set:
            continue
        key = (y, root)
        religious_periods.setdefault(key, []).append((date(y, m, d), hnum))

    for (_y, _root), entries in religious_periods.items():
        entries.sort(key=lambda x: x[1])
        arefe_date = next((d for d, h in entries if h == 0), None)
        holiday_days = [(d, h) for d, h in entries if h >= 1]

        if arefe_date:
            _add(arefe_date, "religious_pre_1")
            pre2 = arefe_date - timedelta(days=2)
            _add(pre2, "religious_pre_2")

        for d, h in holiday_days:
            if h == 1:
                _add(d, "religious_day_1")
            elif h == 2:
                _add(d, "religious_day_2")
            else:
                _add(d, "religious_day_3")

        if holiday_days:
            last_d = max(d for d, _ in holiday_days)
            post1 = _next_workday_ew(last_d, hard_holiday_dates, 1)
            _add(post1, "religious_post_1")
            post2 = _next_workday_ew(last_d, hard_holiday_dates, 2)
            _add(post2, "religious_post_2_3")
            post3 = _next_workday_ew(last_d, hard_holiday_dates, 3)
            _add(post3, "religious_post_2_3")

    # ---- Resmi tatiller ----
    for y in years_set:
        for mo, dd, name in OFFICIAL_RULES:
            try:
                od = date(y, mo, dd)
            except ValueError:
                continue
            
            if name == "Yilbasi_Eve":
                day_lbl = "yilbasi_eve"
                pre_lbl = "official_pre_1"
                post_lbl = "normal"
            elif name == "Yilbasi":
                day_lbl = "yilbasi_day"
                pre_lbl = "normal"
                post_lbl = "official_post_1"
            else:
                is_midweek = od.weekday() in (1, 2, 3)
                day_lbl = "official_midweek_day" if is_midweek else "official_day"
                pre_lbl = "official_midweek_pre_1" if is_midweek else "official_pre_1"
                post_lbl = "official_midweek_post_1" if is_midweek else "official_post_1"

            _add(od, day_lbl)
            
            if name != "Yilbasi":
                pre1 = od - timedelta(days=1)
                if _is_workday_ew(pre1, hard_holiday_dates):
                    _add(pre1, pre_lbl)
            
            if name != "Yilbasi_Eve":
                post1 = _next_workday_ew(od, hard_holiday_dates, 1)
                _add(post1, post_lbl)

    # ---- Köprü günleri -> official_day olarak ----
    for bd in BRIDGE_DATES:
        if bd.year not in years_set:
            continue
        _add(bd, "official_day")
        pre1 = bd - timedelta(days=1)
        if _is_workday_ew(pre1, hard_holiday_dates):
            _add(pre1, "official_pre_1")
        post1 = _next_workday_ew(bd, hard_holiday_dates, 1)
        _add(post1, "official_post_1")

    # Çakışmaları çöz: en düşük öncelik indeksli (en yüksek öncelikli) kazanır
    label_map: dict[date, str] = {}
    for d, lst in candidates.items():
        label_map[d] = min(lst, key=lambda x: x[0])[1]

    # ---- De facto köprü günleri: sadece henüz etiketlenmemiş hafta içi günler ----
    for y in years_set:
        start = date(y, 1, 1)
        end = date(y, 12, 31)
        d = start
        while d <= end:
            if d.weekday() < 5 and d not in label_map and d not in hard_holiday_dates:
                kind = classify_de_facto_bridge_day(d)
                if kind is not None:
                    label_map[d] = "de_facto_bridge"
            d += timedelta(days=1)

    return label_map


# ---------------------------------------------------------------------------
# De facto "köprü" adayı — resmi listeye girmeden, hafta sonu / tatil ile birleştirilebilen
# hafta içi (model şu an `normal` gibi görür, yük profili farklı olabilir).
# ---------------------------------------------------------------------------


def collect_hard_holiday_dates(years_set: set[int]) -> set[date]:
    """
    Dini + resmi + `BRIDGE_DATES` (build_event_window_map ile aynı 'hard' küme).
    years_set, d ± birkaç gün için yıl sınırı geçen tarihler de dahil edilmelidir.
    """
    out: set[date] = set()
    for y, m, d, hnum, root in _RELIGIOUS_ROWS:
        if y in years_set:
            out.add(date(y, m, d))
    for y in years_set:
        for mo, dd, _name in OFFICIAL_RULES:
            try:
                out.add(date(y, mo, dd))
            except ValueError:
                pass
    for bd in BRIDGE_DATES:
        if bd.year in years_set:
            out.add(bd)
    return out


DeFactoBridgeKind = Literal["A", "B", "C"]


def _years_window_for(d: date) -> set[int]:
    """Takvim komşuluğu: yıl sınırı (d-2..d+4) için yeterli yıllar."""
    return {d.year - 1, d.year, d.year + 1}


def classify_de_facto_bridge_day(d: date) -> DeFactoBridgeKind | None:
    """
    Sadece **hafta içi (Pzt–Cum)** ve **hard tatil olmayan** günlerde anlam taşır; aksi halde None.

    - **A (Pazartesi):** Cumartesi–Pazar sonrası Pazartesi; ertesi 1 veya 2 gün içinde hard tatil
      (ör. salı resmi, çarşamba arife).
    - **B (Pazartesi, yoğun blok):** Cumartesi–Pazar sonrası Pazartesi; izleyen 4 günün en az
      3'ü hard (ör. ardından 4 gün Kurban gibi arka arkaya tatil).
    - **C (Cuma):** Perşembe hard tatil, Cuma hard değil (Perşembe resmi/ bayram, Cuma izin birleşimi).
    """
    if d.weekday() > 4:
        return None

    yw = _years_window_for(d)
    hard = collect_hard_holiday_dates(yw)
    if d in hard:
        return None

    if d.weekday() == 0:  # Pazartesi
        next4 = [d + timedelta(days=i) for i in (1, 2, 3, 4)]
        n_hard_4 = sum(1 for x in next4 if x in hard)
        if n_hard_4 >= 3:
            return "B"
        t1, t2 = d + timedelta(days=1), d + timedelta(days=2)
        if t1 in hard or t2 in hard:
            return "A"
        return None

    if d.weekday() == 4:  # Cuma
        thu = d - timedelta(days=1)
        if thu in hard:
            return "C"
    return None


def is_de_facto_bridge_day(d: date) -> bool:
    """True iff `classify_de_facto_bridge_day` dönüşü A/B/C (elle BRIDGE hariç, seyrek türetim)."""
    return classify_de_facto_bridge_day(d) is not None
