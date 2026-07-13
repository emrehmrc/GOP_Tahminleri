"""Faz 1 (2026-07-13) — monitoring/data_quality.py ingest kalite kapısı testleri."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from monitoring.data_quality import evaluate_ingest_quality

DATE_COL, HOUR_COL, TARGET_COL = "Tarih", "Saat", "Enerji"


def _make_day(d: date, values: list[float], hours: list[int] | None = None) -> pd.DataFrame:
    hours = hours if hours is not None else list(range(24))
    return pd.DataFrame({
        DATE_COL: [pd.Timestamp(d)] * len(hours),
        HOUR_COL: hours,
        TARGET_COL: values,
    })


def _make_history(start: date, n_days: int, base: float = 1000.0) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(7)
    for i in range(n_days):
        d = start + timedelta(days=i)
        vals = base + rng.normal(0, 5, 24)  # dar dağılım -> outlier tespiti kolay tetiklenir
        rows.append(_make_day(d, vals.tolist()))
    return pd.concat(rows, ignore_index=True)


def test_clean_day_returns_ok():
    hist = _make_history(date(2026, 6, 1), 30)
    today = date(2026, 7, 1)
    new_day = _make_day(today, [1000.0] * 24)
    result = evaluate_ingest_quality(new_day, hist, TARGET_COL, DATE_COL, HOUR_COL, lookback_days=30)
    assert result["status"] == "ok"
    assert result["issues"] == []


def test_missing_hours_flagged_as_error():
    hist = _make_history(date(2026, 6, 1), 30)
    today = date(2026, 7, 1)
    new_day = _make_day(today, [1000.0] * 20, hours=list(range(20)))
    result = evaluate_ingest_quality(new_day, hist, TARGET_COL, DATE_COL, HOUR_COL)
    assert result["status"] == "error"
    types = [i["type"] for i in result["issues"]]
    assert "missing_or_extra_hours" in types


def test_negative_value_flagged_as_error():
    hist = _make_history(date(2026, 6, 1), 30)
    today = date(2026, 7, 1)
    vals = [1000.0] * 24
    vals[5] = -10.0
    new_day = _make_day(today, vals)
    result = evaluate_ingest_quality(new_day, hist, TARGET_COL, DATE_COL, HOUR_COL)
    assert result["status"] == "error"
    assert any(i["type"] == "negative_value" for i in result["issues"])


def test_zero_value_flagged_as_warning():
    hist = _make_history(date(2026, 6, 1), 30)
    today = date(2026, 7, 1)
    vals = [1000.0] * 24
    vals[3] = 0.0
    new_day = _make_day(today, vals)
    result = evaluate_ingest_quality(new_day, hist, TARGET_COL, DATE_COL, HOUR_COL)
    assert result["status"] == "warning"
    assert any(i["type"] == "zero_value" for i in result["issues"])


def test_duplicate_timestamp_flagged():
    hist = _make_history(date(2026, 6, 1), 30)
    today = date(2026, 7, 1)
    new_day = _make_day(today, [1000.0] * 24)
    dupe_row = new_day.iloc[[0]]
    new_day_with_dupe = pd.concat([new_day, dupe_row], ignore_index=True)
    result = evaluate_ingest_quality(new_day_with_dupe, hist, TARGET_COL, DATE_COL, HOUR_COL)
    assert any(i["type"] == "duplicate_timestamp" for i in result["issues"])


def test_outlier_vs_history_flagged_12_temmuz_style():
    """07-12 Pazar teşhisinin sentetik tekrarı: geçmiş dar bir bantta iken bugün
    çok sapan bir saat -> robust-z eşiğini aşar, ama koşuyu DURDURMAZ (status
    'warning', 'error' değil)."""
    hist = _make_history(date(2026, 6, 1), 30, base=1500.0)
    today = date(2026, 7, 1)
    vals = [1500.0] * 24
    vals[14] = 2200.0  # 14:00'te tarihsel banttan acayip sapma
    new_day = _make_day(today, vals)
    result = evaluate_ingest_quality(new_day, hist, TARGET_COL, DATE_COL, HOUR_COL, z_threshold=4.0)
    assert result["status"] == "warning"
    outlier_issue = next(i for i in result["issues"] if i["type"] == "outlier_vs_history")
    assert any(h["hour"] == 14 for h in outlier_issue["hours"])


def test_no_history_does_not_crash():
    empty_hist = pd.DataFrame(columns=[DATE_COL, HOUR_COL, TARGET_COL])
    today = date(2026, 7, 1)
    new_day = _make_day(today, [1000.0] * 24)
    result = evaluate_ingest_quality(new_day, empty_hist, TARGET_COL, DATE_COL, HOUR_COL)
    assert result["status"] == "ok"


def test_sparse_history_skips_outlier_check():
    """<5 gün geçmişi olan bir saat için outlier kontrolü atlanır (istatistiksel
    olarak anlamsız medyan/MAD üretmemek için)."""
    hist = _make_history(date(2026, 6, 28), 3, base=1000.0)  # sadece 3 gün
    today = date(2026, 7, 1)
    vals = [1000.0] * 24
    vals[10] = 5000.0  # aşırı sapma ama tarihsel örnek yetersiz
    new_day = _make_day(today, vals)
    result = evaluate_ingest_quality(new_day, hist, TARGET_COL, DATE_COL, HOUR_COL)
    assert not any(i["type"] == "outlier_vs_history" for i in result["issues"])
