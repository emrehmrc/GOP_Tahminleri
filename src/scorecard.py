"""
scorecard.py — ADM ince shim (Faz 2, 2026-07-10).
=======================================================
Gerçek mantık monitoring/scorecard.py'de yaşıyor (ADM+GDZ ortak — bkz. o
dosyanın docstring'i). Bu dosya SADECE config_live.TENANT'ı bağlayıp aynı
fonksiyon imzalarını korur ki mevcut çağıranlar (run_daily.py,
ui/tab_tahmin_uret.py, asof_regen.py vb.) HİÇBİR DEĞİŞİKLİK gerektirmesin.
"""

from __future__ import annotations

import pandas as pd

import config_live as C
from monitoring import scorecard as _shared

HOUR_BLOCKS = _shared.HOUR_BLOCKS


def build_daily_scorecard(window_days: int | None = None) -> dict:
    return _shared.build_daily_scorecard(C.TENANT, window_days)


def latest_scorecard(edas_id: str | None = None, horizon: str | None = None) -> pd.DataFrame:
    return _shared.latest_scorecard(C.TENANT, edas_id, horizon)


def window_report(windows: tuple[int, ...] | None = None, edas_id: str | None = None,
                   horizon: str | None = None) -> dict:
    return _shared.window_report(C.TENANT, windows, edas_id, horizon)


def check_alerts(z_threshold: float | None = None) -> list[dict]:
    return _shared.check_alerts(C.TENANT, z_threshold)
