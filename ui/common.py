"""
common.py — Dashboard Ortak Yardımcılar
=========================================
ADM (adm live/pipeline) ve GDZ (gdz talep) modüllerini dosya-yolu bazlı
import eden yardımcılar + güncellik/boşluk hesaplama fonksiyonları.

Dosya-yolu importu (importlib.util) kullanılıyor, sys.path/import ile DEĞİL:
Boray'ın 'config', ADM'in 'config_live', GDZ'nin 'config' modülleri bare
`import config` ile çakışırdı. Her modül spec_from_file_location ile kendi
benzersiz sys.modules anahtarını alıyor, çakışma imkansız.
"""

import sys
import importlib.util
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st

UI_DIR   = Path(__file__).parent
LIVE_DIR = UI_DIR.parent
GDZ_DIR  = LIVE_DIR.parent / "gdz talep"

if str(LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_DIR))
if str(LIVE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(LIVE_DIR / "src"))

from config_live import LIVE_DATA_DIR, MASTER_PARQUET, RAW_TARGET_COL, RAW_DATE_COL, RAW_HOUR_COL
from src.data_scanner import get_ingestion_candidates, scan_available_days


def import_pipeline_step(module_filename: str):
    """adm live/pipeline/<NN_name>.py dosya-yolu ile import (rakamla başlayan isimler normal import'a uygun değil)."""
    path = LIVE_DIR / "pipeline" / f"{module_filename}.py"
    spec = importlib.util.spec_from_file_location(module_filename, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_gdz_ingest():
    """gdz talep/ingest_daily.py dosya-yolu ile import (ayrı repo, sys.path kirletilmez)."""
    path = GDZ_DIR / "ingest_daily.py"
    spec = importlib.util.spec_from_file_location("gdz_ingest_daily", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Güncellik eşikleri ────────────────────────────────────────────────────────
FRESHNESS_WARN_DAYS  = 2
FRESHNESS_STALE_DAYS = 3


def get_pending_for_network(region: str = "aydem") -> list[date]:
    """Ingest edilmemiş günleri döndür (data_scanner üzerinden)."""
    if region == "aydem":
        return get_ingestion_candidates(LIVE_DATA_DIR, MASTER_PARQUET, RAW_DATE_COL, region)
    elif region == "gediz":
        try:
            gdz_mod = import_gdz_ingest()
            return get_ingestion_candidates(LIVE_DATA_DIR, gdz_mod.GDZ_MASTER_PATH, "Tarih", region)
        except Exception:
            return []
    return []


def render_pending_card():
    """Bekleyen veri günleri için üstte bilgi kartı."""
    adm_pending = get_pending_for_network("aydem")
    gdz_pending = get_pending_for_network("gediz")
    total = len(adm_pending) + len(gdz_pending)
    if total > 0:
        msg = f"📥 **{total} bekleyen gün** — "
        if adm_pending:
            msg += f"ADM: {len(adm_pending)} ({adm_pending[0]}..{adm_pending[-1]}) "
        if gdz_pending:
            msg += f"GDZ: {len(gdz_pending)} ({gdz_pending[0]}..{gdz_pending[-1]})"
        st.info(msg)


def scan_gaps(df: pd.DataFrame, date_col: str) -> list[dict]:
    """Tam eksik günleri bul (network-agnostic — sadece takvim günü var/yok)."""
    dates_present = pd.to_datetime(df[date_col]).dt.normalize().unique()
    if len(dates_present) == 0:
        return []
    full_range = pd.date_range(dates_present.min(), dates_present.max(), freq="D")
    missing = sorted(set(full_range) - set(dates_present))
    return [{"date": str(d.date())} for d in missing]


def render_freshness_card(label: str, df: pd.DataFrame, date_col: str, target_col: str,
                            hour_col: str = None):
    """Bir ağ için güncellik kartını çizer: metrik + durum rozeti + özet + son-5-gün expander."""
    dt_series = pd.to_datetime(df[date_col])
    last_date = dt_series.max().date()
    date_min = dt_series.min().date()
    total_rows = len(df)
    days_behind = (date.today() - last_date).days

    st.metric("Son Veri Tarihi", str(last_date), delta=f"{-days_behind} gün", delta_color="inverse")

    if days_behind <= 1:
        st.success("✓ Güncel")
    elif days_behind == FRESHNESS_WARN_DAYS:
        st.warning(f"⚠ {days_behind} gün geride")
    else:
        st.error(f"✗ {days_behind} gün geride — kontrol et")

    st.caption(f"{total_rows:,} satır | {date_min} → {last_date}")

    missing_days = scan_gaps(df, date_col)
    incomplete_days = []
    if hour_col is not None and hour_col in df.columns:
        hours_per_day = df.groupby(dt_series.dt.date)[hour_col].nunique()
        incomplete_days = hours_per_day[hours_per_day < 24].index.tolist()
    else:
        rows_per_day = df.groupby(dt_series.dt.date).size()
        incomplete_days = rows_per_day[rows_per_day < 24].index.tolist()

    if missing_days or incomplete_days:
        with st.expander(f"⚠ Boşluk Detayı ({len(missing_days)} eksik gün, {len(incomplete_days)} eksik-saat gün)"):
            if missing_days:
                st.write("**Eksik günler:**", [d["date"] for d in missing_days])
            if incomplete_days:
                st.write("**Eksik-saat günler:**", [str(d) for d in incomplete_days])

    with st.expander("Son 5 gün"):
        cols = [date_col] + ([hour_col] if hour_col else []) + [target_col]
        sort_cols = [date_col, hour_col] if hour_col else [date_col]
        st.dataframe(df.sort_values(sort_cols).tail(5 * 24)[cols], use_container_width=True)

    return {
        "last_date": last_date,
        "days_behind": days_behind,
        "total_rows": total_rows,
        "date_min": date_min,
    }
