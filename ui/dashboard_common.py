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
import subprocess
import importlib
import importlib.util
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st

UI_DIR   = Path(__file__).parent
LIVE_DIR = UI_DIR.parent
GDZ_DIR  = LIVE_DIR.parent / "gdz talep"
GDZ_LIVE_DIR = GDZ_DIR / "live"

if str(LIVE_DIR) in sys.path:
    sys.path.remove(str(LIVE_DIR))
sys.path.insert(0, str(LIVE_DIR))
if str(LIVE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(LIVE_DIR / "src"))


def refresh_adm_config():
    """Streamlit hot-reload sonrasinda config_live cache'ini dosyayla esitle.

    Streamlit ana scripti yeniden calistirirken bagimli modulleri sys.modules'da
    tutabilir. config_live.py'ye sonradan eklenen sabitler bu durumda dosyada
    bulunsa bile eski modul nesnesinde gorunmez. Her dinamik pipeline importundan
    once ayni modul nesnesini reload etmek hem mevcut baglantilari korur hem de
    yeni sabitleri yukler.
    """
    expected = (LIVE_DIR / "config_live.py").resolve()
    cached = sys.modules.get("config_live")
    cached_file = Path(getattr(cached, "__file__", "")).resolve() if cached else None
    importlib.invalidate_caches()
    if cached is not None and cached_file == expected:
        try:
            return importlib.reload(cached)
        except (ImportError, ModuleNotFoundError):
            # Bazi dinamik import bicimleri modul spec'ini reload icin gecersiz
            # birakabilir. Bu durumda temiz bir normal import ayni isi gorur.
            sys.modules.pop("config_live", None)
    if cached is not None:
        # Ayni isimle baska projenin config'i yuklenmisse ADM config'ine gec.
        sys.modules.pop("config_live", None)
    return importlib.import_module("config_live")


_ADM_CONFIG = refresh_adm_config()
LIVE_DATA_DIR = _ADM_CONFIG.LIVE_DATA_DIR
MASTER_PARQUET = _ADM_CONFIG.MASTER_PARQUET
RAW_TARGET_COL = _ADM_CONFIG.RAW_TARGET_COL
RAW_DATE_COL = _ADM_CONFIG.RAW_DATE_COL
RAW_HOUR_COL = _ADM_CONFIG.RAW_HOUR_COL
from src.data_scanner import get_ingestion_candidates, scan_available_days


def import_pipeline_step(module_filename: str):
    """adm live/pipeline/<NN_name>.py dosya-yolu ile import (rakamla başlayan isimler normal import'a uygun değil)."""
    refresh_adm_config()
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


# ── GDZ canlı pipeline — SUBPROCESS ile (in-process DEĞİL) ────────────────────
# GEREKÇE: ADM'nin bu dosyanın (ve tab_tahmin_uret.py'nin) modül-seviyesinde bare
# `from run_context import ...` / `from forecast_logger import ...` importları var —
# bunlar sys.modules'a ADM'nin versiyonuyla cache'leniyor. GDZ'nin `live/src/`
# altında AYNI isimde ama farklı içerikli (config_live_gdz'ye bağlı) kendi
# run_context.py/forecast_logger.py'si var; GDZ'nin pipeline script'leri de bu
# isimleri bare import ediyor. Aynı process'te çalıştırılırsa Python sys.modules'da
# zaten cache'li ADM versiyonunu döndürür — GDZ kodu yanlış (ADM şemalı) fonksiyonları
# çağırmış olur (sessiz, tehlikeli çakışma). Subprocess = ayrı sys.modules = risk yok.
def gdz_stdout_path(for_date: date | None = None) -> Path:
    d = for_date or date.today()
    return GDZ_LIVE_DIR / "logs" / f"{d}_ui_subprocess.log"


def run_gdz_pipeline_async(args: list[str]) -> subprocess.Popen:
    """gdz talep/live/run_daily.py'yi subprocess olarak başlatır (non-blocking).

    stdout/stderr bir DOSYAYA yönlenir, subprocess.PIPE'a DEĞİL. DÜZELTME
    (2026-07-07, canlı gözlemlendi): PIPE ile başlatılıp UI tarafında hiç
    okunmayınca (sadece ayrı run.log dosyası tail'leniyordu) OS pipe buffer'ı
    dolup GDZ'nin print() çağrıları sonsuza kadar bloke oldu — process canlı
    görünüyordu ama 03_FEATURES'ta 16+ dakika ilerlemesiz kaldı (klasik
    subprocess PIPE-deadlock). Dosyaya yönlendirme bu riski komple ortadan
    kaldırır (parent okusun/okumasın, child'ın yazması asla bloke olmaz) —
    hem POSIX hem Windows'ta subprocess modülünün resmi/güvenli deseni.
    """
    stdout_path = gdz_stdout_path()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            [sys.executable, str(GDZ_LIVE_DIR / "run_daily.py"), *args],
            cwd=str(GDZ_LIVE_DIR),
            stdout=f, stderr=subprocess.STDOUT, text=True,
        )
    return proc


def gdz_log_path(for_date: date | None = None) -> Path:
    d = for_date or date.today()
    return GDZ_LIVE_DIR / "logs" / f"{d}_run.log"


def gdz_summary_path(for_date: date | None = None) -> Path:
    d = for_date or date.today()
    return GDZ_LIVE_DIR / "logs" / f"{d}_summary.json"


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
