"""
02_fetch_weather.py — Gelecek 48h Hava Tahmini
================================================
Open-Meteo Forecast API'dan bugün + yarın + öbürgün hava verisini çeker.
Mevcut fetch_weather_fc.py ile aynı istasyon listesi / GHI ağırlıkları,
ancak Previous-Runs değil gerçek forecast endpoint.

Giriş:  config_live.py'deki istasyon listesi + OPENMETEO_FC_DAYS
Çıkış:  data/weather_cache/weather_fc_live.parquet
          Kolonlar: Tarih, Saat + <STATION>_app_temp_fc + _precip_fc + _cloud_fc
                    + <STATION>_GHI_fc + GHI_<Prov>_W_m2_fc + GHI_ADM_Weighted_fc
                    + Dark_Fraction_Pct_fc kolonları
"""

import sys
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from config_live import (
    WEATHER_FC_PARQUET, WEATHER_CACHE_DIR, WEATHER_HISTORY_PARQUET,
    WEATHER_STATIONS, WEATHER_GHI_WEIGHTS,
    OPENMETEO_FORECAST_URL, OPENMETEO_FC_DAYS, WEATHER_TIMEZONE, WEATHER_MODEL,
)
from src.forecast_logger import update_actuals_log_weather

PROVINCE_STATIONS = {
    "MUGLA":   [s for s in WEATHER_STATIONS if s.startswith("MUGLA_")],
    "DENIZLI": [s for s in WEATHER_STATIONS if s.startswith("DENIZLI_")],
    "AYDIN":   [s for s in WEATHER_STATIONS if s.startswith("AYDIN_")],
}
DARK_FRAC_PREFIX = {"MUGLA_": "MUGLA_", "DENIZLI_": "DNZ_", "AYDIN_": "AYD_"}
FC_VARS = "apparent_temperature,precipitation,cloud_cover,shortwave_radiation"


def _fetch_station(name: str, lat: float, lon: float, retries: int = 4) -> pd.DataFrame | None:
    """Tek istasyon için forecast API çağrısı."""
    params = {
        "latitude":       lat,
        "longitude":      lon,
        "hourly":         FC_VARS,
        "timezone":       WEATHER_TIMEZONE,
        "models":         WEATHER_MODEL,
        "forecast_days":  OPENMETEO_FC_DAYS,
    }
    for attempt in range(retries):
        try:
            r = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=60)
            r.raise_for_status()
            h = r.json()["hourly"]
            return pd.DataFrame({
                "Datetime":               pd.to_datetime(h["time"]),
                f"{name}_app_temp_fc":    h["apparent_temperature"],
                f"{name}_precip_fc":      h["precipitation"],
                f"{name}_cloud_fc":       h["cloud_cover"],
                f"{name}_GHI_fc":         h["shortwave_radiation"],
            })
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"   [{name}] hata ({e}), {wait}s bekleniyor...")
                time.sleep(wait)
            else:
                print(f"   [{name}] {retries} denemede başarısız: {e}")
                return None


def fetch_all_stations() -> pd.DataFrame:
    combined = None
    for i, (name, (lat, lon)) in enumerate(WEATHER_STATIONS.items()):
        print(f"   [{i+1}/{len(WEATHER_STATIONS)}] {name}")
        df = _fetch_station(name, lat, lon)
        if df is None:
            print(f"   [Uyarı] {name} atlandı!")
            continue
        if combined is None:
            combined = df
        else:
            combined = combined.merge(df, on="Datetime", how="outer")
        time.sleep(0.3)  # rate-limit
    return combined


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """GHI il ortalamaları + ağırlıklı GHI + Dark_Fraction."""
    # İl GHI ortalamaları
    for prov, stations in PROVINCE_STATIONS.items():
        cols = [f"{s}_GHI_fc" for s in stations if f"{s}_GHI_fc" in df.columns]
        if cols:
            df[f"GHI_{prov.title()}_W_m2_fc"] = df[cols].mean(axis=1)

    # Ağırlıklı GHI
    df["GHI_ADM_Weighted_fc"] = (
        df.get("GHI_Mugla_W_m2_fc", 0) * WEATHER_GHI_WEIGHTS["MUGLA"] +
        df.get("GHI_Denizli_W_m2_fc", 0) * WEATHER_GHI_WEIGHTS["DENIZLI"] +
        df.get("GHI_Aydin_W_m2_fc", 0) * WEATHER_GHI_WEIGHTS["AYDIN"]
    )

    # Dark_Fraction
    for name in WEATHER_STATIONS:
        ghi_col = f"{name}_GHI_fc"
        if ghi_col not in df.columns:
            continue
        for raw_prefix, out_prefix in DARK_FRAC_PREFIX.items():
            if name.startswith(raw_prefix):
                suffix = name[len(raw_prefix):]
                df[f"{out_prefix}{suffix}_Dark_Fraction_Pct_fc"] = (df[ghi_col] == 0).astype(np.int8)
                break

    return df


def to_tarih_saat(df: pd.DataFrame) -> pd.DataFrame:
    """Datetime → Tarih + Saat kolonları (master.parquet şemasıyla uyumlu).

    DİKKAT (DÜZELTME — önceki not yanlıştı): Boray'ın Eylül 2025 öncesi verisi
    "saat bitişi + wraparound" kodlaması kullanıyordu, ama Eylül 2025'ten
    itibaren (güncel/canlı döneme kadar) Saat DOĞRUDAN dt.hour'a eşit (bkz.
    01_ingest_actual.py'deki aynı not — 2025-09-15 verisiyle doğrulandı).
    """
    df = df.copy()
    df["Tarih"] = df["Datetime"].dt.normalize()
    df["Saat"]  = df["Datetime"].dt.hour
    return df.drop(columns=["Datetime"])


def _update_weather_history(result: pd.DataFrame):
    """weather_fc_live'in ilk 24 saatini weather_history.parquet'e upsert et.
    Eksik günlerin weather verisi forecast'tan gelir — actual'lar periyodik
    Archive API sync ile tamamlanır."""
    if not WEATHER_HISTORY_PARQUET.exists():
        print("     [weather_history] dosya yok, oluşturuluyor...")
        result.iloc[:24].to_parquet(WEATHER_HISTORY_PARQUET, index=False)
        return

    wh = pd.read_parquet(WEATHER_HISTORY_PARQUET)
    new_rows = result.iloc[:24]

    # Sadece history'de common olan sütunları al
    new_data = new_rows[[c for c in wh.columns if c in new_rows.columns]].copy()
    for c in wh.columns:
        if c not in new_data.columns:
            new_data[c] = None

    combined = pd.concat([wh, new_data[wh.columns]], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Tarih", "Saat"], keep="last")
    combined.to_parquet(WEATHER_HISTORY_PARQUET, index=False)
    print(f"     [weather_history] güncellendi: {len(combined)} satır")


def run() -> dict:
    """
    Adım 02 — weather forecast.

    Returns:
        {"status": "ok", "rows": N, "date_range": ["2026-06-30", "2026-07-02"]}
    """
    print("\n[02] Hava tahmini çekiliyor (Open-Meteo Forecast API)...")

    WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    raw = fetch_all_stations()
    if raw is None or raw.empty:
        raise RuntimeError("Hava verisi çekilemedi — tüm istasyonlar başarısız.")

    derived = compute_derived(raw)
    result  = to_tarih_saat(derived)

    # Sadece bugün + gelecek günler (geçmiş saatler olabilir, at)
    today_start = pd.Timestamp(date.today())
    result = result[result["Tarih"] >= today_start].reset_index(drop=True)

    WEATHER_FC_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(WEATHER_FC_PARQUET, index=False)

    # ── weather_history.parquet'i güncelle (ilk 24 saati append et) ─────────
    _update_weather_history(result)

    # Faz 0: actuals_log hava gerçekleşme dalgası (~D+6, weather_history'de dolan _actual'lar)
    try:
        wx_result = update_actuals_log_weather()
        print(f"     [ActualsLog] {wx_result}")
    except Exception as e:
        print(f"     [ActualsLog] Uyarı: {e}")

    date_min = str(result["Tarih"].min().date())
    date_max = str(result["Tarih"].max().date())
    print(f"     {len(result)} satır kaydedildi  |  {date_min} → {date_max}")
    print(f"     Çıktı: {WEATHER_FC_PARQUET.name}")

    return {
        "status": "ok",
        "rows": len(result),
        "date_range": [date_min, date_max],
    }


if __name__ == "__main__":
    result = run()
    print(result)
