"""fix_weather_history.py — weather_history.parquet'teki reanalysis boşluklarını
Open-Meteo Archive API'den GERÇEK (gözlemlenmiş) hava ile doldurur.

NEDEN: `weather_history.parquet`'in `_actual` kolonları normal akışta periyodik bir
Archive API sync ile dolar (bkz. pipeline/02_fetch_weather.py:_update_weather_history
docstring'i — "actual'lar periyodik Archive API sync ile tamamlanır"), ama bu sync
script repoda YOKTU. Bu boşluk 2026-07-06 oturumunda tespit edildi: 07-03 tarihi
weather_history'de HİÇ satır olarak yoktu, 07-04 tamamen NaN'dı — bu, o tarihleri
ufkunda barındıran as-of backtest'lerin "perfect-prog" varsayımını bozup interpolasyonla
uydurulmuş hava kullanmasına (ve MAPE'yi yapay olarak şişirmesine) yol açtı.

Kullanım:
    python fix_weather_history.py 2026-07-03 2026-07-06
    (parametresiz: son 7 gün)
"""
from __future__ import annotations
import sys, time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from config_live import WEATHER_HISTORY_PARQUET, WEATHER_STATIONS, WEATHER_GHI_WEIGHTS, WEATHER_TIMEZONE

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "apparent_temperature,precipitation,cloud_cover,shortwave_radiation"

PROVINCE_STATIONS = {
    "Mugla":   [s for s in WEATHER_STATIONS if s.startswith("MUGLA_")],
    "Denizli": [s for s in WEATHER_STATIONS if s.startswith("DENIZLI_")],
    "Aydin":   [s for s in WEATHER_STATIONS if s.startswith("AYDIN_")],
}
DARK_FRAC_PREFIX = {"MUGLA_": "MUGLA_", "DENIZLI_": "DNZ_", "AYDIN_": "AYD_"}


def _fetch_station(name: str, lat: float, lon: float, start: str, end: str, retries: int = 4) -> pd.DataFrame | None:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": HOURLY_VARS, "timezone": WEATHER_TIMEZONE,
    }
    for attempt in range(retries):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=60)
            r.raise_for_status()
            h = r.json()["hourly"]
            return pd.DataFrame({
                "Datetime": pd.to_datetime(h["time"]),
                f"{name}_app_temp_actual": h["apparent_temperature"],
                f"{name}_precip_actual":   h["precipitation"],
                f"{name}_cloud_actual":    h["cloud_cover"],
                f"{name}_GHI_actual":      h["shortwave_radiation"],
            })
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"   [{name}] hata ({e}), {wait}s bekleniyor...")
                time.sleep(wait)
            else:
                print(f"   [{name}] {retries} denemede basarisiz: {e}")
                return None


def fetch_all_stations(start: str, end: str) -> pd.DataFrame:
    combined = None
    for i, (name, (lat, lon)) in enumerate(WEATHER_STATIONS.items()):
        print(f"   [{i+1}/{len(WEATHER_STATIONS)}] {name}")
        df = _fetch_station(name, lat, lon, start, end)
        if df is None:
            print(f"   [Uyari] {name} atlandi!")
            continue
        combined = df if combined is None else combined.merge(df, on="Datetime", how="outer")
        time.sleep(0.3)
    return combined


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    for prov, stations in PROVINCE_STATIONS.items():
        cols = [f"{s}_GHI_actual" for s in stations if f"{s}_GHI_actual" in df.columns]
        if cols:
            df[f"GHI_{prov}_W_m2"] = df[cols].mean(axis=1)

    df["GHI_ADM_Weighted"] = (
        df.get("GHI_Mugla_W_m2", 0) * WEATHER_GHI_WEIGHTS["MUGLA"]
        + df.get("GHI_Denizli_W_m2", 0) * WEATHER_GHI_WEIGHTS["DENIZLI"]
        + df.get("GHI_Aydin_W_m2", 0) * WEATHER_GHI_WEIGHTS["AYDIN"]
    )

    for name in WEATHER_STATIONS:
        ghi_col = f"{name}_GHI_actual"
        if ghi_col not in df.columns:
            continue
        for raw_prefix, out_prefix in DARK_FRAC_PREFIX.items():
            if name.startswith(raw_prefix):
                suffix = name[len(raw_prefix):]
                df[f"{out_prefix}{suffix}_Dark_Fraction_Pct"] = (df[ghi_col] == 0).astype(np.int8)
                break
    # DÜZELTME (2026-07-06): istasyon-bazlı _GHI_actual kolonları BURADA
    # SİLİNİYORDU — weather_history.parquet'in mevcut şeması bu kolonları
    # gerçek veri olarak taşıyor (DataManager/03_build_features feature
    # matrisinde kullanıyor). Silinince yeni eklenen/güncellenen günlerde bu
    # kolonlar kalıcı NaN kalıyordu (07-02..07-05 boşluğu, NaN guard'ı
    # tetikledi). Aggregate GHI_ADM_Weighted zaten hesaplandı; ham kolonları
    # da koru.
    return df


def run(start: str, end: str) -> dict:
    print(f"\n[fix_weather_history] Open-Meteo Archive API: {start} -> {end}")
    raw = fetch_all_stations(start, end)
    if raw is None or raw.empty:
        raise RuntimeError("Hava verisi cekilemedi - tum istasyonlar basarisiz.")

    derived = compute_derived(raw)
    derived["Tarih"] = derived["Datetime"].dt.normalize()
    derived["Saat"] = derived["Datetime"].dt.hour
    derived = derived.drop(columns=["Datetime"])

    wh = pd.read_parquet(WEATHER_HISTORY_PARQUET)
    wh["Tarih"] = pd.to_datetime(wh["Tarih"])

    wx_cols = [c for c in wh.columns if c in derived.columns and c not in ("Tarih", "Saat")]
    new_data = derived[["Tarih", "Saat"] + wx_cols].drop_duplicates(subset=["Tarih", "Saat"])

    wh_idx = wh.set_index(["Tarih", "Saat"])
    new_idx = new_data.set_index(["Tarih", "Saat"])

    # var olan satirlar: hava kolonlarini YENI (gercek) degerle guncelle (update()
    # sadece new'de NaN OLMAYAN degerleri yazar, eskiyi NaN ile ezmez).
    wh_idx.update(new_idx)

    # weather_history'de HIC satiri olmayan (Tarih,Saat) -> yeni satir olarak ekle
    # (ornek: 07-03 komple eksikti). Diger (hava-disi) kolonlar NaN kalir.
    missing_keys = new_idx.index.difference(wh_idx.index)
    if len(missing_keys):
        add_rows = new_idx.loc[missing_keys].reindex(columns=wh_idx.columns)
        wh_idx = pd.concat([wh_idx, add_rows])

    combined_final = wh_idx.sort_index().reset_index()
    combined_final.to_parquet(WEATHER_HISTORY_PARQUET, index=False)
    n_days = derived["Tarih"].nunique()
    print(f"     weather_history guncellendi: {len(combined_final)} satir ({n_days} gun icin hava yazildi)")
    return {"status": "ok", "rows": len(combined_final), "days_fetched": n_days}


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        start_d, end_d = sys.argv[1], sys.argv[2]
    else:
        end_d = date.today().isoformat()
        start_d = (date.today() - timedelta(days=7)).isoformat()
    result = run(start_d, end_d)
    print(result)
