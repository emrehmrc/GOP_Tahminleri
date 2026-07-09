"""
03_build_features.py — Feature Matrisi Kurulumu
=================================================
master.parquet (geçmiş actual) + weather_fc_live.parquet (gelecek hava) birleştirir,
Boray DataManager ile feature mühendisliğini çalıştırır.

Boray DataManager'ı değiştirmeden import ederiz; sadece veri akışını yönetiriz.
INPUT_FILE_PATH config'e yazılmış — DataManager onu okuyacak.

Çıkış:  data/weather_cache/feature_matrix.parquet  (sonraki adım okur)
         last_known_ts: str  (inference penceresi hesabı için)
"""

import os
import sys
import time
import importlib
import contextlib
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config_live import (
    MASTER_PARQUET, WEATHER_FC_PARQUET, DATA_DIR, WEATHER_HISTORY_PARQUET,
    RAW_TARGET_COL, RAW_DATE_COL, RAW_HOUR_COL,
    INPUT_FILE_PATH,
)

FEATURE_MATRIX_PATH = DATA_DIR / "weather_cache" / "feature_matrix.parquet"


@contextlib.contextmanager
def _suppress_dropna():
    """DataManager.load_and_preprocess() sonunda `df.dropna(inplace=True)`
    cagiriyor — bu, hedefi NaN olan (bizim tahmin penceremiz) satirlari
    TAMAMEN siliyor. Patch: SADECE training row'lardaki NaN'lari temizle,
    forecast row'lara (target=NaN) dokunma.
    """
    from config_live import RAW_TARGET_COL
    original = pd.DataFrame.dropna

    def _smart_dropna(self, *args, **kwargs):
        if RAW_TARGET_COL not in self.columns:
            return original(self, *args, **kwargs)
        forecast_mask = self[RAW_TARGET_COL].isna()
        forecast_rows = self[forecast_mask].copy()
        training = self[~forecast_mask]
        clean_kwargs = {k: v for k, v in kwargs.items() if k != "inplace"}
        training_clean = original(training, *args, **clean_kwargs)
        result = pd.concat([training_clean, forecast_rows]).sort_index()
        if kwargs.get("inplace"):
            self._update_inplace(result)
            return None
        return result

    pd.DataFrame.dropna = _smart_dropna
    try:
        yield
    finally:
        pd.DataFrame.dropna = original


from src.common import add_local_src_path as _add_local_src_path

def merge_actual_and_forecast() -> pd.DataFrame:
    """
    Geçmiş actual + gelecek forecast hava verisini birleştir.

    Yeni akış (raw-only master):
    - master.parquet: sadece (Tarih, Saat, ADM_Dağıtılan_Enerji_(MWh)) — 3 kolon
    - weather_history.parquet: geçmiş hava (Tarih, Saat + weather/holiday kolonları)
    - weather_fc_live.parquet: gelecek hava tahmini (_fc kolonları)
    - Eksik weather satırları (yeni ingest edilen gün) _fc'den tamamlanır
    """
    master = pd.read_parquet(MASTER_PARQUET)
    master[RAW_DATE_COL] = pd.to_datetime(master[RAW_DATE_COL]).dt.normalize()

    history = pd.read_parquet(WEATHER_HISTORY_PARQUET)
    history[RAW_DATE_COL] = pd.to_datetime(history[RAW_DATE_COL]).dt.normalize()

    # Hava kolonları için weather_history OTORİTER (Archive API ile backfill'li, tam
    # span 0 NaN). master'da da AYNI hava kolonları bulunuyor ama 2026-07-06 korupsiyon
    # kalıntısı olarak 06-29'dan itibaren AYDIN vb. NaN taşıyor. Eski merge
    # suffixes=(None,"_dup") master'ı kazandırıyordu → history'nin temiz verisi düşüyor,
    # DataManager.dropna() son ~8 günü SESSİZCE siliyordu (bayat model, silent). master'ın
    # bu bayat hava kopyalarını merge ÖNCESİ düş; history tek kaynak. Takvim/tatil kolonları
    # (yalnız master'da) ve target master'da kalır.
    weather_dup = [c for c in master.columns
                   if c in history.columns and c not in (RAW_DATE_COL, RAW_HOUR_COL)]
    if weather_dup:
        master = master.drop(columns=weather_dup)
        print(f"     Master'dan {len(weather_dup)} bayat hava kolonu düşürüldü (history otoriter)")

    historical = master.merge(history, on=[RAW_DATE_COL, RAW_HOUR_COL], how="left")

    fc = pd.read_parquet(WEATHER_FC_PARQUET)
    fc["Tarih"] = pd.to_datetime(fc["Tarih"])

    # FC kolon eşleştirmesi: history'deki _actual kolonlarına fc'deki _fc'leri eşle
    col_map = {}
    for c in history.columns:
        for suffix in ("app_temp_actual", "precip_actual", "cloud_actual"):
            if c.endswith(suffix):
                fc_col = c.replace(suffix, suffix.replace("_actual", "_fc"))
                if fc_col in fc.columns:
                    col_map[c] = fc_col
    for c in history.columns:
        if "Dark_Fraction_Pct" in c and not c.endswith("_fc"):
            fc_col = c + "_fc"
            if fc_col in fc.columns:
                col_map[c] = fc_col
    for c in history.columns:
        if c.startswith("GHI_") and not c.endswith("_fc"):
            fc_col = c + "_fc"
            if fc_col in fc.columns:
                col_map[c] = fc_col

    print(f"     FC kolon eşleşmesi: {len(col_map)} kolon")

    # Eksik weather satırlarını (yeni ingest edilen gün) _fc'den tamamla
    fc_flat = fc.rename(columns={"Tarih": RAW_DATE_COL, "Saat": RAW_HOUR_COL})
    for hist_col, fc_col in col_map.items():
        if fc_col in fc_flat.columns:
            fc_flat[hist_col] = fc_flat[fc_col]
    fc_fill = fc_flat[[RAW_DATE_COL, RAW_HOUR_COL] + list(col_map.keys())].copy()[:48]
    historical = historical.set_index([RAW_DATE_COL, RAW_HOUR_COL])
    fc_fill = fc_fill.set_index([RAW_DATE_COL, RAW_HOUR_COL])
    historical = historical.combine_first(fc_fill).reset_index()

    # Forecast satırlarını historical'a append et
    fc_work = fc.rename(columns={"Tarih": RAW_DATE_COL, "Saat": RAW_HOUR_COL})
    fc_work[RAW_TARGET_COL] = np.nan

    for hist_col, fc_col in col_map.items():
        if fc_col in fc_work.columns:
            fc_work[hist_col] = fc_work[fc_col]

    extra_cols = [c for c in fc_work.columns if c not in historical.columns]
    fc_work = fc_work.drop(columns=extra_cols, errors="ignore")
    for c in historical.columns:
        if c not in fc_work.columns:
            fc_work[c] = np.nan

    fc_work = fc_work[historical.columns]

    combined = pd.concat([historical, fc_work], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=[RAW_DATE_COL, RAW_HOUR_COL], keep="last"
    )
    combined = combined.sort_values([RAW_DATE_COL, RAW_HOUR_COL]).reset_index(drop=True)
    combined = _backfill_calendar_columns(combined)
    combined = _fill_missing_holiday_flags(combined)
    combined = _add_weekend_features(combined)
    return combined


# Master'ın taşıdığı ikili tatil/takvim bayrakları. 01_ingest yeni günü SADECE
# (Tarih, Saat, target) olarak ekler → yeni ingest edilen/forecast satırlarında bu
# bayraklar NaN kalır. DataManager çoğunu fillna(0) yapar ama `Is_Eve` listede DEĞİL —
# tek başına NaN kalıp o satırın dropna ile SESSİZCE düşmesine yol açıyordu (2026-07-07
# bayat-master bug'ının ikinci katmanı). Burada hepsini NaN→0 doldur.
# NOT: Bu, tatil OLMAYAN günler için doğru (demo penceresi 06-29→07-09 tamamen normal).
# Gelecekte forecast penceresine denk gelen resmi/dini tatiller için Ramazan_Bayram/
# Kurban_Bayram zaten _backfill_calendar_columns'ta takvimden hesaplanıyor; diğer
# bayraklar (Milli_Bayram/Yilbasi/...) hâlâ master'a bağlı — takvimden türetme Faz-2 işi.
_HOLIDAY_FLAG_COLS = [
    "Is_lockdown", "Is_Eve", "Is_Sahur", "Is_Ramadan",
    "weekday_after_bayram", "is_religional_holiday", "After_Bayram",
    "Yilbasi", "before_yilbasi", "weekday_after_yilbasi",
    "Secim_Gunu", "Milli_Bayram",
]


def _fill_missing_holiday_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Master'dan gelen ikili tatil bayraklarının eksiklerini (yeni ingest/forecast
    satırları) 0 ile doldur — aksi halde `Is_Eve` gibi tek bir NaN, güncel eğitim
    satırlarını sessizce düşürüyor."""
    for col in _HOLIDAY_FLAG_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


def _backfill_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Yıl/Ay/Gün/Haftanın_Günü/Ramazan_Bayram/Kurban_Bayram — bu kolonlar
    ham Tarih'e göre DEĞİL, DataManager'ın kendi rekonstrükte ettiği Datetime'a
    göre (Tarih.normalize() + Saat 0->24 düzeltmesi) doldurulmuş olmalı.
    Doğrulandı: gerçek parquette Saat=0 (Tarih=D) satırının Yıl/Ay/Gün/
    Ramazan_Bayram değerleri D+1'in takvim/tatil bilgisini taşıyor.
    TÜM satırlar doldurulur (bug fix: önceki kod sütun yoksa return ediyordu).
    """
    calendar_cols = ["Yıl", "Ay", "Gün", "Haftanın_Günü", "Ramazan_Bayram", "Kurban_Bayram"]
    for col in calendar_cols:
        if col not in df.columns:
            df[col] = np.nan

    needs_fill = df[calendar_cols].isna().any(axis=1)
    if not needs_fill.any():
        return df

    corrected_hours = df.loc[needs_fill, RAW_HOUR_COL].replace(0, 24)
    recon_dt = df.loc[needs_fill, RAW_DATE_COL].dt.normalize() + pd.to_timedelta(corrected_hours, unit="h")

    df.loc[needs_fill, "Yıl"] = recon_dt.dt.year
    df.loc[needs_fill, "Ay"] = recon_dt.dt.month
    df.loc[needs_fill, "Gün"] = recon_dt.dt.day
    df.loc[needs_fill, "Haftanın_Günü"] = recon_dt.dt.isocalendar()["day"].astype(int).values

    _add_local_src_path()
    from src.holiday_calendar import build_holiday_calendar

    years = sorted(set(recon_dt.dt.year) | {int(recon_dt.dt.year.min()) - 1, int(recon_dt.dt.year.max()) + 1})
    cal = build_holiday_calendar(years)

    ramazan_vals, kurban_vals = [], []
    for d in recon_dt.dt.date:
        meta = cal.get(d)
        if meta and meta["category"] == "Ramazan_Bayram":
            ramazan_vals.append(meta["holiday_day_number"])
            kurban_vals.append(0)
        elif meta and meta["category"] == "Kurban_Bayram":
            ramazan_vals.append(0)
            kurban_vals.append(meta["holiday_day_number"])
        else:
            ramazan_vals.append(0)
            kurban_vals.append(0)

    df.loc[needs_fill, "Ramazan_Bayram"] = ramazan_vals
    df.loc[needs_fill, "Kurban_Bayram"] = kurban_vals

    return df


def _add_weekend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hafta sonu binary feature'ları ve Pazar/geçmiş işgünü oranı."""
    if "Haftanın_Günü" not in df.columns:
        return df

    df["Is_Weekend"]   = df["Haftanın_Günü"].isin([6, 7]).astype(np.int8)
    df["Is_Sunday"]    = (df["Haftanın_Günü"] == 7).astype(np.int8)
    df["Is_Saturday"]  = (df["Haftanın_Günü"] == 6).astype(np.int8)

    return df


def run() -> dict:
    """
    Adım 03 — feature matrisini kur.

    Returns:
        {"status": "ok", "n_rows": N, "n_features": F, "last_actual_ts": "..."}
    """
    print("\n[03] Feature matrisi kuruluyor...")

    _add_local_src_path()

    combined = merge_actual_and_forecast()

    actual_mask = combined[RAW_TARGET_COL].notna()
    last_actual = combined.loc[actual_mask, RAW_DATE_COL].iloc[-1]
    last_actual_hour = int(combined.loc[actual_mask, RAW_HOUR_COL].iloc[-1])
    last_actual_ts = f"{last_actual.date()} {last_actual_hour:02d}:00"
    print(f"     Son actual: {last_actual_ts}")

    tmp_xlsx = DATA_DIR / "weather_cache" / "_tmp_combined.xlsx"
    tmp_parquet = DATA_DIR / "weather_cache" / "_tmp_combined.parquet"
    combined.to_parquet(tmp_parquet, index=False)

    if not tmp_xlsx.exists():
        tmp_xlsx.touch()
    old_time = time.time() - 86400
    os.utime(tmp_xlsx, (old_time, old_time))

    import config_live
    config_live.INPUT_FILE_PATH = str(tmp_xlsx)

    try:
        if "src.data_manager" in sys.modules:
            importlib.reload(sys.modules["src.data_manager"])
        from src.data_manager import DataManager
        dm = DataManager()
        with _suppress_dropna():
            dm.load_and_preprocess()
        feature_df = dm.data
    finally:
        config_live.INPUT_FILE_PATH = str(DATA_DIR / "weather_cache" / "_tmp_combined.xlsx")

    # Özellik matrisini kaydet
    FEATURE_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # GHI x Sicaklik etkilesim feature'lari (Faz 2.2)
    ghi_cols = [c for c in feature_df.columns if 'GHI' in c and 'Weighted' in c]
    temp_cols = [c for c in feature_df.columns if 'Hissedilen' in c and 'Mean' in c]
    if ghi_cols and temp_cols:
        ghi = feature_df[ghi_cols[0]]
        temp = feature_df[temp_cols[0]]
        feature_df['GHI_Temp_Interaction'] = ghi * temp / 100
        feature_df['Hot_Solar_GHI'] = ((temp > 28) & (feature_df['Saat'].between(11, 16))).astype(float) * ghi
        feature_df['Cold_Solar_GHI'] = ((temp < 10) & (feature_df['Saat'].between(10, 15))).astype(float) * ghi
        print(f"     GHIxSicaklik etkilesim feature'lari eklendi: GHI={ghi_cols[0]}, Temp={temp_cols[0]}")

    feature_df_old = feature_df  # rename for NaN guard
    feature_df = feature_df.copy()  # de-fragment
    feature_df.to_parquet(FEATURE_MATRIX_PATH)
    feature_df = feature_df_old  # restore for guards below

    # NaN GUARD — training rows'da NaN feature varsa pipeline durur
    train_mask = feature_df[RAW_TARGET_COL].notna()
    if train_mask.any():
        nan_count = feature_df.loc[train_mask].isna().sum().sum()
        if nan_count > 0:
            nan_cols = feature_df.loc[train_mask].isna().sum()
            nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
            msg = f"CRITICAL: {nan_count} NaN cells in training data!\n"
            for c, n in nan_cols.head(10).items():
                msg += f"  {c}: {n} NaN\n"
            raise RuntimeError(msg)
        print("     NaN guard: OK (0 NaN)")

    # FRESHNESS GUARD — en yeni actual günü eğitim setinde OLMALI. Bir hava/merge
    # sorunu son günleri sessizce düşürürse (NaN guard 0 NaN der çünkü satır silinir,
    # doldurulmaz) model bayat kalır ve pipeline yeşil raporlar. Bu durumu GÜRÜLTÜLÜ
    # yakala. (bkz. 2026-07-07 bayat-master bug'ı.)
    if train_mask.any():
        last_actual_day = pd.Timestamp(last_actual).normalize()
        train_days = pd.DatetimeIndex(feature_df.loc[train_mask].index).normalize()
        train_max_day = train_days.max()
        # hour-ending (Saat 0 -> ertesi gün 00:00) kaymasına karşı 1 gün tolerans
        if (last_actual_day - train_max_day).days > 1:
            raise RuntimeError(
                f"CRITICAL: en yeni actual {last_actual_day.date()} ama eğitim "
                f"{train_max_day.date()}'te bitiyor — son günler sessizce düşmüş "
                f"(bayat feature). Hava/merge kaynağını kontrol et."
            )
        print(f"     Freshness guard: OK (eğitim son gün {train_max_day.date()}, actual {last_actual_day.date()})")

    n_features = feature_df.select_dtypes(include=["number", "category", "bool"]).shape[1]
    print(f"     {len(feature_df)} satır  |  {n_features} feature  |  kayıt: {FEATURE_MATRIX_PATH.name}")

    return {
        "status": "ok",
        "n_rows": len(feature_df),
        "n_features": n_features,
        "last_actual_ts": last_actual_ts,
    }


if __name__ == "__main__":
    result = run()
    print(result)
