"""
06_deliver.py — Müşteri Çıktısı + Arşiv
=========================================
48h tahmin içinden T+2 gününü (1 Temmuz) müşteri formatında yazar,
tümünü arşivler.

Giriş:  data/weather_cache/postprocessed_predictions.parquet
Çıkış:  <DELIVERY_ROOT>/YYYY.MM/D/<YYYY-MM-DD>_forecast.xlsx  (müşteri teslim dosyası,
         proje dışı paylaşılan klasör — bkz. src/output_paths.DELIVERY_ROOT)
         output/archive/<YYYY-MM-DD>_full.parquet  (tüm 48h arşiv, yerel)

NOT: Müşteri formatı şu an "basit sade tablo" olarak yazılıyor.
     Faz 3'te gerçek teslim formatına (örnek dosyaya bakarak) güncellenecek.
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config_live import (
    DATA_DIR, ARCHIVE_DIR, OUTPUT_FILENAME_TEMPLATE, DELIVERY_ROOT,
    MASTER_PARQUET, RAW_TARGET_COL, RAW_DATE_COL, RAW_HOUR_COL,
)
from src.output_paths import dated_output_path

log = logging.getLogger("adm_live")

POSTPROC_PATH = DATA_DIR / "weather_cache" / "postprocessed_predictions.parquet"

SANITY_BAND_MARGIN = 0.25   # 7-gün aynı-saat bandının ±%25 dışı -> flag
SANITY_FLAT_RATIO   = 0.15  # tahmin 24h std'si son 7g günlük std ortalamasının altında bu oranın -> flag


def _sanity_check(output_df: pd.DataFrame) -> list:
    """Teslim öncesi son kontrol: 7-gün aynı-saat bandı, düzlük, NaN/negatif.

    Engellemez (pipeline'ı durdurmaz) — sadece loglara ve summary.json'a
    GÖRÜNÜR flag düşer. Gözetimsiz sabah çalışmasında sert durdurma riskli
    (yanlış-pozitif tüm teslimi keser); bunun yerine triyaj sinyali üretir.
    """
    flags = []

    neg = output_df["Tahmin_MWh"] < 0
    if neg.any():
        flags.append({"type": "negative", "n": int(neg.sum()),
                       "hours": output_df.loc[neg, "Saat"].tolist()})
    nan = output_df["Tahmin_MWh"].isna()
    if nan.any():
        flags.append({"type": "nan", "n": int(nan.sum()),
                       "hours": output_df.loc[nan, "Saat"].tolist()})

    if not MASTER_PARQUET.exists():
        return flags
    master = pd.read_parquet(MASTER_PARQUET, columns=[RAW_DATE_COL, RAW_HOUR_COL, RAW_TARGET_COL])
    master[RAW_DATE_COL] = pd.to_datetime(master[RAW_DATE_COL])
    last_date = master[RAW_DATE_COL].max()
    recent = master[master[RAW_DATE_COL] > last_date - pd.Timedelta(days=7)]
    if recent.empty:
        return flags

    by_hour = recent.groupby(RAW_HOUR_COL)[RAW_TARGET_COL].agg(["min", "max"])
    out_of_band = []
    for _, row in output_df.iterrows():
        h = int(row["Saat"])
        if h not in by_hour.index or pd.isna(row["Tahmin_MWh"]):
            continue
        lo, hi = by_hour.loc[h, "min"], by_hour.loc[h, "max"]
        margin = (hi - lo) * SANITY_BAND_MARGIN if hi > lo else hi * SANITY_BAND_MARGIN
        if not (lo - margin <= row["Tahmin_MWh"] <= hi + margin):
            out_of_band.append(h)
    if out_of_band:
        flags.append({"type": "out_of_band", "n": len(out_of_band), "hours": out_of_band,
                       "margin_pct": SANITY_BAND_MARGIN * 100})

    pred_std = output_df["Tahmin_MWh"].std()
    recent_daily_std = recent.groupby(recent[RAW_DATE_COL].dt.date)[RAW_TARGET_COL].std().mean()
    if pd.notna(recent_daily_std) and recent_daily_std > 0 and pred_std < SANITY_FLAT_RATIO * recent_daily_std:
        flags.append({"type": "flat", "pred_std": round(float(pred_std), 1),
                       "recent_daily_std_avg": round(float(recent_daily_std), 1)})

    return flags


def run(target_date: str = None, issue_date: str | date | None = None) -> dict:
    """
    Adım 06 — teslim çıktısı yaz.

    Args:
        target_date: "YYYY-MM-DD" formatında teslim günü.
                     None ise otomatik: bugünden 1 gün sonrası (yarın).
        issue_date: bu run'ın MANTIKSAL issue günü — normal canlı run'da
                     her zaman date.today() ile aynıdır (varsayılan budur).
                     SADECE asof_regen.py gibi "geçmişteymişiz gibi" çalışan
                     script'ler için farklı olur (wall-clock date.today() o
                     script'in GERÇEKTEN çalıştığı gün, simüle ettiği gün
                     DEĞİL). Arşiv dosyasına issue_date bu parametreden
                     yazılır — forecast_log rebuild'i bunu güvenilir kaynak
                     olarak kullanır, dosya adından tahmin etmez.

    Returns:
        {"status": "ok", "output_file": "...", "target_date": "...", "n_hours": 24}
    """
    print("\n[06] Müşteri çıktısı hazırlanıyor...")
    issue_dt = pd.Timestamp(issue_date).date() if issue_date is not None else date.today()

    df = pd.read_parquet(POSTPROC_PATH)

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
    else:
        raise ValueError("Datetime kolonu bulunamadı (postprocessed_predictions.parquet)")

    # Teslim günü: T+2 = bugünden 2 gün sonra (ya da explicit)
    if target_date is None:
        today = date.today()
        target_dt = pd.Timestamp(today + timedelta(days=1))  # yarın = T+2
    else:
        target_dt = pd.Timestamp(target_date)

    # T+2 günü filtrele
    t2_mask = df["Datetime"].dt.date == target_dt.date()
    t2_df = df[t2_mask].copy()

    if t2_df.empty:
        # Tahmin penceresinde T+2 yok → tüm 48h bilgisini ver
        print(f"     [Uyarı] {target_dt.date()} için tahmin bulunamadı, tüm 48h yazılıyor.")
        t2_df = df.copy()
        target_str = "full_48h"
    else:
        target_str = str(target_dt.date())
        print(f"     Teslim günü: {target_str}  ({len(t2_df)} saat)")

    # Çıktı tablosu — şimdilik sade format (Faz 3'te müşteri formatına dönüşecek)
    output_df = pd.DataFrame({
        "Tarih":  t2_df["Datetime"].dt.date,
        "Saat":   t2_df["Datetime"].dt.hour,
        "Tahmin_MWh": t2_df["Final_Pred"].round(2),
    })
    # Savunmacı: müşteri dosyası her koşulda Saat 0..23 kronolojik olsun (upstream
    # sıra bozulsa bile). 04 artık sıralı yazıyor ama teslim çıktısını garantiye al.
    output_df = output_df.sort_values(["Tarih", "Saat"]).reset_index(drop=True)

    # Teslim-öncesi sanity bekçisi — engellemez, sadece görünür flag düşer.
    sanity_flags = _sanity_check(output_df)
    if sanity_flags:
        log.warning(f"[06] SANITY UYARI — teslim şüpheli olabilir: {sanity_flags}")
        print(f"     [Sanity] UYARI: {sanity_flags}")
    else:
        print("     [Sanity] OK")

    # Excel yaz — musteri teslimi artik proje disindaki paylasilan klasore gider.
    filename = OUTPUT_FILENAME_TEMPLATE.format(date=target_str)
    output_path = dated_output_path(DELIVERY_ROOT, target_str, filename, create=True)
    output_df.to_excel(output_path, index=False, sheet_name="Tahmin")
    print(f"     Teslim dosyası: {output_path.name}")

    # Tüm 48h arşive yaz — issue_date kolonu ACIK yazilir (dosya adindan
    # tahmin etmeye gerek kalmasin, bkz. forecast_logger.rebuild_forecast_log_from_archives).
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_name = f"{issue_dt}_run_{target_str}_full48h.parquet"
    archive_path = ARCHIVE_DIR / archive_name
    df_archive = df.copy()
    df_archive["issue_date"] = pd.Timestamp(issue_dt)
    df_archive.to_parquet(archive_path, index=False)
    print(f"     Arşiv:         {archive_path.name}")

    return {
        "status": "ok",
        "output_file": str(output_path),
        "target_date": target_str,
        "n_hours": len(t2_df),
        "sanity_flags": sanity_flags,
    }


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else None
    result = run(target_date=target)
    print(result)
