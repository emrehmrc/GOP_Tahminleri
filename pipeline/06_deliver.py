"""
06_deliver.py — Müşteri Çıktısı + Arşiv
=========================================
48h tahmin içinden T+2 gününü (1 Temmuz) müşteri formatında yazar,
tümünü arşivler.

Giriş:  data/weather_cache/postprocessed_predictions.parquet
Çıkış:  output/<YYYY-MM-DD>_forecast.xlsx   (müşteri teslim dosyası)
         output/archive/<YYYY-MM-DD>_full.parquet  (tüm 48h arşiv)

NOT: Müşteri formatı şu an "basit sade tablo" olarak yazılıyor.
     Faz 3'te gerçek teslim formatına (örnek dosyaya bakarak) güncellenecek.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config_live import (
    DATA_DIR, OUTPUT_DIR, ARCHIVE_DIR, OUTPUT_FILENAME_TEMPLATE,
)

POSTPROC_PATH = DATA_DIR / "weather_cache" / "postprocessed_predictions.parquet"


def run(target_date: str = None) -> dict:
    """
    Adım 06 — teslim çıktısı yaz.

    Args:
        target_date: "YYYY-MM-DD" formatında teslim günü.
                     None ise otomatik: bugünden 1 gün sonrası (yarın = T+2).

    Returns:
        {"status": "ok", "output_file": "...", "target_date": "...", "n_hours": 24}
    """
    print("\n[06] Müşteri çıktısı hazırlanıyor...")

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

    # Excel yaz
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = OUTPUT_FILENAME_TEMPLATE.format(date=target_str)
    output_path = OUTPUT_DIR / filename
    output_df.to_excel(output_path, index=False, sheet_name="Tahmin")
    print(f"     Teslim dosyası: {output_path.name}")

    # Tüm 48h arşive yaz
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_name = f"{date.today()}_run_{target_str}_full48h.parquet"
    archive_path = ARCHIVE_DIR / archive_name
    df.to_parquet(archive_path, index=False)
    print(f"     Arşiv:         {archive_path.name}")

    return {
        "status": "ok",
        "output_file": str(output_path),
        "target_date": target_str,
        "n_hours": len(t2_df),
    }


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else None
    result = run(target_date=target)
    print(result)
