"""
06_deliver.py — Müşteri Çıktısı + Arşiv
=========================================
48h tahmin içinden T+2 gününü (1 Temmuz) müşteri formatında yazar,
tümünü arşivler.

Faz 3 (2026-07-13): ince kabuk — boilerplate (sanity check, excel yazımı,
arşivleme) ortak src/deliver_core.py'ye taşındı, sadece T+2 seçim
stratejisi (tarih filtresi + boşsa 48h fallback) burada kalıyor. GDZ
karşılığı pipeline/06_deliver.py aynı çekirdeği kendi stratejisiyle çağırır.

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
from src.deliver_core import run_delivery

log = logging.getLogger("adm_live")

POSTPROC_PATH = DATA_DIR / "weather_cache" / "postprocessed_predictions.parquet"


def _select_t2(df: pd.DataFrame, target_date: str | None):
    """ADM: T+2 = bugünden 1 gün sonra (yarın), ya da explicit target_date."""
    if target_date is None:
        today = date.today()
        target_dt = pd.Timestamp(today + timedelta(days=1))
    else:
        target_dt = pd.Timestamp(target_date)

    t2_mask = df["Datetime"].dt.date == target_dt.date()
    t2_df = df[t2_mask].copy()
    if t2_df.empty:
        print(f"     [Uyarı] {target_dt.date()} için tahmin bulunamadı, tüm 48h yazılıyor.")
        return df.copy(), "full_48h"
    return t2_df, str(target_dt.date())


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
    return run_delivery(
        postproc_path=POSTPROC_PATH,
        select_t2=lambda df: _select_t2(df, target_date),
        master_path=MASTER_PARQUET,
        date_col=RAW_DATE_COL,
        target_col=RAW_TARGET_COL,
        hour_col=RAW_HOUR_COL,
        delivery_root=DELIVERY_ROOT,
        archive_dir=ARCHIVE_DIR,
        output_filename_template=OUTPUT_FILENAME_TEMPLATE,
        dated_output_path_fn=dated_output_path,
        log=log,
        label="",
        issue_date=issue_date,
    )


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    result = run(target_date=target)
    print(result)
