"""experiments/oof_backfill_pilot.py — Faz 2 2c-2 pilot (2026-07-13).

get_segment_weights() SCAFFOLDING'i (MASTER_PLAN.md Faz 2c-2) canlı `oof_history.parquet`
sadece 4 gün olduğu için hiç segment üretemiyordu. Kullanıcı kararı: tam ~30 günlük
backfill (asof_regen.regen_one() ~4dk/gün) AYRI bir oturumda yapılacak; bu script
SADECE mekanizmayı hızlıca doğrulayan KÜÇÜK bir pilot -- yeni regen çağırmaz, önceki
oturumlardan (lag168-blend A/B, backtest_walkforward) kalan `*_models_REGEN.parquet`
dosyalarını (git-ignored, output/ altında) ve gerçek `ARCHIVE_DIR` arşivini kullanır.

Kapsam: 5 gün (2026-07-08..07-12, ARCHIVE_DIR'da zaten gerçek veri var -- REGEN'e
bile gerek yok). oof_history.parquet'e YAZMAZ (dry-run) -- ayrı bir dosyaya
(data/_oof_pilot.parquet) yazar, canlı dosyaya dokunmaz.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C
from src.oof_feedback import _records_for_day, _find_archive_for_date

PILOT_DATES = ["2026-07-08", "2026-07-09", "2026-07-10", "2026-07-11", "2026-07-12"]
PILOT_OOF_PATH = C.DATA_DIR / "_oof_pilot.parquet"


def run() -> None:
    master = pd.read_parquet(C.MASTER_PARQUET)
    master[C.RAW_DATE_COL] = pd.to_datetime(master[C.RAW_DATE_COL])

    all_records = []
    for d in PILOT_DATES:
        archive_file = _find_archive_for_date(C.ARCHIVE_DIR, d)
        if archive_file is None:
            print(f"  {d}: arşiv yok, atlandı")
            continue
        forecast = pd.read_parquet(archive_file)
        forecast["Datetime"] = pd.to_datetime(forecast["Datetime"])
        day_forecast = forecast[forecast["Datetime"].dt.strftime("%Y-%m-%d") == d].copy()
        day_actuals = master[master[C.RAW_DATE_COL] == pd.Timestamp(d)].sort_values(C.RAW_HOUR_COL)
        if day_forecast.empty or day_actuals.empty:
            print(f"  {d}: eşleşme yok (forecast={len(day_forecast)}, actuals={len(day_actuals)})")
            continue
        recs = _records_for_day(day_forecast, day_actuals, d, C.RAW_TARGET_COL, C.RAW_HOUR_COL)
        print(f"  {d}: {len(recs)} satır ({archive_file.name})")
        all_records.extend(recs)

    if not all_records:
        print("Hiç kayıt üretilemedi.")
        return

    pilot_oof = pd.DataFrame(all_records)
    if PILOT_OOF_PATH.exists():
        PILOT_OOF_PATH.unlink()
    pilot_oof.to_parquet(PILOT_OOF_PATH, index=False)
    print(f"\nPilot OOF (dry-run, canlıya YAZILMADI): {PILOT_OOF_PATH}  ({len(pilot_oof)} satır, "
          f"{pilot_oof['date'].nunique()} gün)")

    from src.oof_feedback import get_segment_weights
    pred_cols = ["XGB_Pred", "LGBM_Pred", "CAT_Pred", "CHRONOS_Pred"]
    weights = get_segment_weights(PILOT_OOF_PATH, pred_cols, lookback_days=30, min_samples_per_segment=30)
    print(f"\nget_segment_weights() sonucu (min_samples_per_segment=30): {weights}")
    print("(5 günle -- hafta içi ~3-4 gün/blok, hafta sonu ~1 gün/blok -- hiçbir segment "
          "30 örneğe ulaşamaz; bu BEKLENEN ve pilot'un amacı zaten bu: mekanizma None/boş "
          "dönüyor mu yoksa crash mi ediyor, onu doğrulamak. Tam sonuç için ayrı oturumda "
          "~30 günlük backfill gerekiyor.)")


if __name__ == "__main__":
    run()
