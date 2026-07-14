"""experiments/oof_backfill_pilot.py — F0.1 ADM OOF backfill (2026-07-14, genişletildi).

MODEL_AUDIT_VE_OPTIMIZASYON_PLANI_2026-07-13.md §E-F0.1: ensemble turnuvasına (F2)
horizon-ayrımlı as-of OOF verisi sağlamak. Canlı `data/oof_history.parquet` sadece
~5 gün/120 satır ve horizon kolonu YOK (B3 bulgusu — `_find_archive_for_date` en
yeni arşivi seçtiği için OOF fiilen sadece T+1 hatasını yakalıyordu, öğrenilen
ağırlıklar T+2'ye de uygulanıyordu).

`output/YYYY.MM/GG/<tarih>_models_REGEN.parquet` dosyaları (asof_regen sandbox
çıktısı, 35 gün mevcut, 2026-06-10'dan itibaren) ZATEN her satırda `horizon_day`
('T+1'/'T+2') taşıyor — bu script sadece bu dosyaları tarar, master'daki actual'larla
eşleştirir, chronos-fallback bayrağını korur ve uzun formatta tek parquet'e yazar.

Çıktı: `data/oof_history_backfill.parquet` (K2: canlı `data/oof_history.parquet`'e
ASLA yazılmaz — ayrı dosya, F2 turnuvasının girdisi).

2026-07-13 5 günlük dry-run pilotu (`_find_archive_for_date` + `_records_for_day`
üzerinden) bu genişletilmiş sürümde de PILOT_DATES/_oof_pilot.parquet olarak
korunuyor (hızlı mekanizma doğrulaması için) ama asıl F0.1 çıktısı `run_full_backfill()`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C
from src.oof_feedback import _records_for_day, _find_archive_for_date, _is_chronos_fallback

PILOT_DATES = ["2026-07-08", "2026-07-09", "2026-07-10", "2026-07-11", "2026-07-12"]
PILOT_OOF_PATH = C.DATA_DIR / "_oof_pilot.parquet"

BACKFILL_OOF_PATH = C.DATA_DIR / "oof_history_backfill.parquet"
PRED_COLS = ["XGB_Pred", "LGBM_Pred", "CAT_Pred", "CHRONOS_Pred", "Ensemble_Pred"]


def _records_for_regen_file(regen_path: Path, master: pd.DataFrame) -> list[dict]:
    """Tek bir *_models_REGEN.parquet dosyasından (T+1+T+2, 48 satır) horizon-etiketli
    OOF kayıtları üretir. `horizon_day` REGEN dosyasında zaten mevcut."""
    df = pd.read_parquet(regen_path)
    if "Datetime" not in df.columns or "horizon_day" not in df.columns:
        return []
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["_date"] = df["Datetime"].dt.strftime("%Y-%m-%d")
    df["_hour"] = df["Datetime"].dt.hour

    records: list[dict] = []
    for (target_date, horizon), block in df.groupby(["_date", "horizon_day"]):
        day_actuals = master[master[C.RAW_DATE_COL] == pd.Timestamp(target_date)].sort_values(C.RAW_HOUR_COL)
        if day_actuals.empty:
            continue
        day_actuals_dict = {int(r[C.RAW_HOUR_COL]): r[C.RAW_TARGET_COL] for _, r in day_actuals.iterrows()}
        chronos_fallback = _is_chronos_fallback(block)
        for _, row in block.iterrows():
            h = int(row["_hour"])
            actual = day_actuals_dict.get(h)
            if actual is None or (isinstance(actual, float) and np.isnan(actual)):
                continue
            rec = {
                "date": target_date, "hour": h, "horizon": horizon,
                "actual": float(actual), "chronos_fallback": chronos_fallback,
                "source_file": regen_path.name,
            }
            for col in PRED_COLS:
                if col in row and pd.notna(row[col]):
                    rec[col] = float(row[col])
            records.append(rec)
    return records


def run_full_backfill() -> pd.DataFrame:
    """F0.1: output/**/*_models_REGEN.parquet taranır (35 gün mevcut), her satır
    horizon-etiketli OOF kaydına dönüşür, aynı (date,hour,horizon) için birden
    fazla REGEN dosyasından gelen kayıtlar arasında EN YENİ dosya kazanır (asof_regen
    sandbox re-run'ları arasında idempotent-son-yazan-kazanır)."""
    master = pd.read_parquet(C.MASTER_PARQUET)
    master[C.RAW_DATE_COL] = pd.to_datetime(master[C.RAW_DATE_COL])

    regen_files = sorted((ROOT / "output").glob("**/*_models_REGEN.parquet"))
    print(f"[F0.1] {len(regen_files)} REGEN dosyası bulundu.")

    all_records: list[dict] = []
    for f in regen_files:
        recs = _records_for_regen_file(f, master)
        print(f"  {f.relative_to(ROOT)}: {len(recs)} satır")
        all_records.extend(recs)

    if not all_records:
        raise SystemExit("[F0.1] Hiç kayıt üretilemedi — REGEN dosyaları veya master boş mu kontrol et.")

    oof = pd.DataFrame(all_records)
    # (date,hour,horizon) tekilliği: birden fazla REGEN dosyası aynı satırı üretmişse
    # dosya adındaki tarih (en yeni asof-regen koşusu) en güvenilir kabul edilir.
    oof = oof.sort_values("source_file").drop_duplicates(subset=["date", "hour", "horizon"], keep="last")
    oof = oof.sort_values(["date", "horizon", "hour"]).reset_index(drop=True)

    BACKFILL_OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(BACKFILL_OOF_PATH, index=False)

    n_days = oof["date"].nunique()
    n_t1 = (oof["horizon"] == "T+1").sum()
    n_t2 = (oof["horizon"] == "T+2").sum()
    n_fallback = int(oof["chronos_fallback"].sum())
    print(f"\n[F0.1] Yazıldı: {BACKFILL_OOF_PATH}")
    print(f"  Toplam satır: {len(oof)} | Benzersiz gün: {n_days} | T+1: {n_t1} | T+2: {n_t2}")
    print(f"  chronos_fallback=True satır: {n_fallback} ({n_fallback/len(oof):.2%})")

    for horizon, grp in oof.groupby("horizon"):
        if "Ensemble_Pred" in grp.columns:
            d = grp.dropna(subset=["Ensemble_Pred", "actual"])
            mape = float(np.mean(np.abs((d["actual"] - d["Ensemble_Pred"]) / (d["actual"] + 1e-10))) * 100)
            print(f"  {horizon}: Ensemble_Pred MAPE = {mape:.2f}% ({len(d)} satır)")

    return oof


def run_pilot() -> None:
    """2026-07-13 mekanizma-doğrulama pilotu (5 gün, dry-run, _oof_pilot.parquet) —
    korunuyor, run_full_backfill()'in yerine geçmiyor."""
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


if __name__ == "__main__":
    if "--pilot" in sys.argv:
        run_pilot()
    else:
        run_full_backfill()
