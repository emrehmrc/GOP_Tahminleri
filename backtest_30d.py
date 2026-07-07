"""backtest_30d.py — Son 30 gunun as-of T+2 backtest'i + per-model ayrinti
================================================================================
asof_regen.py makinesini yeniden kullanir. Yalnizca eksik gunleri (models_REGEN
olmayan) uretir — idempotent.

Kullanim:
    python backtest_30d.py                        # 2026-06-06 .. 2026-07-05
    python backtest_30d.py 2026-06-01 2026-06-10  # ozel aralik
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C
import asof_regen as AR

# ── utils ──────────────────────────────────────────────────────────────────────
def _models_exist(target_date: str) -> bool:
    return (C.OUTPUT_DIR / f"{target_date}_models_REGEN.parquet").exists()

def _regenerate_one(target_date: str) -> dict:
    try:
        res = AR.regen_one(target_date)
        return {"target": target_date, "status": "ok", **res}
    except Exception as e:
        return {"target": target_date, "status": "error", "err": str(e)[:200]}

# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        # range: python backtest_30d.py 2026-06-01 2026-06-10
        start, end = sys.argv[1], sys.argv[2]
        all_dates = pd.date_range(start, end)
    elif len(sys.argv) >= 2:
        # single date: python backtest_30d.py 2026-06-20
        all_dates = [pd.Timestamp(sys.argv[1])]
    else:
        # Son 30 gun: 2026-06-06 .. 2026-07-05
        all_dates = pd.date_range("2026-06-06", "2026-07-05")

    targets = [d.strftime("%Y-%m-%d") for d in all_dates]
    missing = [t for t in targets if not _models_exist(t)]

    print(f"Toplam gun: {len(targets)}  |  Mevcut: {len(targets)-len(missing)}  |  Eksik: {len(missing)}")
    if not missing:
        print("Tum gunler zaten uretilmis — cikiliyor.")
        sys.exit(0)

    print(f"Uretilecek: {missing[0]} .. {missing[-1]}")
    print("=" * 70)

    AR._backup()
    master = pd.read_parquet(AR.BACKUP / C.MASTER_PARQUET.name)
    master[C.RAW_DATE_COL] = pd.to_datetime(master[C.RAW_DATE_COL])

    results = []
    try:
        for i, t in enumerate(missing):
            stamp = f"[{i+1}/{len(missing)}]"
            result = _regenerate_one(t)
            if result["status"] == "ok":
                # MAPE hesapla
                regen_models = pd.read_parquet(C.OUTPUT_DIR / f"{t}_models_REGEN.parquet")
                regen_models["Datetime"] = pd.to_datetime(regen_models["Datetime"])
                tgt_date = pd.Timestamp(t).date()
                t2 = regen_models[regen_models["Datetime"].dt.date == tgt_date]
                act = master[master[C.RAW_DATE_COL].dt.date == tgt_date]

                act_s = act.set_index(C.RAW_HOUR_COL)[C.RAW_TARGET_COL]
                mape_vals = {}
                for col in ["XGB_Pred", "LGBM_Pred", "CAT_Pred", "CHRONOS_Pred", "Ensemble_Pred", "Final_Pred"]:
                    if col in t2.columns:
                        pred_s = t2.set_index(t2["Datetime"].dt.hour)[col]
                        i_idx = pred_s.index.intersection(act_s.index)
                        p, a = pred_s.loc[i_idx], act_s.loc[i_idx]
                        v = a.notna() & p.notna() & (a > 0)
                        if v.sum():
                            mape_vals[col] = round(float(np.mean(np.abs((a[v]-p[v])/a[v])) * 100), 2)
                result["mape"] = mape_vals
                print(f"     {stamp} {t} OK  |  T+2 MAPE: Ensemble={mape_vals.get('Ensemble_Pred','?')}%  Final={mape_vals.get('Final_Pred','?')}%")
            else:
                print(f"     {stamp} {t} HATA: {result.get('err','?')[:120]}")
            results.append(result)
    finally:
        AR._restore()

    # Ozet
    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] != "ok"]
    print(f"\n=== BACKTEST 30G SONUC ===")
    print(f"Basari: {len(ok)}  |  Hata: {len(err)}")

    if ok:
        ensemble_mapes = [r["mape"].get("Ensemble_Pred", float("nan"))
                          for r in ok if "mape" in r]
        final_mapes = [r["mape"].get("Final_Pred", float("nan"))
                       for r in ok if "mape" in r]
        print(f"Ensemble T+2 MAPE  ortalama: {np.nanmean(ensemble_mapes):.2f}%  |  medyan: {np.nanmedian(ensemble_mapes):.2f}%")
        print(f"Final    T+2 MAPE  ortalama: {np.nanmean(final_mapes):.2f}%  |  medyan: {np.nanmedian(final_mapes):.2f}%")

    print(f"\nEksiksiz 30-gun verisi output/ klasorunde hazir.")
    print("Analiz icin: python analyze_models_30d.py")
