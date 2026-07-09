"""backtest_tomorrow.py — GERÇEK teslim gününü ölç: YARIN (T+1, 2-gün-ileri).
=============================================================================
Production: bugün D+1, son actual D-1 (1 gün gecikme), YARIN=D teslim ediliyor.
Yani teslim edilen gün, son-actual'dan 2 gün sonra (T+0 bugün + T+1 yarın).
Bu backtest her hedef günü D için: last_actual=D-2, horizon=[D-1(T+0),D(T+1),D+1(T+2)],
ve D'yi (delivered T+1=YARIN) gerçekle kıyaslar. backtest_7d T+2'yi (öbür gün,
3-gün-ileri) ölçüyordu — yanlış gün.

Kullanım: python backtest_tomorrow.py [D1 D2 ...]
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C
import asof_regen as AR


def horizon_for_tomorrow(target_date: str):
    """target = YARIN (delivered T+1). last_actual = target-2 (T+0=target-1)."""
    tgt = pd.Timestamp(target_date)
    t0 = tgt - pd.Timedelta(days=1)   # bugün (T+0, teslim edilmez)
    t2 = tgt + pd.Timedelta(days=1)   # öbür gün (T+2)
    last_actual = (tgt - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    return last_actual, [t0, tgt, t2]


def mape(pred, act):
    i = pred.index.intersection(act.index); p, a = pred.loc[i], act.loc[i]
    v = a.notna() & p.notna()
    return float(np.mean(np.abs((a[v]-p[v])/a[v]))*100) if v.sum() else float("nan")


def run_one(target_date, backup_master):
    last_actual, horizon = horizon_for_tomorrow(target_date)
    trunc = backup_master[backup_master[C.RAW_DATE_COL] <= pd.Timestamp(last_actual)].copy()
    trunc.to_parquet(C.MASTER_PARQUET, index=False)
    if C.OOF_HISTORY_PATH.exists():
        oof = pd.read_parquet(AR.BACKUP / C.OOF_HISTORY_PATH.name); oof["date"]=pd.to_datetime(oof["date"])
        oof[oof["date"] <= pd.Timestamp(last_actual)].to_parquet(C.OOF_HISTORY_PATH, index=False)
    AR._build_synth_fc(horizon).to_parquet(C.WEATHER_FC_PARQUET, index=False)

    import importlib.util
    for modname in ["03_build_features", "04_predict_48h", "05_postprocess"]:
        spec = importlib.util.spec_from_file_location(f"p{modname}", str(ROOT / "pipeline" / f"{modname}.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); mod.run()

    post = pd.read_parquet(C.DATA_DIR / "weather_cache" / "postprocessed_predictions.parquet")
    post["Datetime"] = pd.to_datetime(post["Datetime"])
    tomorrow = pd.Timestamp(target_date)
    pr = post[post["Datetime"].dt.date == tomorrow.date()].copy()
    pr["h"] = pr["Datetime"].dt.hour; pr = pr.set_index("h")["Final_Pred"]
    a = backup_master[backup_master[C.RAW_DATE_COL].dt.date == tomorrow.date()].set_index(C.RAW_HOUR_COL)[C.RAW_TARGET_COL]
    return {"tomorrow(D)": target_date, "T+1_MAPE": round(mape(pr, a), 2),
            "pred_std": round(float(pr.std()), 0), "act_std": round(float(a.std()), 0)}


if __name__ == "__main__":
    targets = sys.argv[1:] or ["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"]
    AR._backup()
    master = pd.read_parquet(AR.BACKUP / C.MASTER_PARQUET.name); master[C.RAW_DATE_COL]=pd.to_datetime(master[C.RAW_DATE_COL])
    rows = []
    try:
        for t in targets:
            try: rows.append(run_one(t, master))
            except Exception as e: rows.append({"tomorrow(D)": t, "T+1_MAPE": None, "err": str(e)[:100]})
    finally:
        AR._restore()
    hl = C.RECENCY_HALFLIFE_DAYS if C.ENABLE_RECENCY_WEIGHTING else "KAPALI"
    print(f"\n=== GERÇEK TESLİM GÜNÜ (YARIN=T+1, 2-gün-ileri) BACKTEST | recency={hl} ===")
    df = pd.DataFrame(rows); print(df.to_string(index=False))
    v = df["T+1_MAPE"].dropna()
    if len(v): print(f"\nOrtalama YARIN MAPE: {v.mean():.2f}% | medyan: {v.median():.2f}% | en kötü: {v.max():.2f}%")
