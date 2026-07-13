"""backtest_7d.py — Son N günün as-of T+2 backtest'i (deney ortamı)
====================================================================
Her hedef gün için: master'ı (hedef-3)'e kes, ufuk havasını weather_history'den
(perfect-prog) kur, GÜNCEL pipeline'ı (03-06) çalıştır, T+2 tahminini o günün
GERÇEĞİYLE kıyasla. Tek backup/restore; canlı state korunur. asof_regen'in
makinesini yeniden kullanır.

Kullanım:
    python backtest_7d.py                       # 2026-06-28 .. 2026-07-04
    python backtest_7d.py 2026-07-01 2026-07-04
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C
from src.output_paths import resolve_output_file
import asof_regen as AR


def _actual_series(backup_master: pd.DataFrame, target_date: str) -> pd.Series:
    d = pd.to_datetime(target_date).date()
    s = backup_master[backup_master[C.RAW_DATE_COL].dt.date == d]
    return s.set_index(C.RAW_HOUR_COL)[C.RAW_TARGET_COL]


def _mape(pred: pd.Series, act: pd.Series) -> float:
    i = pred.index.intersection(act.index)
    p, a = pred.loc[i], act.loc[i]
    v = a.notna() & p.notna()
    if v.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((a[v] - p[v]) / a[v])) * 100)


def _me(pred: pd.Series, act: pd.Series) -> float:
    i = pred.index.intersection(act.index)
    p, a = pred.loc[i], act.loc[i]
    v = a.notna() & p.notna()
    return float(np.mean(p[v] - a[v])) if v.sum() else float("nan")


if __name__ == "__main__":
    targets = sys.argv[1:] or [
        "2026-06-28", "2026-06-29", "2026-06-30",
        "2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04",
    ]
    AR._backup()
    master = pd.read_parquet(AR.BACKUP / C.MASTER_PARQUET.name)
    master[C.RAW_DATE_COL] = pd.to_datetime(master[C.RAW_DATE_COL])

    rows = []
    try:
        for t in targets:
            try:
                res = AR.regen_one(t)  # full as-of pipeline; REGEN dosyası yazar
                regen = pd.read_excel(resolve_output_file(C.OUTPUT_DIR, f"{t}_forecast_REGEN.xlsx"),
                                      sheet_name="Tahmin").set_index("Saat")["Tahmin_MWh"]
                act = _actual_series(master, t)
                rows.append({"target": t, "T+2_MAPE": round(_mape(regen, act), 2),
                             "ME": round(_me(regen, act), 0),
                             "pred_std": round(float(regen.std()), 0),
                             "act_std": round(float(act.std()), 0)})
            except Exception as e:
                rows.append({"target": t, "T+2_MAPE": None, "ME": None,
                             "pred_std": None, "act_std": None, "err": str(e)[:120]})
    finally:
        AR._restore()

    hl = getattr(C, "RECENCY_HALFLIFE_DAYS", None)
    print(f"\n=== SON {len(targets)} GÜN as-of T+2 BACKTEST (recency halflife={hl}g) ===")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    valid = df["T+2_MAPE"].dropna()
    if len(valid):
        print(f"\nOrtalama T+2 MAPE: {valid.mean():.2f}%  | medyan: {valid.median():.2f}%  | en kötü: {valid.max():.2f}%")
