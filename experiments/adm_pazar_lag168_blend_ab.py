"""experiments/adm_pazar_lag168_blend_ab.py — Faz 2 2b-2 (2026-07-13) walkforward A/B.

Hipotez (MASTER_PLAN.md §2b-2): ADM Pazar tahminini `alpha*pred_final +
(1-alpha)*lag168_actual` ile blend etmek MAPE'yi düşürür mü? Feature importance
incelemesi (ADM XGB weekend modeli) Lag24h'nin Lag168h'den çok daha ağırlıklı
olduğunu gösterdi — model Cumartesi'nin desenine Pazar'dan daha çok güveniyor.

Yöntem: asof_regen.regen_one() ile (GERÇEK dosyalara DOKUNMAYAN, tamamen
sandbox'lı, günlük-yeniden-eğitilmiş modellerle as-of üretim) 4 tarihsel Pazar
için REGEN tahminleri üretildi (bu script'ten AYRI, önceden çalıştırıldı --
output/2026.0X/XX/<tarih>_models_REGEN.parquet, git-ignored). Bu script SADECE
o REGEN dosyalarını + master.parquet'i okur, hiçbir canlı dosyaya yazmaz.

Governance (MASTER_PLAN.md): bu raporun sonucu OLUMLU olsa bile canlıya
otomatik girmez -- kullanıcı onayı + config flag'i ayrı bir adım.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C

SUNDAYS = ["2026-06-14", "2026-06-21", "2026-06-28", "2026-07-05"]
ALPHAS = [round(a, 2) for a in np.arange(0.0, 1.01, 0.1)]


def _regen_path(target_date: str) -> Path:
    y, m, d = target_date.split("-")
    return C.OUTPUT_DIR / f"{y}.{m}" / str(int(d)) / f"{target_date}_models_REGEN.parquet"


def mape(pred: pd.Series, actual: pd.Series) -> float:
    v = actual.notna() & pred.notna() & (actual != 0)
    if v.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[v] - pred[v]) / actual[v])) * 100)


def load_day(target_date: str, master: pd.DataFrame) -> pd.DataFrame:
    regen_path = _regen_path(target_date)
    if not regen_path.exists():
        print(f"  UYARI: {regen_path} yok, atlandi")
        return pd.DataFrame()

    df = pd.read_parquet(regen_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    tgt = pd.Timestamp(target_date)
    day = df[df["Datetime"].dt.date == tgt.date()].copy()  # sadece T+2 (teslim günü)
    if day.empty:
        return pd.DataFrame()
    day["hour"] = day["Datetime"].dt.hour

    act = master[[C.RAW_DATE_COL, C.RAW_HOUR_COL, C.RAW_TARGET_COL]].copy()
    act["date"] = act[C.RAW_DATE_COL].dt.date
    act = act.rename(columns={C.RAW_TARGET_COL: "actual", C.RAW_HOUR_COL: "hour"})

    day = day.merge(act[["date", "hour", "actual"]], left_on=[day["Datetime"].dt.date, "hour"],
                     right_on=["date", "hour"], how="left")

    lag168_date = (tgt - timedelta(days=7)).date()
    lag = act[act["date"] == lag168_date][["hour", "actual"]].rename(columns={"actual": "lag168_actual"})
    day = day.merge(lag, on="hour", how="left")

    day["target_date"] = target_date
    return day[["target_date", "hour", "Final_Pred", "actual", "lag168_actual"]]


def run() -> None:
    master = pd.read_parquet(C.MASTER_PARQUET)
    master[C.RAW_DATE_COL] = pd.to_datetime(master[C.RAW_DATE_COL])

    all_days = [load_day(t, master) for t in SUNDAYS]
    all_days = [d for d in all_days if not d.empty]
    if not all_days:
        print("Hicbir Pazar icin REGEN verisi bulunamadi.")
        return
    combined = pd.concat(all_days, ignore_index=True)
    combined = combined.dropna(subset=["actual", "lag168_actual", "Final_Pred"])

    print(f"Toplam saat: {len(combined)}  ({combined['target_date'].nunique()} Pazar)")
    print()

    baseline_mape = mape(combined["Final_Pred"], combined["actual"])
    naive_mape = mape(combined["lag168_actual"], combined["actual"])
    print(f"Baseline (Final_Pred, mevcut model)  MAPE = {baseline_mape:.2f}%")
    print(f"Naive lag168 (salt gecen hafta)       MAPE = {naive_mape:.2f}%")
    print()

    results = []
    for alpha in ALPHAS:
        blended = alpha * combined["Final_Pred"] + (1 - alpha) * combined["lag168_actual"]
        m = mape(blended, combined["actual"])
        results.append({"alpha": alpha, "mape": m})
        print(f"  alpha={alpha:.1f}  (model_agirligi)  MAPE={m:.2f}%")

    res_df = pd.DataFrame(results)
    best = res_df.loc[res_df["mape"].idxmin()]
    print()
    print(f"En iyi alpha={best['alpha']:.1f}  MAPE={best['mape']:.2f}%  "
          f"(baseline'a gore {baseline_mape - best['mape']:+.2f}pp)")

    per_day = combined.groupby("target_date").apply(
        lambda g: pd.Series({
            "baseline_mape": mape(g["Final_Pred"], g["actual"]),
            "naive_mape": mape(g["lag168_actual"], g["actual"]),
            f"blend_alpha_{best['alpha']:.1f}_mape": mape(
                best["alpha"] * g["Final_Pred"] + (1 - best["alpha"]) * g["lag168_actual"], g["actual"]),
        }), include_groups=False,
    )
    print()
    print("Gun bazinda:")
    print(per_day.round(2).to_string())

    out_dir = C.OUTPUT_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "ensemble_ab_adm_pazar_lag168_2026-07-13.md"
    lines = [
        "# Walkforward A/B — ADM Pazar lag168-blend (Faz 2 2b-2, 2026-07-13)",
        "",
        f"4 tarihsel Pazar (as-of, günlük-yeniden-eğitilmiş modellerle asof_regen.regen_one()): "
        f"{', '.join(SUNDAYS)}",
        "",
        f"- Baseline (mevcut Final_Pred) MAPE: **{baseline_mape:.2f}%**",
        f"- Naive lag168 (salt geçen hafta) MAPE: **{naive_mape:.2f}%**",
        f"- En iyi blend (alpha={best['alpha']:.1f}) MAPE: **{best['mape']:.2f}%** "
        f"({baseline_mape - best['mape']:+.2f}pp baseline'a göre)",
        "",
        "## Alpha grid",
        "",
        res_df.round(2).to_markdown(index=False),
        "",
        "## Gün bazında",
        "",
        per_day.round(2).to_markdown(),
        "",
        "## Governance notu",
        "",
        "Bu rapor MASTER_PLAN.md §Faz 2c governance kuralı gereği üretildi: "
        "hiçbir ağırlık/model değişikliği bu tür bir walkforward A/B raporu olmadan canlıya girmez. "
        "Sonuç ne olursa olsun canlıya alma AYRI bir kullanıcı onayı gerektirir.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapor: {report_path}")


if __name__ == "__main__":
    run()
