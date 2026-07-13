"""experiments/gdz_ramp_bias_correction_ab.py — Faz 2 2b-3 (2026-07-13) walkforward A/B.

Hipotez: 2a-2 segment breakdown (`model_segment_mape_GDZ.csv`) evening+night
bloklarında sistematik under-forecast (ME negatif), morning bloğunda
sistematik over-forecast (ME pozitif) gösterdi — GDZ Final_Pred'e SAAT
BAZINDA sabit bir bias-correction eklemek MAPE'yi düşürür mü?

Yöntem (retrain YOK — sadece post-hoc additive correction, canlı log verisi
üzerinden): GDZ forecast_log_v/actuals_log_v'den T+2 saatlik veri okunur,
FIT penceresinde (ilk N gün) saat-bazlı ortalama hata (ME[hour] = mean(pred-
actual)) hesaplanır, TEST penceresinde (kalan gün, fit'e hiç karışmaz)
`corrected = final_pred - ME[hour]` uygulanıp MAPE karşılaştırılır. Bu
monitoring.duckdb'yi SADECE read-only okur, hiçbir canlı dosyaya yazmaz.

Governance (MASTER_PLAN.md): sonuç olumlu olsa bile canlıya otomatik girmez.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C
from monitoring.scorecard import load_hourly_report

FIT_END = "2026-07-06"    # dahil -- bias bu tarihe kadarki gunlerden ogrenilir
TEST_START = "2026-07-07"  # dahil -- duzeltme SADECE bu pencerede test edilir


def _load_gdz_tenant():
    sys.path.insert(0, str(C.GDZ_LIVE_ROOT))
    import config_live_gdz as CG  # noqa: PLC0415
    return CG


def mape(pred: pd.Series, actual: pd.Series) -> float:
    v = actual.notna() & pred.notna() & (actual != 0)
    if v.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[v] - pred[v]) / actual[v])) * 100)


def run() -> None:
    CG = _load_gdz_tenant()
    hourly = load_hourly_report(CG.TENANT, window_days=60, horizon="T+2")
    if hourly.empty:
        print("GDZ icin veri yok.")
        return

    fit = hourly[hourly["target_date"] <= FIT_END].copy()
    test = hourly[hourly["target_date"] >= TEST_START].copy()
    print(f"Fit penceresi: {fit['target_date'].min()}..{fit['target_date'].max()} "
          f"({fit['target_date'].nunique()} gun, {len(fit)} saat)")
    print(f"Test penceresi: {test['target_date'].min()}..{test['target_date'].max()} "
          f"({test['target_date'].nunique()} gun, {len(test)} saat)")
    if fit.empty or test.empty:
        print("Fit veya test penceresi bos, deney yapilamiyor.")
        return

    bias_by_hour = (fit["y_pred_final"] - fit["y_actual"]).groupby(fit["hour"]).mean()
    print("\nSaat bazli ogrenilen bias (fit penceresi, + = over-forecast):")
    print(bias_by_hour.round(1).to_string())

    test = test.copy()
    test["bias"] = test["hour"].map(bias_by_hour).fillna(0.0)
    test["corrected_pred"] = test["y_pred_final"] - test["bias"]

    baseline_mape = mape(test["y_pred_final"], test["y_actual"])
    corrected_mape = mape(test["corrected_pred"], test["y_actual"])
    print(f"\nTest penceresi baseline (Final_Pred) MAPE = {baseline_mape:.2f}%")
    print(f"Test penceresi duzeltilmis MAPE          = {corrected_mape:.2f}%")
    print(f"Fark: {baseline_mape - corrected_mape:+.2f}pp "
          f"({'IYILESME' if corrected_mape < baseline_mape else 'KOTULESME'})")

    # Saat-blogu bazinda kirilim (evening/night hipotezi ozelinde)
    from monitoring.scorecard import HOUR_BLOCKS
    test["hour_block"] = None
    for label, hrs in HOUR_BLOCKS.items():
        test.loc[test["hour"].isin(hrs), "hour_block"] = label
    per_block = test.groupby("hour_block").apply(
        lambda g: pd.Series({
            "baseline_mape": mape(g["y_pred_final"], g["y_actual"]),
            "corrected_mape": mape(g["corrected_pred"], g["y_actual"]),
        }), include_groups=False,
    )
    print("\nSaat-blogu bazinda:")
    print(per_block.round(2).to_string())

    out_dir = CG.OUTPUT_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "ensemble_ab_gdz_ramp_bias_2026-07-13.md"
    lines = [
        "# Walkforward A/B — GDZ saat-bazlı bias-correction (Faz 2 2b-3, 2026-07-13)",
        "",
        f"Fit penceresi: {fit['target_date'].min()}..{fit['target_date'].max()} "
        f"({fit['target_date'].nunique()} gün)  |  "
        f"Test penceresi: {test['target_date'].min()}..{test['target_date'].max()} "
        f"({test['target_date'].nunique()} gün)",
        "",
        f"- Baseline (mevcut Final_Pred) MAPE: **{baseline_mape:.2f}%**",
        f"- Saat-bazlı bias-correction sonrası MAPE: **{corrected_mape:.2f}%** "
        f"({baseline_mape - corrected_mape:+.2f}pp)",
        "",
        "## Fit penceresinden öğrenilen saatlik bias (+ = over-forecast)",
        "",
        bias_by_hour.round(1).to_frame("bias_mwh").to_markdown(),
        "",
        "## Saat-bloğu bazında (test penceresi)",
        "",
        per_block.round(2).to_markdown(),
        "",
        "## Governance notu",
        "",
        "Bu rapor MASTER_PLAN.md §Faz 2c governance kuralı gereği üretildi: "
        "hiçbir ağırlık/model değişikliği bu tür bir walkforward A/B raporu olmadan canlıya girmez.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapor: {report_path}")


if __name__ == "__main__":
    run()
