"""analyze_models_30d.py — Her model: hangi gun, hangi saat, hangi gun tipinde sicti?
=========================================================================================
Faz 2 2a-2 (2026-07-13): eskiden *_models_REGEN.parquet dump'larini okuyordu --
sadece o dump'larin uretildigi gunler icin calisabiliyordu. Artik ADM + GDZ
forecast_log_v / actuals_log_v'den (monitoring.duckdb) besleniyor, yani canli
veriyle HER GUN kosulabilir.

Cikti (her tenant icin OUTPUT_DIR/analysis/ altina):
  1. Konsola detayli rapor (ADM + GDZ ayri ayri)
  2. model_analysis_daily_<edas>.csv     -- gunluk MAPE tablosu (per model)
  3. model_segment_mape_<edas>.csv       -- Faz 2 2a-2: model x saat-blogu x gun-tipi MAPE/ME
  4. model_worst_hours_<edas>.csv        -- her modelin en kotu 15 saati
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C
from monitoring.scorecard import load_hourly_report, model_segment_breakdown, MODEL_PRED_COLS

WINDOW_DAYS = 30
MODEL_LABELS = {
    "xgb": "XGB", "lgbm": "LGBM", "cat": "CAT", "chronos": "CHRONOS",
    "ens_raw": "Ensemble", "final": "Final",
}


def _load_gdz_tenant():
    sys.path.insert(0, str(C.GDZ_LIVE_ROOT))
    import config_live_gdz as CG  # noqa: PLC0415
    return CG


def mape_of(pred: pd.Series, actual: pd.Series) -> float:
    valid = pred.notna() & actual.notna() & (actual != 0)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[valid] - pred[valid]) / actual[valid])) * 100)


def analyze_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Gunluk MAPE per model."""
    rows = []
    for target_date, g in hourly.groupby("target_date"):
        row = {"target_date": target_date, "day_type": g["day_type"].iloc[0],
               "flag_holiday": bool(g["flag_holiday"].iloc[0])}
        for model, pred_col in MODEL_PRED_COLS.items():
            if pred_col in g.columns:
                row[f"{MODEL_LABELS[model]}_MAPE"] = mape_of(g[pred_col], g["y_actual"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("target_date")


def analyze_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Saat bazli MAPE per model."""
    rows = []
    for h in range(24):
        hour_df = hourly[hourly["hour"] == h]
        row = {"hour": h, "n_samples": len(hour_df)}
        for model, pred_col in MODEL_PRED_COLS.items():
            if pred_col in hour_df.columns:
                row[f"{MODEL_LABELS[model]}_MAPE"] = mape_of(hour_df[pred_col], hour_df["y_actual"])
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_worst_hours(hourly: pd.DataFrame, n: int = 15) -> dict[str, pd.DataFrame]:
    """Her modelin en kotu N saati (tarih+saat bazinda)."""
    worst = {}
    for model, pred_col in MODEL_PRED_COLS.items():
        if pred_col not in hourly.columns:
            continue
        g = hourly[[pred_col, "y_actual", "target_ts", "day_type", "wx_temp_fcst", "wx_ghi_fcst"]].copy()
        g["ape"] = np.abs((g["y_actual"] - g[pred_col]) / (g["y_actual"] + 1e-10)) * 100
        g["error"] = g[pred_col] - g["y_actual"]
        top = g.nlargest(n, "ape")
        worst[MODEL_LABELS[model]] = top
    return worst


def print_tenant_report(edas_id: str, hourly: pd.DataFrame, segment: pd.DataFrame,
                         out_dir: Path) -> None:
    print("=" * 80)
    print(f"  {edas_id} CANLI {WINDOW_DAYS}-GUN MODEL HATA ANALIZI")
    if hourly.empty:
        print("  (veri yok -- forecast_log/actuals_log henuz bos veya monitoring.duckdb yok)")
        print("=" * 80)
        return
    print(f"  Tarih araligi: {hourly['target_date'].min()} .. {hourly['target_date'].max()}")
    print(f"  Gun sayisi: {hourly['target_date'].nunique()}  Toplam saat: {len(hourly)}")
    print("=" * 80)

    available_models = [m for m, c in MODEL_PRED_COLS.items() if c in hourly.columns]

    print("\n" + "-" * 70)
    print("  1. GENEL PERFORMANS OZETI")
    print("-" * 70)
    for model in available_models:
        pred_col = MODEL_PRED_COLS[model]
        m = mape_of(hourly[pred_col], hourly["y_actual"])
        me = float((hourly[pred_col] - hourly["y_actual"]).mean())
        print(f"  {MODEL_LABELS[model]:<12s}  MAPE={m:.2f}%  ME={me:.1f} MWh")

    daily = analyze_daily(hourly)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / f"model_analysis_daily_{edas_id}.csv", index=False)
    print("\n" + "-" * 70)
    print("  2. GUNLUK MAPE (EN KOTU 10 GUN, Final_MAPE)")
    print("-" * 70)
    final_col = f"{MODEL_LABELS['final']}_MAPE"
    if final_col in daily.columns:
        worst_days = daily.nlargest(10, final_col)
        for _, r in worst_days.iterrows():
            tag = " [TATIL]" if r.get("flag_holiday") else ""
            print(f"  {r['target_date']} {r.get('day_type',''):<20s}{tag}  Final={r[final_col]:.2f}%")

    hourly_mape = analyze_hourly(hourly)
    print("\n" + "-" * 70)
    print("  3. SAAT BAZLI MAPE (her modelin en kotu 3 saati)")
    print("-" * 70)
    for model in available_models:
        col = f"{MODEL_LABELS[model]}_MAPE"
        if col not in hourly_mape.columns:
            continue
        worst = hourly_mape.nlargest(3, col)
        parts = [f"{int(r['hour']):02d}h={r[col]:.1f}%" for _, r in worst.iterrows()]
        print(f"  {MODEL_LABELS[model]:<12s}  {' | '.join(parts)}")

    print("\n" + "-" * 70)
    print("  4. MODEL x SAAT-BLOGU x GUN-TIPI KIRILIMI (Faz 2 2a-2)")
    print("-" * 70)
    if segment.empty:
        print("  (segment kirinimi bos)")
    else:
        segment.to_csv(out_dir / f"model_segment_mape_{edas_id}.csv", index=False)
        pivot = segment.pivot_table(
            index=["hour_block", "day_type_group"], columns="model", values="mape"
        )
        print(pivot.round(2).to_string())

    worst_hours = analyze_worst_hours(hourly)
    print("\n" + "-" * 70)
    print("  5. HER MODELIN EN KOTU 5 SAATI (tarih + saat)")
    print("-" * 70)
    combined_worst = []
    for model_label, wdf in worst_hours.items():
        wdf = wdf.copy()
        wdf["model"] = model_label
        combined_worst.append(wdf)
        print(f"\n  > {model_label}")
        for _, r in wdf.head(5).iterrows():
            print(f"    {r['target_ts']}  APE={r['ape']:.1f}%  actual={r['y_actual']:.0f}  "
                  f"error={r['error']:.0f} MWh  {r['day_type']}")
    if combined_worst:
        pd.concat(combined_worst, ignore_index=True).to_csv(
            out_dir / f"model_worst_hours_{edas_id}.csv", index=False)

    print("\n" + "=" * 80)
    print(f"  Rapor ciktilari: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    adm_out = C.OUTPUT_DIR / "analysis"
    adm_hourly = load_hourly_report(C.TENANT, window_days=WINDOW_DAYS)
    adm_segment = model_segment_breakdown(C.TENANT, window_days=WINDOW_DAYS)
    print_tenant_report(C.EDAS_ID, adm_hourly, adm_segment, adm_out)

    print()
    try:
        CG = _load_gdz_tenant()
    except ImportError as e:
        print(f"GDZ config yuklenemedi ({e}) -- GDZ analizi atlandi.")
    else:
        gdz_out = CG.OUTPUT_DIR / "analysis"
        gdz_hourly = load_hourly_report(CG.TENANT, window_days=WINDOW_DAYS)
        gdz_segment = model_segment_breakdown(CG.TENANT, window_days=WINDOW_DAYS)
        print_tenant_report(CG.EDAS_ID, gdz_hourly, gdz_segment, gdz_out)
