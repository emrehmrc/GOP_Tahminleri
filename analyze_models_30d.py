"""analyze_models_30d.py — Her model: hangi gun, hangi saat, nerede, NEDEN sicti?
=====================================================================================
30 gunluk models_REGEN.parquet'leri ve master.parquet'yi okuyup per-model
derinlemesine hata analizi yapar.

Cikti:
  1. Konsola detayli rapor
  2. output/model_analysis_report.csv  — gunluk MAPE tablosu
  3. output/model_hourly_mape.csv      — saat bazli MAPE
  4. output/model_worst_hours.csv      — her modelin en kotu 10 saati
  5. output/model_root_cause.txt       — kok neden tezleri
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from datetime import date
from collections import defaultdict
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C

OUT = C.OUTPUT_DIR

TARGET_DATES = sorted(
    f.stem.replace("_models_REGEN", "")
    for f in OUT.glob("*_models_REGEN.parquet")
)
TARGET_DATES = [d for d in TARGET_DATES if "2026-06" <= d[:7] <= "2026-07"]
TARGET_DATES.sort()

MODEL_COLS = ["XGB_Pred", "LGBM_Pred", "CAT_Pred", "CHRONOS_Pred",
              "Ensemble_Pred", "Final_Pred"]

# ── data load ─────────────────────────────────────────────────────────────────
def load_all():
    master = pd.read_parquet(C.MASTER_PARQUET)
    master[C.RAW_DATE_COL] = pd.to_datetime(master[C.RAW_DATE_COL])

    all_preds = []
    for td in TARGET_DATES:
        f = OUT / f"{td}_models_REGEN.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df["target_date"] = td
        tgt = pd.Timestamp(td)
        # Sadece T+2 satirlari (teslim gunu)
        df_t2 = df[df["Datetime"].dt.date == tgt.date()].copy()
        df_t2["hour"] = df_t2["Datetime"].dt.hour
        all_preds.append(df_t2)

    preds = pd.concat(all_preds, ignore_index=True)
    preds["date"] = preds["Datetime"].dt.date

    # Merge actuals
    act = master[[C.RAW_DATE_COL, C.RAW_HOUR_COL, C.RAW_TARGET_COL]].copy()
    act["date"] = act[C.RAW_DATE_COL].dt.date
    act["hour"] = act[C.RAW_HOUR_COL]
    act = act.rename(columns={C.RAW_TARGET_COL: "Actual_MWh"})

    merged = preds.merge(
        act[["date", "hour", "Actual_MWh"]],
        on=["date", "hour"], how="inner"
    )

    # APE columns
    for col in MODEL_COLS:
        if col in merged.columns:
            merged[f"APE_{col}"] = np.abs(merged[col] - merged["Actual_MWh"]) / merged["Actual_MWh"] * 100
            merged[f"Error_{col}"] = merged[col] - merged["Actual_MWh"]  # positive = over-predict

    return merged, master, preds


# ── metrics ───────────────────────────────────────────────────────────────────
def mape_of(series_pred, series_act):
    v = series_act.notna() & series_pred.notna() & (series_act > 0)
    if v.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((series_act[v] - series_pred[v]) / series_act[v])) * 100)


# ── ANALYSIS FUNCTIONS ────────────────────────────────────────────────────────

def analyze_daily(merged, cols):
    """Gunluk MAPE per model."""
    rows = []
    for d in sorted(merged["date"].unique()):
        day_df = merged[merged["date"] == d]
        row = {"date": str(d)}
        row["day_type"] = day_df["day_type"].iloc[0] if "day_type" in day_df.columns else "?"
        row["flag_holiday"] = day_df["flag_holiday"].iloc[0] if "flag_holiday" in day_df.columns else False
        row["flag_ramadan"] = day_df["flag_ramadan"].iloc[0] if "flag_ramadan" in day_df.columns else False
        for col in cols:
            if col in day_df.columns:
                row[f"{col}_MAPE"] = mape_of(day_df[col], day_df["Actual_MWh"])
                row[f"{col}_ME"] = round(float(day_df[f"Error_{col}"].mean()), 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def analyze_hourly(merged, cols):
    """Saat bazli MAPE per model."""
    rows = []
    for h in range(24):
        hour_df = merged[merged["hour"] == h]
        row = {"hour": h}
        row["solar"] = h in C.PV_BIAS_SOLAR_HOURS
        row["n_samples"] = len(hour_df)
        for col in cols:
            if col in hour_df.columns:
                row[f"{col}_MAPE"] = mape_of(hour_df[col], hour_df["Actual_MWh"])
                row[f"{col}_ME_mean"] = round(float(hour_df[f"Error_{col}"].mean()), 1)
                row[f"{col}_ME_std"] = round(float(hour_df[f"Error_{col}"].std()), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_by_daytype(merged, cols):
    """Gun tipine gore MAPE."""
    rows = []
    for dt in sorted(merged["day_type"].dropna().unique()):
        df_dt = merged[merged["day_type"] == dt]
        row = {"day_type": dt, "n_hours": len(df_dt)}
        for col in cols:
            if col in df_dt.columns:
                row[f"{col}_MAPE"] = mape_of(df_dt[col], df_dt["Actual_MWh"])
                row[f"{col}_ME"] = round(float(df_dt[f"Error_{col}"].mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_by_solar(merged, cols):
    """Gunesli vs gunessiz saatler."""
    rows = []
    for label, mask in [("gunesli", merged["wx_ghi_fcst"] > 50),
                         ("gunesli_zayif", (merged["wx_ghi_fcst"] > 0) & (merged["wx_ghi_fcst"] <= 50)),
                         ("gunessiz", merged["wx_ghi_fcst"] == 0)]:
        df_seg = merged[mask]
        row = {"ghi_segment": label, "n_hours": len(df_seg)}
        for col in cols:
            if col in df_seg.columns:
                row[f"{col}_MAPE"] = mape_of(df_seg[col], df_seg["Actual_MWh"])
                row[f"{col}_ME"] = round(float(df_seg[f"Error_{col}"].mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_worst_combined(merged, cols):
    """Her modelin en kotu N saati (tarih+saat bazinda)."""
    worst = {}
    for col in cols:
        ap_col = f"APE_{col}"
        if ap_col not in merged.columns:
            continue
        top = merged.nlargest(15, ap_col)[
            ["date", "hour", col, "Actual_MWh", ap_col,
             "wx_ghi_fcst", "wx_temp_fcst", "day_type",
             f"Error_{col}"]
        ].copy()
        top = top.rename(columns={ap_col: "APE", f"Error_{col}": "Error_MWh"})
        worst[col] = top
    return worst


def analyze_correlation(merged, cols):
    """Model hatalari arasinda korelasyon — birlikte mi siciyorlar?"""
    error_cols = [f"Error_{c}" for c in cols if f"Error_{c}" in merged.columns]
    if len(error_cols) < 2:
        return None
    return merged[error_cols].corr()


def detect_patterns(merged, cols):
    """Otomatik oruntu tespiti — her model icin hatalarin neyle iliskili oldugunu bul."""
    patterns = {}
    for col in cols:
        ap_col = f"APE_{col}"
        if ap_col not in merged.columns:
            continue
        ape = merged[ap_col]
        findings = []

        # 1. Gun tipi etkisi
        if "day_type" in merged.columns:
            for dt in merged["day_type"].dropna().unique():
                seg_ape = merged.loc[merged["day_type"] == dt, ap_col]
                overall_ape = ape.mean()
                seg_mean = seg_ape.mean()
                if len(seg_ape) >= 10 and seg_mean > overall_ape * 1.4:
                    findings.append(f"  ⚠  '{dt}' gunlerinde MAPE={seg_mean:.1f}% (genel ort {overall_ape:.1f}%'in cok ustunde)")

        # 2. Saat etkisi
        worst_hours = merged.groupby("hour")[ap_col].mean().nlargest(3)
        for h, v in worst_hours.items():
            if v > ape.mean() * 1.5:
                findings.append(f"  ⚠  Saat {h:02d}:00'da MAPE={v:.1f}% (genel ort {ape.mean():.1f}%'in cok ustunde)")

        # 3. Gunes etkisi
        if "wx_ghi_fcst" in merged.columns:
            hi_ghi = merged[merged["wx_ghi_fcst"] > 400]["APE_XGB_Pred" if col == "XGB_Pred" else ap_col]
            lo_ghi = merged[merged["wx_ghi_fcst"] == 0][ap_col]
            if len(hi_ghi) > 20 and hi_ghi.mean() > ape.mean() * 1.3:
                findings.append(f"  ⚠  Yuksek GHI (>400) saatlerinde MAPE={hi_ghi.mean():.1f}% (genel ort {ape.mean():.1f}%)")
            if len(lo_ghi) > 20 and lo_ghi.mean() > ape.mean() * 1.3:
                findings.append(f"  ⚠  GHI=0 (gece) saatlerinde MAPE={lo_ghi.mean():.1f}% (genel ort {ape.mean():.1f}%)")

        # 4. Sistematik bias
        err = merged[f"Error_{col}"]
        if abs(err.mean()) > 20:
            direction = "fazla (over)" if err.mean() > 0 else "az (under)"
            findings.append(f"  ⚠  Sistematik {direction}-tahmin: ME={err.mean():.0f} MWh")

        patterns[col] = findings
    return patterns


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("  ADM CANLI 30-GUN MODEL HATA ANALIZI")
    print(f"  Tarih araligi: {TARGET_DATES[0] if TARGET_DATES else '?'} .. {TARGET_DATES[-1] if TARGET_DATES else '?'}")
    print(f"  Gun sayisi: {len(TARGET_DATES)}")
    print("=" * 80)

    merged, master, preds = load_all()
    available_cols = [c for c in MODEL_COLS if c in merged.columns]
    print(f"\nAnaliz edilen modeller: {available_cols}")
    print(f"Toplam saat: {len(merged)}")

    # ── 1. GENEL OZET ──────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  1. GENEL 30-GUN PERFORMANS OZETI")
    print("─" * 70)
    for col in available_cols:
        m = mape_of(merged[col], merged["Actual_MWh"])
        me = merged[f"Error_{col}"].mean() if f"Error_{col}" in merged.columns else float("nan")
        print(f"  {col:<20s}  MAPE={m:.2f}%  ME={me:.1f} MWh")

    # ── 2. GUNLUK DAGILIM ──────────────────────────────────────────────────
    daily = analyze_daily(merged, available_cols)
    daily.to_csv(OUT / "model_analysis_report.csv", index=False)
    print("\n" + "─" * 70)
    print("  2. GUNLUK MAPE (EN KOTU 10 GUN)")
    print("─" * 70)
    ens_col = "Ensemble_Pred_MAPE"
    if ens_col in daily.columns:
        worst_days = daily.nlargest(10, ens_col)
        for _, r in worst_days.iterrows():
            flags = []
            if r.get("flag_holiday"): flags.append("TATIL")
            if r.get("flag_ramadan"): flags.append("RAMAZAN")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"  {r['date']} {r.get('day_type',''):<15s}{flag_str}  Ensemble={r[ens_col]:.2f}%  "
                  f"XGB={r.get('XGB_Pred_MAPE',0):.2f}%  LGBM={r.get('LGBM_Pred_MAPE',0):.2f}%  "
                  f"CHRONOS={r.get('CHRONOS_Pred_MAPE',0):.2f}%  Final={r.get('Final_Pred_MAPE',0):.2f}%")

    # ── 3. SAAT BAZLI ──────────────────────────────────────────────────────
    hourly = analyze_hourly(merged, available_cols)
    hourly.to_csv(OUT / "model_hourly_mape.csv", index=False)
    print("\n" + "─" * 70)
    print("  3. SAAT BAZLI MAPE (her modelin en kotu 3 saati)")
    print("─" * 70)
    for col in available_cols:
        m_col = f"{col}_MAPE"
        if m_col not in hourly.columns:
            continue
        worst = hourly.nlargest(3, m_col)
        parts = []
        for _, r in worst.iterrows():
            parts.append(f"{int(r['hour']):02d}h={r[m_col]:.1f}%")
        print(f"  {col:<20s}  {' | '.join(parts)}")

    # ── 4. GUN TIPINE GORE ─────────────────────────────────────────────────
    daytype = analyze_by_daytype(merged, available_cols)
    print("\n" + "─" * 70)
    print("  4. GUN TIPINE GORE PERFORMANS")
    print("─" * 70)
    print(daytype.to_string(index=False))

    # ── 5. SOLAR vs NON-SOLAR ──────────────────────────────────────────────
    solar = analyze_by_solar(merged, available_cols)
    print("\n" + "─" * 70)
    print("  5. GUNESLI / GUNESSIZ SAAT KIRILIMI")
    print("─" * 70)
    print(solar.to_string(index=False))

    # ── 6. KORELASYON ──────────────────────────────────────────────────────
    corr = analyze_correlation(merged, available_cols)
    if corr is not None:
        print("\n" + "─" * 70)
        print("  6. MODEL HATA KORELASYONU (birlikte mi siciyorlar?)")
        print("─" * 70)
        print(corr.round(2).to_string())

    # ── 7. HER MODELIN EN KOTU ANLARI ──────────────────────────────────────
    worst = analyze_worst_combined(merged, available_cols)
    print("\n" + "─" * 70)
    print("  7. HER MODELIN EN KOTU 10 SAATI (tarih + saat)")
    print("─" * 70)
    for col in available_cols:
        if col not in worst:
            continue
        wdf = worst[col].head(10)
        print(f"\n  ▸ {col}")
        for _, r in wdf.iterrows():
            print(f"    {r['date']} {int(r['hour']):02d}:00  APE={r['APE']:.1f}%  "
                  f"pred={r[col]:.0f}  actual={r['Actual_MWh']:.0f}  "
                  f"error={r['Error_MWh']:.0f} MWh  GHI={r['wx_ghi_fcst']:.0f}  "
                  f"temp={r['wx_temp_fcst']:.1f}C  {r['day_type']}")

    # ── 8. KOK NEDEN ANALIZI ───────────────────────────────────────────────
    patterns = detect_patterns(merged, available_cols)
    print("\n" + "─" * 70)
    print("  8. OTOMATIK KOK NEDEN TEZ'LERI")
    print("─" * 70)
    for col, findings in patterns.items():
        print(f"\n  ▸ {col}")
        if findings:
            for f in findings:
                print(f)
        else:
            print("    ✓ Belirgin bir sorun oruntusu bulunamadi")

    # ── 9. MODEL SIRALAMASI ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  9. MODEL BASARI SIRALAMASI (30-gun T+2 MAPE)")
    print("─" * 70)
    model_scores = [(c, mape_of(merged[c], merged["Actual_MWh"])) for c in available_cols]
    model_scores.sort(key=lambda x: x[1])
    for rank, (name, score) in enumerate(model_scores, 1):
        bar = "█" * int(score * 2)
        print(f"  {rank}. {name:<20s} {score:.2f}% {bar}")

    # ── 10. BIAS/VARYANS AYRISTIRMASI ────────────────────────────────────
    print("\n" + "─" * 70)
    print("  10. BIAS / SISTEMATIK HATA ANALIZI")
    print("─" * 70)
    for col in available_cols:
        err = merged[f"Error_{col}"]
        actual_mean = merged["Actual_MWh"].mean()
        bias_pct = err.mean() / actual_mean * 100
        std_pct = err.std() / actual_mean * 100
        direction = "OVER (fazla tahmin)" if err.mean() > 0 else "UNDER (az tahmin)"
        print(f"  {col:<20s}  ME={err.mean():+.1f} MWh ({bias_pct:+.1f}%) → {direction}  "
              f"Std={err.std():.0f} MWh ({std_pct:.1f}%)")

    # ── 11. ENAYI TAHMIN ───────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  11. EN IYI TAHMIN EDILEN SAATLER (TUM MODELLER ICIN)")
    print("─" * 70)
    best_hours = hourly.nsmallest(5, "Ensemble_Pred_MAPE") if "Ensemble_Pred_MAPE" in hourly.columns else hourly.head()
    for _, r in best_hours.iterrows():
        solar_mark = "☀" if r.get("solar") else "☽"
        print(f"  {int(r['hour']):02d}h {solar_mark}  Ensemble={r.get('Ensemble_Pred_MAPE',0):.2f}%  "
              f"XGB={r.get('XGB_Pred_MAPE',0):.2f}%  LGBM={r.get('LGBM_Pred_MAPE',0):.2f}%  "
              f"CHRONOS={r.get('CHRONOS_Pred_MAPE',0):.2f}%")

    print("\n" + "=" * 80)
    print("  Rapor ciktilari:")
    print(f"    {OUT / 'model_analysis_report.csv'}")
    print(f"    {OUT / 'model_hourly_mape.csv'}")
    print("=" * 80)
