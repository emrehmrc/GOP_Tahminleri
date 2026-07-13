"""export_hourly_mape_7d.py — Son 7 günün saat-saat MAPE dökümü (4 model + ens_raw + final + düzeltme).
forecast_log_v ⋈ actuals_log_v; sadece okuma. Çıktı: output/hourly_mape_7d_<bugun>.xlsx + CSV'ler.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import duckdb

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C
from src.output_paths import date_folder
from src import forecast_logger as FL

FL.rebuild_duckdb_views()

con = duckdb.connect(str(C.MONITORING_DB), read_only=True)
df = con.execute("""
    SELECT f.target_date, f.horizon_day, f.target_ts,
           f.y_pred_xgb, f.y_pred_lgbm, f.y_pred_cat, f.y_pred_chronos,
           f.y_pred_ens_raw, f.y_pred_final,
           f.override_delta, f.subst_delta, f.pv_bias_delta,
           f.meta_w_xgb, f.meta_w_lgbm, f.meta_w_cat, f.meta_w_chronos, f.meta_method,
           a.y_actual
    FROM forecast_log_v f
    INNER JOIN actuals_log_v a
      ON f.edas_id = a.edas_id AND f.target_ts = a.target_ts
    WHERE a.y_actual IS NOT NULL
    ORDER BY f.target_ts
""").df()
con.close()

df["target_ts"] = pd.to_datetime(df["target_ts"])
df["hour"] = df["target_ts"].dt.hour
df["target_date"] = pd.to_datetime(df["target_date"]).dt.strftime("%Y-%m-%d")
last7 = sorted(df["target_date"].unique())[-7:]
df = df[df["target_date"].isin(last7)].copy()

MODELS = ["xgb", "lgbm", "cat", "chronos", "ens_raw", "final"]
COL = {"xgb": "y_pred_xgb", "lgbm": "y_pred_lgbm", "cat": "y_pred_cat",
       "chronos": "y_pred_chronos", "ens_raw": "y_pred_ens_raw", "final": "y_pred_final"}
for m in MODELS:
    df[f"ape_{m}"] = np.abs((df["y_actual"] - df[COL[m]]) / df["y_actual"]) * 100

# ── Sheet 1: gün x horizon x model MAPE özeti ────────────────────────────────
summ = (df.groupby(["target_date", "horizon_day"])
          .apply(lambda g: pd.Series({m: g[f"ape_{m}"].mean() for m in MODELS} |
                                     {"n_saat": len(g), "meta": g["meta_method"].dropna().iloc[0]
                                      if g["meta_method"].notna().any() else None}))
          .reset_index())

# ── Sheet 2: saat x model APE matrisi (her gün ayrı bloklu, uzun form) ────────
long_rows = []
for (d, hz), g in df.groupby(["target_date", "horizon_day"]):
    g = g.sort_values("hour")
    for _, r in g.iterrows():
        long_rows.append({
            "gun": d, "horizon": hz, "saat": int(r["hour"]), "y_actual": r["y_actual"],
            **{m: r[COL[m]] for m in MODELS},
            **{f"APE_{m}": r[f"ape_{m}"] for m in MODELS},
            "pv_bias": r["pv_bias_delta"], "override": r["override_delta"], "subst": r["subst_delta"],
            "duzeltme_kazanci_bps": (r["ape_ens_raw"] - r["ape_final"]) * 100
            if pd.notna(r["ape_ens_raw"]) else np.nan,
        })
long_df = pd.DataFrame(long_rows)

# ── Sheet 3: her gün için saat(satır) x model(kolon) APE pivotu ───────────────
pivots = {}
for (d, hz), g in df.groupby(["target_date", "horizon_day"]):
    piv = g.set_index("hour")[[f"ape_{m}" for m in MODELS]].round(2)
    piv.columns = [c.replace("ape_", "") for c in piv.columns]
    pivots[f"{d}_{hz}"] = piv

# ── yaz ──────────────────────────────────────────────────────────────────────
stamp = date.today().isoformat()
dated_dir = date_folder(C.OUTPUT_DIR, stamp, create=True)
xlsx = dated_dir / f"hourly_mape_7d_{stamp}.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
    summ.round(2).to_excel(xw, sheet_name="OZET_gun_model", index=False)
    long_df.round(2).to_excel(xw, sheet_name="saat_saat_uzun", index=False)
    for name, piv in pivots.items():
        piv.to_excel(xw, sheet_name=("APE_" + name.replace("+", ""))[:31])

# CSV'ler (Excel açamayan için)
summ.round(2).to_csv(dated_dir / f"hourly_mape_7d_OZET_{stamp}.csv", index=False)
long_df.round(2).to_csv(dated_dir / f"hourly_mape_7d_saat_saat_{stamp}.csv", index=False)

# ── ekrana bas ───────────────────────────────────────────────────────────────
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 60); pd.set_option("display.max_columns", 40)
print(f"YAZILDI:\n  {xlsx}\n  hourly_mape_7d_OZET_{stamp}.csv\n  hourly_mape_7d_saat_saat_{stamp}.csv\n")
print("Gunler (actual dolu):", last7)
print("\n=== OZET: gun x horizon x model MAPE (%) ===")
print(summ.round(2).to_string(index=False))
for name, piv in pivots.items():
    print(f"\n=== {name}: saat x model APE (%) ===")
    print(piv.to_string())
