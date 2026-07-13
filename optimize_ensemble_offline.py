"""
optimize_ensemble_offline.py — ADM + GDZ ensemble/bias offline kalibrasyon
==========================================================================
Yeni model egitimi YOK. Mevcut tahmin artefaktlari + master actuals uzerinde
agirlik / bias grid arar. Iki EDAŞ tamamen AYRI optimize edilir.

ADM kaynak: output/*_models_REGEN.parquet (T+2 teslim gunu)
GDZ kaynak: gdz talep/live/output/archive/*_full48h.parquet

Cikti:
  output/ensemble_opt_report_ADM.json
  output/ensemble_opt_report_GDZ.json
  output/ensemble_opt_daily_ADM.csv
  output/ensemble_opt_daily_GDZ.csv

Kullanim:
  python optimize_ensemble_offline.py
  python optimize_ensemble_offline.py --edas ADM
  python optimize_ensemble_offline.py --edas GDZ --holdout-days 5
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config_live as C
from src.output_paths import glob_output_files
from src.metrics import calculate_mape

MODEL_COLS_ADM = ["XGB_Pred", "LGBM_Pred", "CAT_Pred", "CHRONOS_Pred"]
MODEL_KEYS = ["XGB", "LGBM", "CAT", "CHRONOS"]

# Current live defaults (for baseline comparison)
ADM_LIVE_WEIGHTS = dict(C.CALIBRATED_ENSEMBLE_WEIGHTS)
ADM_BIAS = {
    "t1": float(C.ENSEMBLE_BIAS_CORRECTION_T1_MWH),
    "t2": float(C.ENSEMBLE_BIAS_CORRECTION_T2_MWH),
    "we_t1": float(C.ENSEMBLE_BIAS_WEEKEND_SCALE_T1),
    "we_t2": float(C.ENSEMBLE_BIAS_WEEKEND_SCALE_T2),
    "su_t1": float(C.ENSEMBLE_BIAS_SUNDAY_SCALE_T1),
    "su_t2": float(C.ENSEMBLE_BIAS_SUNDAY_SCALE_T2),
}

GDZ_LIVE_DIR = ROOT.parent / "gdz talep" / "live"
GDZ_MASTER = ROOT.parent / "gdz talep" / "Input" / "GDZ_MASTER.parquet"
GDZ_RAW_TARGET = "GDZ- Dağıtılan Enerji (MWh)"


def _mape(pred: np.ndarray, act: np.ndarray) -> float:
    return float(calculate_mape(act, pred))


def _me(pred: np.ndarray, act: np.ndarray) -> float:
    return float(np.mean(pred - act))


def _day_type_from_dt(ts: pd.Timestamp) -> str:
    d = ts.dayofweek
    if d == 5:
        return "cumartesi"
    if d == 6:
        return "pazar"
    return "hafta_ici"


def apply_bias(
    ens: np.ndarray,
    day_type: np.ndarray,
    is_t2: np.ndarray,
    bias: dict,
) -> np.ndarray:
    """05_postprocess day-aware bias — birebir formül (holiday/PV haric)."""
    base = np.where(is_t2, bias["t2"], bias["t1"]).astype(float)
    is_sunday = day_type == "pazar"
    is_weekend = (day_type == "cumartesi") | is_sunday
    scale_t1 = np.where(is_sunday, bias["su_t1"], np.where(is_weekend, bias["we_t1"], 1.0))
    scale_t2 = np.where(is_sunday, bias["su_t2"], np.where(is_weekend, bias["we_t2"], 1.0))
    scale = np.where(is_t2, scale_t2, scale_t1)
    return ens + base * scale


def weighted_ensemble(df: pd.DataFrame, weights: dict, cols: list[str]) -> np.ndarray:
    """weights keys like XGB_Pred or XGB — normalize to cols present."""
    w = {}
    for c in cols:
        key = c
        short = c.replace("_Pred", "")
        if key in weights:
            w[c] = float(weights[key])
        elif short in weights:
            w[c] = float(weights[short])
        else:
            w[c] = 0.0
    s = sum(w.values())
    if s <= 0:
        # equal
        arr = df[cols].to_numpy(dtype=float)
        return np.nanmean(arr, axis=1)
    return sum(df[c].to_numpy(dtype=float) * (w[c] / s) for c in cols)


def weight_grid(step: float = 0.05):
    """Non-negative weights sum to 1 for 4 models, step grid."""
    levels = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    for a, b, c in itertools.product(levels, repeat=3):
        d = round(1.0 - a - b - c, 4)
        if d < -1e-9 or d > 1.0 + 1e-9:
            continue
        if d < 0:
            continue
        yield {
            "XGB_Pred": float(a),
            "LGBM_Pred": float(b),
            "CAT_Pred": float(c),
            "CHRONOS_Pred": float(d),
        }


# ── ADM load ──────────────────────────────────────────────────────────────────

def load_adm() -> pd.DataFrame:
    master = pd.read_parquet(C.MASTER_PARQUET)
    master[C.RAW_DATE_COL] = pd.to_datetime(master[C.RAW_DATE_COL])
    act = master[[C.RAW_DATE_COL, C.RAW_HOUR_COL, C.RAW_TARGET_COL]].copy()
    act["date"] = act[C.RAW_DATE_COL].dt.date
    act["hour"] = act[C.RAW_HOUR_COL].astype(int)
    act = act.rename(columns={C.RAW_TARGET_COL: "Actual_MWh"})

    rows = []
    for f in glob_output_files(C.OUTPUT_DIR, "*_models_REGEN.parquet"):
        td = f.stem.replace("_models_REGEN", "")
        df = pd.read_parquet(f)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        tgt = pd.Timestamp(td).date()
        # T+2 delivery day only
        t2 = df[df["Datetime"].dt.date == tgt].copy()
        if t2.empty:
            continue
        t2["target_date"] = td
        t2["hour"] = t2["Datetime"].dt.hour.astype(int)
        t2["date"] = t2["Datetime"].dt.date
        rows.append(t2)

    if not rows:
        raise RuntimeError("ADM: no REGEN parquets found")

    preds = pd.concat(rows, ignore_index=True)
    merged = preds.merge(act[["date", "hour", "Actual_MWh"]], on=["date", "hour"], how="inner")
    if "day_type" not in merged.columns or merged["day_type"].isna().all():
        merged["day_type"] = merged["Datetime"].map(_day_type_from_dt)
    else:
        merged["day_type"] = merged["day_type"].fillna(merged["Datetime"].map(_day_type_from_dt))
    # residual postprocess beyond bias (holiday + pv) if available
    for col in ("subst_delta", "pv_bias_delta"):
        if col not in merged.columns:
            merged[col] = 0.0
        else:
            merged[col] = merged[col].fillna(0.0)
    merged["is_t2"] = True  # REGEN T+2 slice
    return merged


# ── GDZ load ──────────────────────────────────────────────────────────────────

def load_gdz() -> pd.DataFrame:
    if not GDZ_MASTER.exists():
        raise RuntimeError(f"GDZ master missing: {GDZ_MASTER}")
    master = pd.read_parquet(GDZ_MASTER)
    master["Tarih"] = pd.to_datetime(master["Tarih"])
    act = master[["Tarih", GDZ_RAW_TARGET]].copy()
    act["date"] = act["Tarih"].dt.date
    act["hour"] = act["Tarih"].dt.hour.astype(int)
    act = act.rename(columns={GDZ_RAW_TARGET: "Actual_MWh"})

    arch = GDZ_LIVE_DIR / "output" / "archive"
    rows = []
    for f in sorted(arch.glob("*_full48h.parquet")):
        # filename: {issue}_run_{target}_full48h.parquet
        name = f.stem  # 2026-07-09_run_2026-07-10_full48h
        parts = name.replace("_full48h", "").split("_run_")
        if len(parts) != 2:
            continue
        target_date = parts[1]
        df = pd.read_parquet(f)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        # Prefer T+2 rows for delivery KPI
        hz = df["horizon_day"].astype(str)
        t2 = df[hz.str.contains(r"T\+2|T2", regex=True, na=False)].copy()
        if t2.empty:
            # fallback: rows matching target date
            t2 = df[df["Datetime"].dt.date == pd.Timestamp(target_date).date()].copy()
        if t2.empty:
            continue
        # coalesce T1/T2 model cols → canonical
        for short, t1, t2c in [
            ("XGB_Pred", "XGB_T1_Pred", "XGB_T2_Pred"),
            ("LGBM_Pred", "LGBM_T1_Pred", "LGBM_T2_Pred"),
            ("CAT_Pred", "CAT_T1_Pred", "CAT_T2_Pred"),
            ("CHRONOS_Pred", "CHRONOS_T1_Pred", "CHRONOS_T2_Pred"),
        ]:
            a = t2[t1] if t1 in t2.columns else pd.Series(np.nan, index=t2.index)
            b = t2[t2c] if t2c in t2.columns else pd.Series(np.nan, index=t2.index)
            t2[short] = b.fillna(a)
        if t2[MODEL_COLS_ADM].isna().all(axis=None):
            continue
        t2["target_date"] = target_date
        t2["hour"] = t2["Datetime"].dt.hour.astype(int)
        t2["date"] = t2["Datetime"].dt.date
        t2["day_type"] = t2["Datetime"].map(_day_type_from_dt)
        t2["is_t2"] = True
        t2["subst_delta"] = 0.0
        t2["pv_bias_delta"] = 0.0
        rows.append(t2)

    if not rows:
        raise RuntimeError("GDZ: no archive rows with predictions")

    preds = pd.concat(rows, ignore_index=True)
    # dedupe: same target_date keep latest issue (last file wins already via sort + concat — keep last)
    preds = preds.sort_values(["target_date", "Datetime"])
    preds = preds.drop_duplicates(subset=["target_date", "hour"], keep="last")
    merged = preds.merge(act[["date", "hour", "Actual_MWh"]], on=["date", "hour"], how="inner")
    return merged


def daily_metrics(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for td, g in df.groupby("target_date"):
        p, a = g[pred_col].to_numpy(), g["Actual_MWh"].to_numpy()
        rows.append({
            "target_date": td,
            "day_type": g["day_type"].iloc[0] if "day_type" in g.columns else "",
            "mape": _mape(p, a),
            "me": _me(p, a),
            "n": len(g),
        })
    return pd.DataFrame(rows).sort_values("target_date")


def overall(df: pd.DataFrame, pred: np.ndarray) -> dict:
    act = df["Actual_MWh"].to_numpy()
    return {
        "mape": round(_mape(pred, act), 4),
        "me": round(_me(pred, act), 2),
        "n_hours": int(len(df)),
        "n_days": int(df["target_date"].nunique()),
    }


def split_tune_holdout(df: pd.DataFrame, holdout_days: int):
    days = sorted(df["target_date"].unique())
    if len(days) <= holdout_days + 2:
        # too few — use last 30% as holdout min 2
        n_h = max(2, len(days) // 3)
    else:
        n_h = holdout_days
    hold = set(days[-n_h:])
    tune = set(days[:-n_h]) if len(days) > n_h else set(days)
    return df[df["target_date"].isin(tune)].copy(), df[df["target_date"].isin(hold)].copy(), sorted(tune), sorted(hold)


def simulate_final(df: pd.DataFrame, weights: dict, bias: dict | None) -> np.ndarray:
    ens = weighted_ensemble(df, weights, MODEL_COLS_ADM)
    if bias is None:
        # zero level bias — still add residual post deltas if present
        out = ens + df["subst_delta"].to_numpy() + df["pv_bias_delta"].to_numpy()
        return out
    day_type = df["day_type"].to_numpy()
    is_t2 = df["is_t2"].to_numpy() if "is_t2" in df.columns else np.ones(len(df), dtype=bool)
    biased = apply_bias(ens, day_type, is_t2, bias)
    return biased + df["subst_delta"].to_numpy() + df["pv_bias_delta"].to_numpy()


def optimize_weights(tune: pd.DataFrame, bias: dict | None, step: float = 0.05) -> tuple[dict, dict]:
    best_w, best_mape, best_me = None, 1e9, 0.0
    n = 0
    for w in weight_grid(step=step):
        pred = simulate_final(tune, w, bias)
        m = _mape(pred, tune["Actual_MWh"].to_numpy())
        n += 1
        if m < best_mape - 1e-9:
            best_mape = m
            best_me = _me(pred, tune["Actual_MWh"].to_numpy())
            best_w = w
    return best_w, {"mape": best_mape, "me": best_me, "grid_n": n}


def optimize_bias(tune: pd.DataFrame, weights: dict) -> tuple[dict, dict]:
    """Grid T2 bias and scales; T1 linked as 2/3 of T2 (approx live ratio 10/15)."""
    best_b, best_mape, best_me = None, 1e9, 0.0
    t2_levels = [0, 5, 8, 10, 12, 15, 18, 20, 25, 30]
    we_scales = [0.0, 0.1, 0.2, 0.3, 0.5]
    su_scales = [0.0, 0.3, 0.5, 0.6, 0.8, 1.0]
    n = 0
    for t2, we, su in itertools.product(t2_levels, we_scales, su_scales):
        b = {
            "t1": round(t2 * (10 / 15), 2) if t2 else 0.0,
            "t2": float(t2),
            "we_t1": float(we),
            "we_t2": float(we),
            "su_t1": float(su),
            "su_t2": float(su),
        }
        pred = simulate_final(tune, weights, b)
        m = _mape(pred, tune["Actual_MWh"].to_numpy())
        n += 1
        if m < best_mape - 1e-9:
            best_mape = m
            best_me = _me(pred, tune["Actual_MWh"].to_numpy())
            best_b = b
    # also evaluate zero bias
    pred0 = simulate_final(tune, weights, {"t1": 0, "t2": 0, "we_t1": 0, "we_t2": 0, "su_t1": 0, "su_t2": 0})
    m0 = _mape(pred0, tune["Actual_MWh"].to_numpy())
    if m0 < best_mape:
        best_b = {"t1": 0, "t2": 0, "we_t1": 0, "we_t2": 0, "su_t1": 0, "su_t2": 0}
        best_mape = m0
        best_me = _me(pred0, tune["Actual_MWh"].to_numpy())
    return best_b, {"mape": best_mape, "me": best_me, "grid_n": n}


def run_edas(
    edas: str,
    df: pd.DataFrame,
    live_weights: dict,
    live_bias: dict | None,
    holdout_days: int,
    weight_step: float,
    min_holdout_gain: float,
) -> dict:
    print(f"\n{'='*60}\n{edas}: {df['target_date'].nunique()} days, {len(df)} hours\n{'='*60}")

    # solo model MAPE (full period with actuals)
    solo = {}
    for c in MODEL_COLS_ADM:
        if c in df.columns and df[c].notna().any():
            mask = df[c].notna()
            solo[c] = round(_mape(df.loc[mask, c].to_numpy(), df.loc[mask, "Actual_MWh"].to_numpy()), 4)

    # baseline: stored Final if present else live weights + live bias
    if "Final_Pred" in df.columns and df["Final_Pred"].notna().any():
        base_pred = df["Final_Pred"].to_numpy()
        baseline_label = "stored_Final_Pred"
    else:
        base_pred = simulate_final(df, live_weights, live_bias)
        baseline_label = "live_weights+bias"

    base_all = overall(df, base_pred)
    print(f"  solo MAPE: {solo}")
    print(f"  baseline ({baseline_label}): MAPE={base_all['mape']:.3f}% ME={base_all['me']:+.1f}")

    tune, hold, tune_days, hold_days = split_tune_holdout(df, holdout_days)
    print(f"  tune days ({len(tune_days)}): {tune_days[0]}..{tune_days[-1]}")
    print(f"  hold days ({len(hold_days)}): {hold_days}")

    # baseline on holdout
    if "Final_Pred" in hold.columns:
        hold_base = hold["Final_Pred"].to_numpy()
    else:
        hold_base = simulate_final(hold, live_weights, live_bias)
    hold_base_m = overall(hold, hold_base)

    # Phase A: optimize weights with LIVE bias (or zero if live_bias None)
    bias_for_w = live_bias
    best_w, w_tune = optimize_weights(tune, bias_for_w, step=weight_step)
    hold_w_pred = simulate_final(hold, best_w, bias_for_w)
    hold_w = overall(hold, hold_w_pred)
    print(f"  best weights (tune MAPE {w_tune['mape']:.3f}%): {best_w}")
    print(f"  holdout weights-only: MAPE={hold_w['mape']:.3f}% (base {hold_base_m['mape']:.3f}%)")

    # Phase B: re-optimize bias on top of best weights
    best_b, b_tune = optimize_bias(tune, best_w)
    hold_wb_pred = simulate_final(hold, best_w, best_b)
    hold_wb = overall(hold, hold_wb_pred)
    print(f"  best bias (tune MAPE {b_tune['mape']:.3f}%): {best_b}")
    print(f"  holdout weights+bias: MAPE={hold_wb['mape']:.3f}% ME={hold_wb['me']:+.1f}")

    # full-period metrics for chosen configs
    full_w = overall(df, simulate_final(df, best_w, bias_for_w))
    full_wb = overall(df, simulate_final(df, best_w, best_b))
    full_live_sim = overall(df, simulate_final(df, live_weights, live_bias))

    # gates
    gain_w = hold_base_m["mape"] - hold_w["mape"]
    gain_wb = hold_base_m["mape"] - hold_wb["mape"]
    accept_weights = gain_w >= min_holdout_gain and abs(hold_w["me"]) <= 20
    # prefer wb if better than w and ME ok
    accept_bias = gain_wb >= min_holdout_gain and abs(hold_wb["me"]) <= 20 and hold_wb["mape"] <= hold_w["mape"] + 0.05

    recommended = {
        "weights": best_w if accept_weights else live_weights,
        "bias": best_b if accept_bias else live_bias,
        "accept_weights": accept_weights,
        "accept_bias": accept_bias,
        "holdout_gain_weights_pp": round(gain_w, 4),
        "holdout_gain_weights_bias_pp": round(gain_wb, 4),
    }
    if accept_weights and accept_bias:
        rec_pred = simulate_final(df, best_w, best_b)
        rec_hold = hold_wb
    elif accept_weights:
        rec_pred = simulate_final(df, best_w, live_bias)
        rec_hold = hold_w
    else:
        rec_pred = base_pred
        rec_hold = hold_base_m
        recommended["weights"] = live_weights
        recommended["bias"] = live_bias

    rec_full = overall(df, rec_pred)
    print(f"  GATE weights: {'ACCEPT' if accept_weights else 'REJECT'} (gain {gain_w:+.3f}pp)")
    print(f"  GATE bias:    {'ACCEPT' if accept_bias else 'REJECT'} (gain {gain_wb:+.3f}pp)")
    print(f"  recommended full MAPE={rec_full['mape']:.3f}% (baseline {base_all['mape']:.3f}%)")

    # daily csv helper
    df_out = df.copy()
    df_out["pred_recommended"] = rec_pred
    df_out["pred_best_w"] = simulate_final(df, best_w, bias_for_w)
    df_out["pred_best_wb"] = simulate_final(df, best_w, best_b)
    daily = daily_metrics(df_out, "pred_recommended")
    if "Final_Pred" in df.columns:
        daily_base = daily_metrics(df.assign(tmp=df["Final_Pred"]), "tmp") if False else None
        d_base = []
        for td, g in df.groupby("target_date"):
            d_base.append({
                "target_date": td,
                "mape_baseline": _mape(g["Final_Pred"].to_numpy(), g["Actual_MWh"].to_numpy()),
                "me_baseline": _me(g["Final_Pred"].to_numpy(), g["Actual_MWh"].to_numpy()),
            })
        daily = daily.merge(pd.DataFrame(d_base), on="target_date", how="left")
    daily["mape_recommended"] = daily["mape"]
    daily["me_recommended"] = daily["me"]

    report = {
        "edas": edas,
        "generated": str(date.today()),
        "n_days": int(df["target_date"].nunique()),
        "tune_days": tune_days,
        "holdout_days": hold_days,
        "solo_mape": solo,
        "baseline": {"label": baseline_label, **base_all},
        "live_sim": full_live_sim,
        "live_weights": live_weights,
        "live_bias": live_bias,
        "best_weights": best_w,
        "best_bias": best_b,
        "tune_weight_opt": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in w_tune.items()},
        "tune_bias_opt": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b_tune.items()},
        "holdout_baseline": hold_base_m,
        "holdout_best_weights": hold_w,
        "holdout_best_weights_bias": hold_wb,
        "full_best_weights": full_w,
        "full_best_weights_bias": full_wb,
        "recommended": recommended,
        "recommended_full": rec_full,
        "recommended_holdout": rec_hold if isinstance(rec_hold, dict) else overall(hold, rec_pred),
        "min_holdout_gain_pp": min_holdout_gain,
        "note": "Perfect-prog / as-of REGEN for ADM; live archive for GDZ. Separate models per EDAS.",
    }
    return report, daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edas", choices=["ADM", "GDZ", "BOTH"], default="BOTH")
    ap.add_argument("--holdout-days", type=int, default=7)
    ap.add_argument("--weight-step", type=float, default=0.05)
    ap.add_argument("--min-holdout-gain", type=float, default=0.20,
                    help="Minimum holdout MAPE improvement (pp) to accept config change")
    args = ap.parse_args()

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = ["ADM", "GDZ"] if args.edas == "BOTH" else [args.edas]

    # GDZ live weights from config if importable
    gdz_weights = {"XGB_Pred": 0.27, "LGBM_Pred": 0.27, "CAT_Pred": 0.24, "CHRONOS_Pred": 0.22}
    gdz_bias = None  # pass-through today
    try:
        sys.path.insert(0, str(GDZ_LIVE_DIR))
        import config_live_gdz as CG  # noqa
        dw = getattr(CG, "DEFAULT_ENSEMBLE_WEIGHTS", None)
        if dw:
            gdz_weights = {
                "XGB_Pred": float(dw.get("XGB", dw.get("XGB_Pred", 0.25))),
                "LGBM_Pred": float(dw.get("LGBM", dw.get("LGBM_Pred", 0.25))),
                "CAT_Pred": float(dw.get("CAT", dw.get("CAT_Pred", 0.25))),
                "CHRONOS_Pred": float(dw.get("CHRONOS", dw.get("CHRONOS_Pred", 0.25))),
            }
    except Exception as e:
        print(f"[warn] GDZ config import failed, using defaults: {e}")

    for edas in targets:
        if edas == "ADM":
            df = load_adm()
            report, daily = run_edas(
                "ADM", df, ADM_LIVE_WEIGHTS, ADM_BIAS,
                args.holdout_days, args.weight_step, args.min_holdout_gain,
            )
        else:
            df = load_gdz()
            report, daily = run_edas(
                "GDZ", df, gdz_weights, gdz_bias,
                min(args.holdout_days, max(2, df["target_date"].nunique() // 3)),
                args.weight_step, args.min_holdout_gain,
            )

        out_json = C.OUTPUT_DIR / f"ensemble_opt_report_{edas}.json"
        out_csv = C.OUTPUT_DIR / f"ensemble_opt_daily_{edas}.csv"
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        daily.to_csv(out_csv, index=False)
        print(f"  wrote {out_json.name} + {out_csv.name}")


if __name__ == "__main__":
    main()
