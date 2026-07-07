"""
05_postprocess.py — Post-Process (Holiday Sub + PV Bias)
=========================================================
Donmuş artefaktları yükleyip ham tahmine uygular.
Backtest'in _apply_final_postprocess() ile aynı mantık, tek-fold versiyonu.

Giriş:  data/weather_cache/raw_predictions.parquet
         data/weather_cache/feature_matrix.parquet  (context için)
         models/holiday_blend_alphas*.json
         models/pv_bias_lookup*.json
Çıkış:  data/weather_cache/postprocessed_predictions.parquet
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config_live import (
    DATA_DIR, MODELS_DIR,
    RAW_TARGET_COL, RAW_DATE_COL, RAW_HOUR_COL, TEST_SIZE,
    ENABLE_HOLIDAY_SUBSTITUTION, HOLIDAY_SUBSTITUTION_TEMP_COL,
    HOLIDAY_BLEND_ALPHAS, HOLIDAY_BLEND_ALPHAS_T2,
    ENABLE_PV_BIAS_CORRECTION,
    PV_BIAS_SOLAR_HOURS, PV_BIAS_MIN_SAMPLES_PER_CELL,
    PV_BIAS_FIT_EXCLUDE_HOLIDAYS, PV_BIAS_FALLBACK_ENABLED,
    PV_BIAS_LOOKUP_T1, PV_BIAS_LOOKUP_T2,
    POST_HOLIDAY_MULTIPLIERS_T1, POST_HOLIDAY_MULTIPLIERS_T2,
    ENSEMBLE_BIAS_CORRECTION_T1_MWH, ENSEMBLE_BIAS_CORRECTION_T2_MWH,
    ENSEMBLE_BIAS_WEEKEND_SCALE_T1, ENSEMBLE_BIAS_WEEKEND_SCALE_T2,
    ENSEMBLE_BIAS_SUNDAY_SCALE_T1, ENSEMBLE_BIAS_SUNDAY_SCALE_T2,
)

FEATURE_MATRIX_PATH    = DATA_DIR / "weather_cache" / "feature_matrix.parquet"
RAW_PREDICTIONS_PATH   = DATA_DIR / "weather_cache" / "raw_predictions.parquet"
POSTPROC_PATH          = DATA_DIR / "weather_cache" / "postprocessed_predictions.parquet"


def _add_local_src_path():
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def apply_holiday_substitution(preds: pd.Series, feature_df: pd.DataFrame, predict_idx) -> pd.Series:
    """Tatil substitution — donmuş alpha tablosunu yükleyip uygula."""
    _add_local_src_path()
    from src.holiday_substitution import apply_substitution

    if not HOLIDAY_BLEND_ALPHAS.exists():
        print("     [HolidaySub] alpha dosyası bulunamadı, atlanıyor.")
        return preds

    # is_t2 maskesi: 48 saatin ikinci yarısı (24-47) T+2
    is_t2 = np.zeros(len(predict_idx), dtype=bool)
    is_t2[TEST_SIZE // 2:] = True

    temp_col = (
        HOLIDAY_SUBSTITUTION_TEMP_COL
        if HOLIDAY_SUBSTITUTION_TEMP_COL in feature_df.columns else None
    )
    cols = [c for c in (RAW_TARGET_COL, temp_col, "days_since_holiday_end") if c and c in feature_df.columns]
    df_h = feature_df.loc[predict_idx, cols] if cols else feature_df.loc[predict_idx, [RAW_TARGET_COL]]

    blend_t2_path = str(HOLIDAY_BLEND_ALPHAS_T2) if HOLIDAY_BLEND_ALPHAS_T2.exists() else None

    sub_pred, sub_stats = apply_substitution(
        predict_idx,
        preds,
        feature_df[cols] if cols else feature_df[[RAW_TARGET_COL]],
        RAW_TARGET_COL,
        base_temp_col=temp_col,
        days_since_series=(
            feature_df["days_since_holiday_end"].reindex(predict_idx)
            if "days_since_holiday_end" in feature_df.columns else None
        ),
        blend_alphas_path=str(HOLIDAY_BLEND_ALPHAS),
        is_t2=is_t2,
        blend_alphas_t2_path=blend_t2_path,
        post_holiday_multipliers=POST_HOLIDAY_MULTIPLIERS_T1,
        post_holiday_multipliers_t2=POST_HOLIDAY_MULTIPLIERS_T2,
    )
    print(f"     [HolidaySub] uygulandı | stats: {sub_stats}")
    return sub_pred


def apply_pv_bias(preds: pd.Series, feature_df: pd.DataFrame, predict_idx) -> pd.Series:
    """PV bias correction — donmuş lookup tablolarını yükleyip uygula."""
    _add_local_src_path()
    from src.pv_bias_correction import PVBiasCorrector

    if not PV_BIAS_LOOKUP_T1.exists():
        print("     [PVBias] lookup bulunamadı, atlanıyor.")
        return preds

    # GHI verisi
    if "GHI_ADM_Weighted" not in feature_df.columns:
        print("     [PVBias] GHI_ADM_Weighted eksik, atlanıyor.")
        return preds

    ghi_series = feature_df["GHI_ADM_Weighted"].reindex(predict_idx)

    is_t2_arr = np.zeros(len(predict_idx), dtype=bool)
    is_t2_arr[TEST_SIZE // 2:] = True

    corrected = preds.copy()
    for h_flag, h_path, h_label in (
        (False, PV_BIAS_LOOKUP_T1, "T1"),
        (True,  PV_BIAS_LOOKUP_T2, "T2"),
    ):
        if not h_path.exists():
            continue
        # DİKKAT: PVBiasCorrector.load() bir @classmethod — YENİ, fit edilmiş
        # bir nesne DÖNDÜRÜR, instance'ı yerinde değiştirmez. Dönüş değerini
        # yakalamamak "fit() henüz çağrılmadı" hatasına yol açıyordu.
        corrector = PVBiasCorrector.load(str(h_path))
        sel = np.array(is_t2_arr == h_flag)
        sel_idx = np.array(predict_idx)[sel]
        if not sel.any():
            continue
        preds_h = preds.iloc[sel]
        ghi_h   = ghi_series.iloc[sel]
        corrected.iloc[sel] = corrector.transform(preds_h, ghi_h).to_numpy()
        print(f"     [PVBias] {h_label} uygulandı ({int(sel.sum())} saat)")

    return corrected


def run() -> dict:
    """
    Adım 05 — post-process.

    Returns:
        {"status": "ok", "pre_mean": ..., "post_mean": ..., "delta_mean": ...}
    """
    print("\n[05] Post-process uygulanıyor...")

    raw_preds_df  = pd.read_parquet(RAW_PREDICTIONS_PATH)
    feature_df    = pd.read_parquet(FEATURE_MATRIX_PATH)

    # Bilinen sorun (Boray main.py'de de aynı savunma var — "2018-04-11 00:00
    # collision"): DataManager'ın rekonstrükte ettiği DatetimeIndex nadiren
    # duplicate üretebiliyor. reindex/loc bunu tolere etmiyor, dedup şart.
    if feature_df.index.has_duplicates:
        feature_df = feature_df[~feature_df.index.duplicated(keep="first")]

    preds = raw_preds_df["Ensemble_Pred"].copy()
    # DİKKAT: raw_preds_df, 04_predict_48h.py'de reset_index(drop=True) ile
    # düz 0-47 integer index'e sahip (Datetime ayrı bir kolon). feature_df ise
    # DataManager'ın kurduğu DatetimeIndex'i kullanıyor — bu yüzden predict_idx
    # feature_df'i indekslemek için raw_preds_df["Datetime"] değerlerinden
    # kurulmalı, düz integer index'ten DEĞİL.
    predict_idx = pd.DatetimeIndex(raw_preds_df["Datetime"])
    preds.index = predict_idx
    pre_mean = float(preds.mean())

    # ── Ensemble bias düzeltme (sistematik under-estimation karşıtı) ──────────
    # Gün-tipi duyarlı: hafta sonu yükleri düşük, bias over-correction riski var.
    # Pazar < Cumartesi < Hafta içi bias scaling.
    is_t2 = raw_preds_df["horizon_day"].to_numpy() == "T+2"
    bias_arr = np.where(is_t2, ENSEMBLE_BIAS_CORRECTION_T2_MWH, ENSEMBLE_BIAS_CORRECTION_T1_MWH)

    day_type = raw_preds_df["day_type"].to_numpy()
    is_sunday = day_type == "pazar"
    is_weekend = (day_type == "cumartesi") | is_sunday

    scale_t1 = np.where(is_sunday, ENSEMBLE_BIAS_SUNDAY_SCALE_T1,
                        np.where(is_weekend, ENSEMBLE_BIAS_WEEKEND_SCALE_T1, 1.0))
    scale_t2 = np.where(is_sunday, ENSEMBLE_BIAS_SUNDAY_SCALE_T2,
                        np.where(is_weekend, ENSEMBLE_BIAS_WEEKEND_SCALE_T2, 1.0))
    bias_scale = np.where(is_t2, scale_t2, scale_t1)

    bias_arr = bias_arr * bias_scale
    preds = preds + bias_arr
    bias_total = float(bias_arr.sum())

    after_sub = preds
    if ENABLE_HOLIDAY_SUBSTITUTION:
        after_sub = apply_holiday_substitution(preds, feature_df, predict_idx)
    else:
        print("     [HolidaySub] devre dışı")
    subst_delta = (after_sub - preds).to_numpy()

    after_pv = after_sub
    if ENABLE_PV_BIAS_CORRECTION:
        after_pv = apply_pv_bias(after_sub, feature_df, predict_idx)
    else:
        print("     [PVBias] devre dışı")
    pv_bias_delta = (after_pv - after_sub).to_numpy()

    preds = after_pv
    post_mean = float(preds.mean())
    result_df = raw_preds_df.copy()
    result_df["bias_delta"] = bias_arr
    result_df["subst_active"] = subst_delta != 0
    result_df["subst_delta"] = subst_delta
    result_df["pv_bias_delta"] = pv_bias_delta
    result_df["Final_Pred"] = preds.values

    POSTPROC_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(POSTPROC_PATH, index=False)

    print(f"     Ham ortalama: {pre_mean:.1f} MWh  →  Final: {post_mean:.1f} MWh  (Δ {post_mean-pre_mean:+.1f}, bias {bias_total:+.1f})")

    return {
        "status": "ok",
        "pre_mean":   round(pre_mean, 2),
        "post_mean":  round(post_mean, 2),
        "delta_mean": round(post_mean - pre_mean, 2),
        "bias_total": round(bias_total, 2),
    }


if __name__ == "__main__":
    result = run()
    print(result)
