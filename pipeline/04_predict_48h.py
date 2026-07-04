"""
04_predict_48h.py — 48 Saatlik Ham Tahmin (Full Ensemble)
==========================================================
4 taban model (XGB + LGBM + CatBoost + Chronos-2) + Ridge stacker.
3-kademeli recursive: T+0 (valid Lag) → T+1 → T+2.
Holiday-aware lag cleaning recursive T+2 için.

Giriş:  data/weather_cache/feature_matrix.parquet
Çıkış:  data/weather_cache/raw_predictions.parquet
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config_live import (
    DATA_DIR, MODELS_DIR, OOF_HISTORY_PATH,
    RAW_TARGET_COL, RAW_DATE_COL, RAW_HOUR_COL,
    TEST_SIZE, MAX_TRAIN_SIZE, WARMUP_PERIOD,
    MODEL_XGB_PATH, MODEL_LGBM_PATH, MODEL_LGBM_WD_SAT, MODEL_LGBM_WE,
    MODEL_XGB_WD_SAT, MODEL_XGB_WE, MODEL_STACKER_PATH,
    CHRONOS_MODEL_ID, CHRONOS_ADAPTER_PATH, CHRONOS_CONTEXT_LENGTH,
    CHRONOS_FORCE_CPU, CHRONOS_USE_COVARIATES,
)

FEATURE_MATRIX_PATH   = DATA_DIR / "weather_cache" / "feature_matrix.parquet"
RAW_PREDICTIONS_PATH  = DATA_DIR / "weather_cache" / "raw_predictions.parquet"


def _add_local_src_path():
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


# ── Split ────────────────────────────────────────────────────────────────────

def split_train_predict(feature_df: pd.DataFrame):
    """3-kademeli recursive: T+0 → T+1 → T+2. Teslim: T+1 + T+2 = 48h."""
    all_nan = feature_df.index[feature_df[RAW_TARGET_COL].isna()]
    n_nan = len(all_nan)
    if n_nan < 48:
        raise ValueError(f"En az 48 NaN satır bekleniyor, {n_nan} var.")

    train_idx = feature_df.index[~feature_df[RAW_TARGET_COL].isna()]
    if len(train_idx) > MAX_TRAIN_SIZE:
        train_idx = train_idx[-MAX_TRAIN_SIZE:]

    if n_nan >= 72:
        steps = [all_nan[:24], all_nan[24:48], all_nan[48:72]]
        deliver = steps[1:]
    elif n_nan >= 48:
        steps = [all_nan[:24], all_nan[24:48]]
        deliver = steps
    else:
        steps = [all_nan[:24]]
        deliver = steps

    return train_idx, steps, deliver


def get_feature_cols(feature_df: pd.DataFrame) -> list:
    cols = feature_df.select_dtypes(include=["number", "category", "bool"]).columns.tolist()
    if RAW_TARGET_COL in cols:
        cols.remove(RAW_TARGET_COL)
    return cols


# ── Holiday-aware recursive lag update ───────────────────────────────────────

def _recompute_lags_for_t2(df: pd.DataFrame, t1_idx, t2_idx):
    """
    T+1 tahminleriyle T+2 lag feature'larını güncelle.
    Holiday-aware: tatil saatlerinde pre-cleaned lag kullanır,
    religious_post_1 günlerinde recovery_lag kullanır.
    (Boray recursive_evaluator.py:111-212 ile aynı mantık)
    """
    _add_local_src_path()
    from config_live import ENABLE_HOLIDAY_LAG_CLEAN
    from src.holiday_calendar import build_holiday_calendar, is_holiday_hour_for_lag, build_event_window_map

    s = df[RAW_TARGET_COL].copy()

    # Holiday calendar hazırlığı
    cal = None
    event_map = {}
    if ENABLE_HOLIDAY_LAG_CLEAN:
        years = list(range(df.index.year.min() - 1, df.index.year.max() + 2))
        cal = build_holiday_calendar(years)
        event_map = build_event_window_map(years=years)

    # T+1 saatleri için holiday-aware clean prediction'lar hazırla
    preds_clean_t1 = []
    for h in range(len(t1_idx)):
        ts_t1 = t1_idx[h]
        is_hol = ENABLE_HOLIDAY_LAG_CLEAN and is_holiday_hour_for_lag(ts_t1, cal)
        lag_24_col = f"{RAW_TARGET_COL}_Lag24h"
        if is_hol and lag_24_col in df.columns:
            pred_val = df.loc[ts_t1, lag_24_col]
        else:
            pred_val = s.loc[ts_t1]
        preds_clean_t1.append(pred_val)

    # T+2 satırlarına uygula
    for h in range(len(t2_idx)):
        row_ts = t2_idx[h]
        pred_val = preds_clean_t1[h]

        # religious_post_1 gününde recovery_lag kullan
        is_recovery = False
        if ENABLE_HOLIDAY_LAG_CLEAN:
            lbl = event_map.get(row_ts.date(), "normal")
            if lbl == "religious_post_1":
                is_recovery = True

        if is_recovery and "post_holiday_recovery_lag_24" in df.columns:
            pred_val_for_lag = df.loc[row_ts, "post_holiday_recovery_lag_24"]
        else:
            pred_val_for_lag = pred_val

        # Lag24-27h
        for lag_h, offset in [(24, 0), (25, 1), (26, 2), (27, 3)]:
            col = f"{RAW_TARGET_COL}_Lag{lag_h}h"
            if col in df.columns and h - offset >= 0:
                df.loc[row_ts, col] = preds_clean_t1[h - offset]
            elif col in df.columns and h - offset < 0:
                df.loc[row_ts, col] = s.shift(lag_h).loc[row_ts]

        # lag_24_clean
        if "lag_24_clean" in df.columns:
            df.loc[row_ts, "lag_24_clean"] = pred_val_for_lag

        # lag_24_t2_proxy
        if "lag_24_t2_proxy" in df.columns:
            df.loc[row_ts, "lag_24_t2_proxy"] = pred_val_for_lag

        # lag_24_chain_clean
        if "lag_24_chain_clean" in df.columns:
            df.loc[row_ts, "lag_24_chain_clean"] = pred_val_for_lag

        # Mean_Last_3_Days_Same_Hour
        if "Mean_Last_3_Days_Same_Hour" in df.columns:
            lag48_col = f"{RAW_TARGET_COL}_Lag48h"
            lag72_col = f"{RAW_TARGET_COL}_Lag72h"
            lag48_val = df.loc[row_ts, lag48_col] if lag48_col in df.columns else pred_val_for_lag
            lag72_val = df.loc[row_ts, lag72_col] if lag72_col in df.columns else pred_val_for_lag
            df.loc[row_ts, "Mean_Last_3_Days_Same_Hour"] = (pred_val_for_lag + lag48_val + lag72_val) / 3.0

        # Last_Workday_Lag (Mon→72h, Sun→48h, else→24h)
        if "Last_Workday_Lag" in df.columns:
            dow = row_ts.dayofweek
            if dow in (1, 2, 3, 4, 5):
                df.loc[row_ts, "Last_Workday_Lag"] = pred_val_for_lag

        # A-family features
        if "Load_Chg_24h" in df.columns:
            df.loc[row_ts, "Load_Chg_24h"] = pred_val_for_lag - s.shift(48).loc[row_ts]
        if "Load_Chg_168h" in df.columns:
            df.loc[row_ts, "Load_Chg_168h"] = pred_val_for_lag - s.shift(168).loc[row_ts]
        if "Load_Dev_3d" in df.columns and "Mean_Last_3_Days_Same_Hour" in df.columns:
            df.loc[row_ts, "Load_Dev_3d"] = pred_val_for_lag - df.loc[row_ts, "Mean_Last_3_Days_Same_Hour"]

        eps = 1.0
        if "Load_Ratio_3d" in df.columns and "Mean_Last_3_Days_Same_Hour" in df.columns:
            df.loc[row_ts, "Load_Ratio_3d"] = pred_val_for_lag / (df.loc[row_ts, "Mean_Last_3_Days_Same_Hour"] + eps)
        if "Load_Ratio_168h" in df.columns:
            df.loc[row_ts, "Load_Ratio_168h"] = pred_val_for_lag / (s.shift(168).loc[row_ts] + eps)
        if "Load_Ratio_Workday" in df.columns and "Last_Workday_Lag" in df.columns:
            df.loc[row_ts, "Load_Ratio_Workday"] = pred_val_for_lag / (df.loc[row_ts, "Last_Workday_Lag"] + eps)

        # Rolling_Mean_3h_Lag24h
        if "Rolling_Mean_3h_Lag24h" in df.columns:
            rm = s.shift(24).rolling(3, min_periods=1).mean()
            df.loc[row_ts, "Rolling_Mean_3h_Lag24h"] = rm.loc[row_ts]

        # lag_24_anomaly
        if "lag_24_anomaly" in df.columns and f"{RAW_TARGET_COL}_Lag168h" in df.columns:
            lag168_val = df.loc[row_ts, f"{RAW_TARGET_COL}_Lag168h"]
            df.loc[row_ts, "lag_24_anomaly"] = abs(pred_val_for_lag - lag168_val)

        # transition_signal
        if "transition_signal" in df.columns and "Mean_Last_3_Days_Same_Hour" in df.columns:
            df.loc[row_ts, "transition_signal"] = df.loc[row_ts, "Mean_Last_3_Days_Same_Hour"] - pred_val_for_lag


# ── Recursive prediction ─────────────────────────────────────────────────────

def _predict_recursive(model, feature_df, train_idx, steps, feature_cols):
    """Çok adımlı recursive: Adım 0 valid Lag, sonra her adımda Lag update."""
    X_train = feature_df.loc[train_idx, feature_cols]
    y_train = feature_df.loc[train_idx, RAW_TARGET_COL]
    model.train_model(X_train, y_train, X_train.iloc[:1], y_train.iloc[:1])

    df_rec = feature_df.copy()
    all_preds = []

    for step_i, step_idx in enumerate(steps):
        if step_i > 0:
            _recompute_lags_for_t2(df_rec, steps[step_i - 1], step_idx)

        X_step = df_rec.loc[step_idx, feature_cols]
        step_pred = model.model.predict(X_step)
        df_rec.loc[step_idx, RAW_TARGET_COL] = step_pred
        all_preds.append(step_pred)

    return np.concatenate(all_preds)


# ── Model training + prediction ──────────────────────────────────────────────

def train_and_predict_gbdt(feature_df, train_idx, steps, feature_cols) -> dict:
    """XGB + LGBM + CatBoost recursive retrain + predict."""
    _add_local_src_path()
    from src.model_manager import ModelManager
    from src.lightgbm_manager import LightGBMManager

    all_pred_idx = steps[0]
    for s in steps[1:]:
        all_pred_idx = all_pred_idx.append(s)

    preds = {}

    print("     [XGB] recursive (%d kademe)..." % len(steps))
    mm = ModelManager()
    xgb_pred = _predict_recursive(mm, feature_df, train_idx, steps, feature_cols)
    mm.save_model(str(MODEL_XGB_PATH))
    preds["XGB_Pred"] = pd.Series(xgb_pred, index=all_pred_idx)

    print("     [LGBM] recursive (%d kademe)..." % len(steps))
    lgbm = LightGBMManager()
    lgbm_pred = _predict_recursive(lgbm, feature_df, train_idx, steps, feature_cols)
    lgbm.save_model(str(MODEL_LGBM_PATH))
    preds["LGBM_Pred"] = pd.Series(lgbm_pred, index=all_pred_idx)

    # CatBoost (opsiyonel — kütüphane yoksa atla)
    try:
        from src.catboost_manager import CatBoostManager
        print("     [CAT] recursive (%d kademe)..." % len(steps))
        cat = CatBoostManager()
        cat_pred = _predict_recursive(cat, feature_df, train_idx, steps, feature_cols)
        cat.save_model(str(MODELS_DIR / "live_catboost.cbm"))
        preds["CAT_Pred"] = pd.Series(cat_pred, index=all_pred_idx)
    except ImportError:
        print("     [CAT] catboost kütüphanesi yok, atlanıyor")
    except Exception as e:
        print(f"     [CAT] hata: {e}, atlanıyor")

    return preds


# ── Chronos (correct API) ────────────────────────────────────────────────────

def predict_chronos(feature_df, train_idx, all_pred_idx, n_steps) -> pd.Series:
    """Chronos-2 LoRA inference — predict_horizon ile doğru API çağrısı."""
    _add_local_src_path()
    from src.chronos_bridge import prepare_panel_for_chronos, panel_slice_to_predict_frames
    from src.chronos_manager import ChronosInferenceWrapper

    panel = prepare_panel_for_chronos(feature_df.copy(), RAW_TARGET_COL)

    train_positions = np.array([panel.index.get_loc(idx) for idx in train_idx])
    pred_positions = np.array([panel.index.get_loc(idx) for idx in all_pred_idx])

    ctx_df, fut_df = panel_slice_to_predict_frames(
        panel, train_positions, pred_positions, CHRONOS_CONTEXT_LENGTH
    )

    device = "cpu" if CHRONOS_FORCE_CPU else None
    chronos = ChronosInferenceWrapper(
        model_id=CHRONOS_MODEL_ID,
        adapter_path=CHRONOS_ADAPTER_PATH,
        device_map=device,
        context_length=CHRONOS_CONTEXT_LENGTH,
    )

    print(f"     [Chronos] predict_horizon ({n_steps}h)...")
    chronos_pred = chronos.predict_horizon(
        context_df=ctx_df,
        future_df=fut_df,
        prediction_length=n_steps,
    )

    if len(chronos_pred) > len(all_pred_idx):
        chronos_pred = chronos_pred[-len(all_pred_idx):]

    return pd.Series(np.asarray(chronos_pred, dtype=np.float64).ravel(), index=all_pred_idx)


# ── Stacking ─────────────────────────────────────────────────────────────────

def stack_predictions(preds: dict, predict_idx) -> pd.Series:
    """Rolling Ridge (OOF) → frozen stacker → basit ortalama."""
    pred_df = pd.DataFrame(preds, index=predict_idx)
    pred_cols = list(preds.keys())

    # 1. Rolling Ridge (OOF feedback'ten)
    try:
        from src.oof_feedback import get_rolling_ridge
        rolling = get_rolling_ridge(OOF_HISTORY_PATH, pred_cols)
        if rolling is not None:
            available = [c for c in pred_cols if c in pred_df.columns]
            ensemble = rolling.predict(pred_df[available].values)
            print(f"     [Stacker] Rolling Ridge (OOF) uygulandı")
            return pd.Series(ensemble, index=predict_idx)
    except Exception as e:
        print(f"     [Stacker] Rolling Ridge yok: {e}")

    # 2. Frozen stacker (Boray dict)
    if MODEL_STACKER_PATH.exists():
        import joblib
        try:
            stacker = joblib.load(MODEL_STACKER_PATH)
            if isinstance(stacker, dict) and "meta_model" in stacker:
                meta_model = stacker["meta_model"]
                meta_cols = stacker.get("meta_feature_cols", pred_cols)
                available = [c for c in meta_cols if c in pred_df.columns]
                if len(available) == len(meta_cols):
                    ensemble = meta_model.predict(pred_df[meta_cols])
                    print(f"     [Stacker] Frozen Ridge ({stacker.get('best_method', '?')})")
                    return pd.Series(ensemble, index=predict_idx)
                else:
                    missing = set(meta_cols) - set(available)
                    print(f"     [Stacker] Eksik: {missing} → basit ortalama")
            elif hasattr(stacker, "predict"):
                ensemble = stacker.predict(pred_df[pred_cols])
                print(f"     [Stacker] Frozen stacker")
                return pd.Series(ensemble, index=predict_idx)
        except Exception as e:
            print(f"     [Stacker] Uyarı: {e} → basit ortalama")

    # 3. Basit ortalama
    ensemble = pred_df[pred_cols].mean(axis=1)
    print("     [Stacker] Basit ortalama")
    return ensemble


# ── Holiday override (CatBoost solo on weekday holidays) ─────────────────────

def _apply_holiday_override(df: pd.DataFrame, preds: pd.Series) -> pd.Series:
    """Tatil saatlerinde CatBoost solo tahminini kullan (Boray stacking_manager:421-454)."""
    if "CAT_Pred" not in df.columns:
        return preds

    is_holiday = np.zeros(len(df), dtype=bool)
    for col in ["Yilbasi", "Milli_Bayram", "Ramazan_Bayram", "Kurban_Bayram"]:
        if col in df.columns:
            is_holiday |= (df[col] == 1).to_numpy()

    if isinstance(df.index, pd.DatetimeIndex):
        dow = df.index.dayofweek.to_numpy()
    else:
        dow = np.zeros(len(df))

    is_weekday_holiday = is_holiday & (dow < 5)

    if not is_weekday_holiday.any():
        return preds

    out = preds.copy()
    cat_preds = df["CAT_Pred"].to_numpy()
    out.iloc[np.where(is_weekday_holiday)[0]] = cat_preds[is_weekday_holiday]
    print(f"     [Override] {is_weekday_holiday.sum()} tatil saati CatBoost solo ile değiştirildi")
    return out


# ── Main run ─────────────────────────────────────────────────────────────────

def run() -> dict:
    """Adım 04 — 48h recursive tahmin (T+0 → T+1 → T+2)."""
    print("\n[04] 48h recursive tahmin üretiliyor...")

    feature_df = pd.read_parquet(FEATURE_MATRIX_PATH)
    feature_cols = get_feature_cols(feature_df)
    train_idx, steps, deliver = split_train_predict(feature_df)

    deliver_idx = deliver[0].append(deliver[1]) if len(deliver) > 1 else deliver[0]
    all_pred_idx = steps[0]
    for s in steps[1:]:
        all_pred_idx = all_pred_idx.append(s)

    n_total = sum(len(s) for s in steps)
    print(f"     Train: {len(train_idx)}  |  Kademe: {len(steps)}  |  Teslim: {len(deliver_idx)}h")

    # GBDT predictions (XGB + LGBM + CatBoost)
    gbdt_preds = train_and_predict_gbdt(feature_df, train_idx, steps, feature_cols)

    # Chronos prediction
    try:
        chronos_pred = predict_chronos(feature_df, train_idx, all_pred_idx, n_steps=n_total)
        gbdt_preds["CHRONOS_Pred"] = chronos_pred
        print("     [Chronos] tamamlandı")
    except Exception as e:
        print(f"     [Chronos] Uyarı: {e} — atlandı")
        import traceback; traceback.print_exc()
        gbdt_preds["CHRONOS_Pred"] = gbdt_preds["XGB_Pred"].loc[all_pred_idx]

    # Stacking
    ensemble = stack_predictions(
        {k: v.loc[deliver_idx] for k, v in gbdt_preds.items()},
        deliver_idx
    )

    # Holiday override
    pred_df_full = pd.DataFrame(
        {k: v.loc[deliver_idx] for k, v in gbdt_preds.items()},
        index=deliver_idx
    )
    ensemble = _apply_holiday_override(pred_df_full, ensemble)

    gbdt_preds["Ensemble_Pred"] = ensemble

    # Output
    result_df = pd.DataFrame(
        {k: v.loc[deliver_idx] for k, v in gbdt_preds.items()},
        index=deliver_idx
    )
    result_df["Datetime"] = (
        feature_df.loc[deliver_idx, RAW_DATE_COL].values +
        pd.to_timedelta(feature_df.loc[deliver_idx, RAW_HOUR_COL].values, unit="h")
    )
    result_df = result_df.reset_index(drop=True)

    RAW_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(RAW_PREDICTIONS_PATH, index=False)

    print(f"     Ensemble ortalama: {ensemble.mean():.1f} MWh  |  kayıt: {RAW_PREDICTIONS_PATH.name}")

    return {
        "status": "ok",
        "n_predict": len(result_df),
        "xgb_mean":      round(float(gbdt_preds["XGB_Pred"].loc[deliver_idx].mean()), 2),
        "lgbm_mean":     round(float(gbdt_preds["LGBM_Pred"].loc[deliver_idx].mean()), 2),
        "ensemble_mean": round(float(ensemble.mean()), 2),
    }


if __name__ == "__main__":
    result = run()
    print(result)
