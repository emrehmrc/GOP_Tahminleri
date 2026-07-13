"""
test_pipeline_core.py — Pipeline koruma testleri
======================================================
6 test: split, lag, ensemble renorm, bias, csv parse, smoke.
Calistirmak icin: pytest tests/ -v
"""
from __future__ import annotations
import sys, json, math, importlib.util
from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

# Pipeline modulleri (numerik baslangicli dosya adi -> importlib ile yukle)
def _load_pipeline_step(name: str):
    path = ROOT / "pipeline" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Config import (dogrudan calisir, cunku config_live.py gecerli bir modul adi)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from config_live import (
    TEST_SIZE, MAX_TRAIN_SIZE, CALIBRATED_ENSEMBLE_WEIGHTS,
    ENSEMBLE_BIAS_CORRECTION_T1_MWH, ENSEMBLE_BIAS_CORRECTION_T2_MWH,
    ENSEMBLE_BIAS_WEEKEND_SCALE_T1, ENSEMBLE_BIAS_WEEKEND_SCALE_T2,
    ENSEMBLE_BIAS_SUNDAY_SCALE_T1, ENSEMBLE_BIAS_SUNDAY_SCALE_T2,
)
from src.recency_weight import recency_sample_weight


# ── 1. split_train_predict ─────────────────────────────────────────────────

def _make_feature_df(n_nan: int = 48, total: int = 1000):
    dates = pd.date_range("2025-01-01", periods=total, freq="h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "ADM_Dağıtılan_Enerji_(MWh)": rng.uniform(1000, 2000, total).astype(float),
        "feat_a": rng.uniform(0, 1, total),
        "feat_b": rng.uniform(0, 1, total),
    }, index=dates)
    df.iloc[-n_nan:, 0] = np.nan
    return df


def test_split_train_predict_48():
    step04 = _load_pipeline_step("04_predict_48h")
    df = _make_feature_df(n_nan=48)
    train_idx, steps, deliver = step04.split_train_predict(df)
    assert len(steps) >= 2
    assert len(steps[0]) == 24
    assert len(steps[1]) == 24
    assert len(train_idx) == len(df) - 48


def test_split_train_predict_72():
    step04 = _load_pipeline_step("04_predict_48h")
    df = _make_feature_df(n_nan=72)
    train_idx, steps, deliver = step04.split_train_predict(df)
    assert len(steps) == 3
    assert len(deliver) == 2


def test_split_train_predict_under_48_raises():
    step04 = _load_pipeline_step("04_predict_48h")
    df = _make_feature_df(n_nan=24)
    import pytest
    with pytest.raises(ValueError, match="En az 48 NaN"):
        step04.split_train_predict(df)


# ── 2. lag recompute shapes ─────────────────────────────────────────────────

def test_lag_recompute_shapes():
    step04 = _load_pipeline_step("04_predict_48h")
    dates = pd.date_range("2025-06-01", periods=200, freq="h")
    target_col = "ADM_Dağıtılan_Enerji_(MWh)"
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        target_col: rng.uniform(1000, 2000, 200).astype(float),
        f"{target_col}_Lag24h": rng.uniform(1000, 2000, 200).astype(float),
        f"{target_col}_Lag48h": rng.uniform(1000, 2000, 200).astype(float),
        f"{target_col}_Lag72h": rng.uniform(1000, 2000, 200).astype(float),
        f"{target_col}_Lag168h": rng.uniform(1000, 2000, 200).astype(float),
        "Mean_Last_3_Days_Same_Hour": rng.uniform(1000, 2000, 200).astype(float),
        "Last_Workday_Lag": rng.uniform(1000, 2000, 200).astype(float),
        "Load_Chg_24h": rng.uniform(-50, 50, 200).astype(float),
        "Load_Chg_168h": rng.uniform(-50, 50, 200).astype(float),
        "Load_Dev_3d": rng.uniform(-50, 50, 200).astype(float),
        "Load_Ratio_3d": rng.uniform(0.9, 1.1, 200).astype(float),
        "Load_Ratio_168h": rng.uniform(0.9, 1.1, 200).astype(float),
        "Load_Ratio_Workday": rng.uniform(0.9, 1.1, 200).astype(float),
        "lag_24_clean": rng.uniform(1000, 2000, 200).astype(float),
        "lag_48_clean": rng.uniform(1000, 2000, 200).astype(float),
        "lag_24_t2_proxy": rng.uniform(1000, 2000, 200).astype(float),
        "lag_24_chain_clean": rng.uniform(1000, 2000, 200).astype(float),
        "lag_24_anomaly": rng.uniform(0, 100, 200).astype(float),
        "transition_signal": rng.uniform(-50, 50, 200).astype(float),
        "Rolling_Mean_3h_Lag24h": rng.uniform(1000, 2000, 200).astype(float),
        "post_holiday_recovery_lag_24": rng.uniform(1000, 2000, 200).astype(float),
    }, index=dates)
    t1_idx = dates[-48:-24]
    t2_idx = dates[-24:]
    rng2 = np.random.default_rng(63)
    df.loc[t1_idx, target_col] = rng2.uniform(1000, 2000, 24).astype(float)
    step04._recompute_lags_for_t2(df, t1_idx, t2_idx)
    has_nan = df.loc[t2_idx].isna().sum().sum()
    assert has_nan == 0, f"T+2 feature'larinda {has_nan} NaN kaldi"


# ── 3. stack_chronos_fail_renorm ────────────────────────────────────────────

def _dummy_preds(keys: list, n=48):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-07-01", periods=n, freq="h")
    return {k: pd.Series(rng.uniform(1000, 2000, n), index=idx) for k in keys}


def test_stack_chronos_fail_renorm():
    step04 = _load_pipeline_step("04_predict_48h")
    keys = list(CALIBRATED_ENSEMBLE_WEIGHTS.keys())
    preds = _dummy_preds(keys)
    idx = list(preds.values())[0].index
    ens_ok, method_ok, meta_ok = step04.stack_predictions(preds, idx, chronos_ok=True)
    ens_fail, method_fail, meta_fail = step04.stack_predictions(preds, idx, chronos_ok=False)
    assert "no_chronos" in method_fail or "calibrated_static" in method_fail
    assert meta_fail.get("meta_w_chronos") is None, f"Chronos weight silinmeli: {meta_fail}"
    remaining = [meta_fail.get(k) for k in ("meta_w_xgb", "meta_w_lgbm", "meta_w_cat") if meta_fail.get(k) is not None]
    if remaining:
        assert math.isclose(sum(remaining), 1.0, abs_tol=0.01), f"Renormalize weights sum={sum(remaining)}"


def test_stack_chronos_fail_no_double_xgb():
    step04 = _load_pipeline_step("04_predict_48h")
    keys = list(CALIBRATED_ENSEMBLE_WEIGHTS.keys())
    preds = _dummy_preds(keys)
    idx = list(preds.values())[0].index
    ens_ok, method_ok, meta_ok = step04.stack_predictions(preds, idx, chronos_ok=True)
    ens_fail, method_fail, meta_fail = step04.stack_predictions(preds, idx, chronos_ok=False)
    w_xgb_ok = meta_ok.get("meta_w_xgb", 0) or 0
    w_xgb_fail = meta_fail.get("meta_w_xgb", 0) or 0
    if w_xgb_fail > 0:
        assert w_xgb_fail > w_xgb_ok


# ── 4. bias_weekend_scale ────────────────────────────────────────────────────

def _apply_bias(preds, is_t2, day_types):
    bias_arr = np.where(is_t2, ENSEMBLE_BIAS_CORRECTION_T2_MWH, ENSEMBLE_BIAS_CORRECTION_T1_MWH)
    is_sunday = np.array([dt == "pazar" for dt in day_types])
    is_weekend = np.array([dt in ("cumartesi", "pazar") for dt in day_types])
    scale_t1 = np.where(is_sunday, ENSEMBLE_BIAS_SUNDAY_SCALE_T1,
                        np.where(is_weekend, ENSEMBLE_BIAS_WEEKEND_SCALE_T1, 1.0))
    scale_t2 = np.where(is_sunday, ENSEMBLE_BIAS_SUNDAY_SCALE_T2,
                        np.where(is_weekend, ENSEMBLE_BIAS_WEEKEND_SCALE_T2, 1.0))
    bias_scale = np.where(is_t2, scale_t2, scale_t1)
    return bias_arr * bias_scale


def test_bias_weekday_full():
    bias = _apply_bias(np.ones(1), [False], ["hafta_ici"])
    assert math.isclose(bias[0], ENSEMBLE_BIAS_CORRECTION_T1_MWH, abs_tol=0.01)


def test_bias_weekday_t2():
    bias = _apply_bias(np.ones(1), [True], ["hafta_ici"])
    assert math.isclose(bias[0], ENSEMBLE_BIAS_CORRECTION_T2_MWH, abs_tol=0.01)


def test_bias_saturday_t1():
    bias = _apply_bias(np.ones(1), [False], ["cumartesi"])
    expected = ENSEMBLE_BIAS_CORRECTION_T1_MWH * ENSEMBLE_BIAS_WEEKEND_SCALE_T1
    assert math.isclose(bias[0], expected, abs_tol=0.01)


def test_bias_saturday_t2():
    bias = _apply_bias(np.ones(1), [True], ["cumartesi"])
    expected = ENSEMBLE_BIAS_CORRECTION_T2_MWH * ENSEMBLE_BIAS_WEEKEND_SCALE_T2
    assert math.isclose(bias[0], expected, abs_tol=0.01)


def test_bias_sunday_t1():
    bias = _apply_bias(np.ones(1), [False], ["pazar"])
    expected = ENSEMBLE_BIAS_CORRECTION_T1_MWH * ENSEMBLE_BIAS_SUNDAY_SCALE_T1
    assert math.isclose(bias[0], expected, abs_tol=0.01)


def test_bias_sunday_t2():
    bias = _apply_bias(np.ones(1), [True], ["pazar"])
    expected = ENSEMBLE_BIAS_CORRECTION_T2_MWH * ENSEMBLE_BIAS_SUNDAY_SCALE_T2
    assert math.isclose(bias[0], expected, abs_tol=0.01)


# ── 5. ingest_csv_decimal_comma ─────────────────────────────────────────────

def test_ingest_csv_decimal_comma(tmp_path):
    csv_content = (
        "Asset Id;Starts dd.mm.YYYY HH:MM;Time zone;Energy MWh\n"
        "DemandaBereket_Aydem;29.06.2026 00:00;Europe/Istanbul;1406,877983\n"
        "DemandaBereket_Aydem;29.06.2026 01:00;Europe/Istanbul;1378,221054\n"
    )
    csv_file = tmp_path / "test_aydem.csv"
    csv_file.write_text(csv_content, encoding="utf-8")
    step01 = _load_pipeline_step("01_ingest_actual")
    df = step01.load_source_csv(csv_file)
    assert len(df) == 2
    assert "ADM_Dağıtılan_Enerji_(MWh)" in df.columns
    assert math.isclose(df["ADM_Dağıtılan_Enerji_(MWh)"].iloc[0], 1406.877983, abs_tol=0.001)
    assert df["Saat"].iloc[0] == 0 and df["Saat"].iloc[1] == 1


def test_ingest_csv_decimal_dot(tmp_path):
    csv_content = (
        "Asset Id;Starts dd.mm.YYYY HH:MM;Time zone;Energy MWh\n"
        "DemandaBereket_Aydem;29.06.2026 00:00;Europe/Istanbul;1406.88\n"
    )
    csv_file = tmp_path / "test_aydem_dot.csv"
    csv_file.write_text(csv_content, encoding="utf-8")
    step01 = _load_pipeline_step("01_ingest_actual")
    df = step01.load_source_csv(csv_file)
    assert math.isclose(df["ADM_Dağıtılan_Enerji_(MWh)"].iloc[0], 1406.88, abs_tol=0.01)


# ── 6. SMOKE: config integrity + recency_weight ────────────────────────────

def test_config_weight_sum():
    total = sum(CALIBRATED_ENSEMBLE_WEIGHTS.values())
    assert math.isclose(total, 1.0, abs_tol=0.01)


def test_config_bias_non_negative():
    assert ENSEMBLE_BIAS_CORRECTION_T1_MWH >= 0
    assert ENSEMBLE_BIAS_CORRECTION_T2_MWH >= 0
    assert 0 <= ENSEMBLE_BIAS_WEEKEND_SCALE_T1 <= 1
    assert 0 <= ENSEMBLE_BIAS_WEEKEND_SCALE_T2 <= 1
    assert 0 <= ENSEMBLE_BIAS_SUNDAY_SCALE_T1 <= 1
    assert 0 <= ENSEMBLE_BIAS_SUNDAY_SCALE_T2 <= 1


def test_recency_weight_shape():
    dates = pd.date_range("2025-07-01", periods=100, freq="h")
    sw = recency_sample_weight(dates)
    assert sw is not None
    assert len(sw) == 100
    assert sw[0] < sw[-1], f"En eski ornek en dusuk agirlik: {sw[0]} < {sw[-1]}"
    assert sw[-1] == 1.0, f"En yeni ornek weight=1: {sw[-1]}"
    assert all(sw >= 0) and all(sw <= 1), f"Weight [0,1] araliginda: min={sw.min()}, max={sw.max()}"


def test_recency_weight_none_when_disabled(monkeypatch):
    monkeypatch.setattr("config_live.ENABLE_RECENCY_WEIGHTING", False)
    dates = pd.date_range("2025-07-01", periods=10, freq="h")
    sw = recency_sample_weight(dates)
    assert sw is None


# ── 7. Smart dropna patch (03_build_features) ────────────────────────────

def _load_pipeline_core():
    """03_build_features modulunu importlib ile yukle."""
    path = ROOT / "pipeline" / "03_build_features.py"
    spec = importlib.util.spec_from_file_location("03_build_features", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smart_dropna_preserves_forecast():
    """_smart_dropna_patch target=NaN forecast row'lari (NaN feature olsa bile) korur."""
    mod = _load_pipeline_core()
    df = pd.DataFrame({
        "ADM_Dağıtılan_Enerji_(MWh)": [100.0, 200.0, np.nan, np.nan],
        "feat_a": [1.0, 2.0, 3.0, np.nan],
        "feat_b": [1.0, np.nan, 3.0, 4.0],
    })
    result = mod._smart_dropna_patch(df)
    # Row 1 (target=200, feat_b=NaN): training NaN row → silinir
    # Row 0 (target=100): clean training → kalir
    # Row 2 (target=NaN, feat_b=3): forecast → kalir
    # Row 3 (target=NaN, feat_a=NaN): forecast → kalir (NaN feature korunur)
    assert len(result) == 3, f"1 training + 2 forecast = 3: {len(result)}"
    assert result["ADM_Dağıtılan_Enerji_(MWh)"].isna().sum() == 2, "2 forecast row"


def test_smart_dropna_cleans_training():
    """Training row'lardaki NaN feature'lar temizlenmeli, forecast row'lar korunmali."""
    mod = _load_pipeline_core()
    df = pd.DataFrame({
        "ADM_Dağıtılan_Enerji_(MWh)": [100.0, 200.0, 300.0, np.nan],
        "feat_a": [1.0, np.nan, 3.0, np.nan],
    })
    result = mod._smart_dropna_patch(df)
    # Row 1 (target=200, feat_a=NaN): training NaN feature → silinir
    # Row 0/2 (clean training) + Row 3 (forecast, feat=NaN: korunur) = 3
    assert len(result) == 3, f"2 training + 1 forecast = 3: {len(result)}"
    training_result = result[result["ADM_Dağıtılan_Enerji_(MWh)"].notna()]
    assert training_result["feat_a"].isna().sum() == 0, "Training NaN temizlenmeli"


def test_smart_dropna_inplace():
    """inplace=True modu orijinal df'i degistirmeli."""
    mod = _load_pipeline_core()
    df = pd.DataFrame({
        "ADM_Dağıtılan_Enerji_(MWh)": [100.0, 200.0, np.nan],
        "feat_a": [1.0, np.nan, 3.0],
    })
    result = mod._smart_dropna_patch(df, inplace=True)
    assert result is None
    # Row 0 (clean training) + Row 2 (forecast) = 2
    assert len(df) == 2, f"inplace sonrasi df 2 satir: {len(df)}"