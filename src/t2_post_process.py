"""
T+2-specific post-processing (Fix 3.2).

Approach: For the worst T+2 categories, train a lightweight Ridge model
on historical data to predict T+1->T+2 correction.

Input features: T+1 prediction, day-of-week, holiday distance, temperature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


def train_t2_correction_model(
    df: pd.DataFrame,
    pred_col: str = "Ensemble_Pred",
    actual_col: str = "Actual",
    t2_col: str = "is_t2",
    temp_col: str | None = None,
) -> tuple[object, object, list[str]]:
    """
    Train a Ridge model that predicts T+2 correction factor:
        correction = actual / prediction (for T+2 rows only)

    Returns (model, scaler, feature_cols)
    """
    t2_mask = (df.get(t2_col, pd.Series(0, index=df.index)) == 1)
    t2_df = df.loc[t2_mask].copy()
    if len(t2_df) < 100:
        return None, None, []

    t2_df["correction"] = t2_df[actual_col] / (t2_df[pred_col].abs() + 1.0)
    t2_df["correction"] = t2_df["correction"].clip(0.5, 2.0)

    features = pd.DataFrame(index=t2_df.index)
    features["t1_pred"] = t2_df[pred_col]

    if isinstance(t2_df.index, pd.DatetimeIndex):
        features["dow"] = t2_df.index.dayofweek.astype(float)
        features["hour"] = t2_df.index.hour.astype(float)
    else:
        features["dow"] = 0.0
        features["hour"] = 0.0

    if "days_since_holiday_end" in t2_df.columns:
        features["days_since_holiday"] = t2_df["days_since_holiday_end"].fillna(-1).clip(-1, 30)
    else:
        features["days_since_holiday"] = -1.0

    if temp_col and temp_col in t2_df.columns:
        features["temp"] = t2_df[temp_col].fillna(20.0)
    elif "Temp_Avg" in t2_df.columns:
        features["temp"] = t2_df["Temp_Avg"].fillna(20.0)
    else:
        features["temp"] = 20.0

    feature_cols = list(features.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    y = t2_df["correction"].values

    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    model.fit(X_scaled, y)

    return model, scaler, feature_cols


def apply_t2_correction(
    preds: pd.Series,
    context_df: pd.DataFrame,
    model: object,
    scaler: object,
    feature_cols: list[str],
    t2_col: str = "is_t2",
    temp_col: str | None = None,
) -> pd.Series:
    """
    Apply T+2 correction to predictions. T+1 hours are unchanged.
    """
    if model is None or scaler is None:
        return preds

    t2_mask = (context_df.get(t2_col, pd.Series(0, index=context_df.index)) == 1)
    if not t2_mask.any():
        return preds

    t2_idx = context_df.index[t2_mask.values]
    features = pd.DataFrame(index=t2_idx)

    features["t1_pred"] = preds.loc[t2_idx].values

    if isinstance(t2_idx, pd.DatetimeIndex):
        features["dow"] = t2_idx.dayofweek.astype(float)
        features["hour"] = t2_idx.hour.astype(float)
    else:
        features["dow"] = 0.0
        features["hour"] = 0.0

    if "days_since_holiday_end" in context_df.columns:
        features["days_since_holiday"] = (
            context_df.loc[t2_idx, "days_since_holiday_end"].fillna(-1).clip(-1, 30).values
        )
    else:
        features["days_since_holiday"] = -1.0

    if temp_col and temp_col in context_df.columns:
        features["temp"] = context_df.loc[t2_idx, temp_col].fillna(20.0).values
    elif "Temp_Avg" in context_df.columns:
        features["temp"] = context_df.loc[t2_idx, "Temp_Avg"].fillna(20.0).values
    else:
        features["temp"] = 20.0

    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0.0

    X_scaled = scaler.transform(features[feature_cols])
    corrections = model.predict(X_scaled)
    corrections = np.clip(corrections, 0.5, 2.0)

    result = preds.copy()
    result.loc[t2_idx] = result.loc[t2_idx].values * corrections
    return result
