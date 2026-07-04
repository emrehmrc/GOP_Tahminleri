# -*- coding: utf-8 -*-
"""
Leak-safe adaptive stacking + T+2 strategy selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config_live import (
    ADAPTIVE_STRATEGY,
    HOLIDAY_BLEND_ALPHAS_JSON,
    HOLIDAY_BLEND_ALPHAS_T2_JSON,
    ENABLE_HOLIDAY_SUBSTITUTION,
)
from src.metrics import calculate_mape as _calculate_mape


@dataclass
class AdaptiveStackingResult:
    predictions: pd.Series
    mape: float
    accepted: bool
    base_method: str
    final_method: str
    metadata: dict[str, Any]


def _event_window_labels(index: pd.DatetimeIndex) -> pd.Series:
    try:
        from src.holiday_calendar import build_event_window_map
        years_in_data = range(index.year.min() - 1, index.year.max() + 2)
        ew_map = build_event_window_map(years=years_in_data)
        labels = [ew_map.get(ts.date(), "normal") for ts in index]
        return pd.Series(labels, index=index)
    except Exception:
        return pd.Series("normal", index=index)


def _rolling_strategy_ridge(
    actual: pd.Series,
    fold_ids: pd.Series,
    strategy_preds: dict[str, pd.Series],
    base_predictions: pd.Series,
    lookback_days: int = 240,
    alpha: float = 10000,
    min_history_days: int = 30,
):
    from sklearn.linear_model import Ridge

    fold_ids_arr = fold_ids.values
    unique_folds = sorted(fold_ids.unique())
    n_hours_per_fold = int(np.median(fold_ids.value_counts().values))
    min_history_hours = min_history_days * 24

    preds = pd.Series(np.full(len(actual), np.nan), index=actual.index, dtype=float)
    strategy_cols = list(strategy_preds.keys())

    for fid in unique_folds:
        fmask = fold_ids_arr == fid
        current_idx = fold_ids.index[fmask]
        past_mask = fold_ids_arr < fid

        if past_mask.sum() < min_history_hours:
            preds.loc[current_idx] = base_predictions.reindex(current_idx).values
            continue

        past_actual = actual.loc[past_mask]
        past_X = pd.DataFrame({col: strategy_preds[col].loc[past_mask].values for col in strategy_cols})

        if len(past_actual) > lookback_days * 24:
            past_actual = past_actual.iloc[-lookback_days * 24:]
            past_X = past_X.iloc[-lookback_days * 24:]

        # Bazi stratejiler (ozellikle warm-up disi expanding-window kollari) nadiren
        # NaN tahmin uretebiliyor; Ridge NaN kabul etmiyor, bu satirlari at.
        valid_rows = past_X.notna().all(axis=1).values & past_actual.notna().values
        if not valid_rows.all():
            past_actual = past_actual.loc[valid_rows]
            past_X = past_X.loc[valid_rows]

        if len(past_actual) < 48:
            preds.loc[current_idx] = base_predictions.reindex(current_idx).values
            continue

        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(past_X, past_actual)

        current_X = pd.DataFrame({col: strategy_preds[col].loc[current_idx].values for col in strategy_cols})
        preds.loc[current_idx] = ridge.predict(current_X)

    mape = _calculate_mape(actual, preds)
    metadata = {
        "lookback_days": lookback_days,
        "alpha": alpha,
        "min_history_days": min_history_days,
    }
    return preds, metadata


class HourlyAdaptiveSelector:
    def __init__(self, lookback_days=90, min_samples=30):
        self.lookback = lookback_days
        self.min_samples = min_samples
        self.hourly_weights = {}

    def fit(self, strategy_preds, actual, index):
        from sklearn.linear_model import Ridge
        strategy_cols = list(strategy_preds.keys())
        hours = index.hour

        for h in range(24):
            mask = hours == h
            if mask.sum() < self.min_samples:
                continue
            X_h = np.column_stack([strategy_preds[col].values[mask] for col in strategy_cols])
            y_h = actual.values[mask]
            valid_rows = ~np.isnan(X_h).any(axis=1) & ~np.isnan(y_h)
            if valid_rows.sum() < self.min_samples:
                continue
            X_h, y_h = X_h[valid_rows], y_h[valid_rows]
            ridge = Ridge(alpha=100.0)
            ridge.fit(X_h, y_h)
            self.hourly_weights[h] = {col: ridge.coef_[i] for i, col in enumerate(strategy_cols)}

    def predict(self, strategy_preds, index):
        strategy_cols = list(strategy_preds.keys())
        hours = index.hour
        results = np.zeros(len(index))
        for h in range(24):
            mask = hours == h
            if h in self.hourly_weights:
                w = self.hourly_weights[h]
                pred = sum(strategy_preds[col].values[mask] * w[col] for col in strategy_cols)
                results[mask] = pred
            else:
                results[mask] = np.column_stack([strategy_preds[col].values[mask] for col in strategy_cols]).mean(axis=1)
        return pd.Series(results, index=index)


def apply_leak_safe_adaptive_stacking(
    full_year_results: pd.DataFrame,
    all_results: dict,
    dm: Any = None,
    best_predictions: pd.Series | None = None,
    best_mape: float | None = None,
    base_method: str = "Meta_Ridge",
    check_holiday_sub: bool = True,
    min_history_folds: int = 10,
    min_segment_hours: int = 48,
    shrinkage_hours: int = 720,
    improvement_margin_pp: float = 0.01,
    rolling_strategy_lookback_days: int = 240,
    rolling_strategy_alpha: float = 10000,
    rolling_strategy_min_history_days: int = 30,
) -> AdaptiveStackingResult:
    index = full_year_results.index
    actual = full_year_results["Actual"].astype(float)
    fold_ids = full_year_results.get("fold_id")
    if fold_ids is None:
        fold_ids = pd.Series(range(len(full_year_results)), index=index)

    strategy_preds = {name: preds.reindex(index).astype(float) for name, (_, preds) in all_results.items()}

    if best_predictions is None:
        base_predictions = strategy_preds.get(base_method, strategy_preds[list(strategy_preds.keys())[0]])
    else:
        base_predictions = best_predictions.reindex(index).astype(float)
    base_mape_for_method = best_mape if best_mape is not None else _calculate_mape(actual, base_predictions)

    # Event-window switch
    ew_labels = _event_window_labels(index)
    unique_labels = ew_labels.unique()
    event_switch = pd.Series(np.full(len(actual), np.nan), index=index, dtype=float)
    decisions = []

    for label in unique_labels:
        segment_mask = ew_labels == label
        segment_idx = index[segment_mask]

        if segment_mask.sum() < min_segment_hours:
            event_switch.loc[segment_idx] = base_predictions.loc[segment_idx]
            continue

        # Find best strategy for this segment using past folds only
        unique_folds_seg = sorted(fold_ids[segment_mask].unique())
        base_for_fold = base_method
        reason = "below_min_folds"

        for fid in unique_folds_seg:
            seg_fold_mask = segment_mask & (fold_ids == fid)
            if seg_fold_mask.sum() < 24:
                event_switch.loc[seg_fold_mask] = base_predictions.loc[seg_fold_mask]
                continue

            past_mask = fold_ids < fid
            if past_mask.sum() < min_history_folds * 48:
                event_switch.loc[seg_fold_mask] = base_predictions.loc[seg_fold_mask]
                decisions.append({
                    "fold_id": int(fid),
                    "event_window": str(label),
                    "strategy": base_method,
                    "reason": "below_min_folds",
                    "history_hours": int(past_mask.sum()),
                    "base_score": float("nan"),
                    "selected_score": float("nan"),
                })
                continue

            segment_past_mask = segment_mask & past_mask
            n_segment = int(segment_past_mask.sum())

            if n_segment < min_segment_hours:
                event_switch.loc[seg_fold_mask] = base_predictions.loc[seg_fold_mask]
                continue

            segment_test_mask = segment_mask & (fold_ids == fid)

            shrunk_scores = {}
            for name, preds_series in strategy_preds.items():
                seg_preds = preds_series.loc[segment_past_mask]
                seg_actuals = actual.loc[segment_past_mask]
                ok = np.isfinite(seg_preds) & np.isfinite(seg_actuals) & (seg_actuals > 1.0)
                if ok.sum() < 48:
                    shrunk_scores[name] = 999
                    continue
                segment_mape = float(np.mean(np.abs((seg_actuals[ok] - seg_preds[ok]) / seg_actuals[ok])) * 100)
                global_scores = {}
                for gn, gp in strategy_preds.items():
                    gp_past = gp.loc[past_mask]
                    ga_past = actual.loc[past_mask]
                    gok = np.isfinite(gp_past) & np.isfinite(ga_past) & (ga_past > 1.0)
                    global_scores[gn] = float(np.mean(np.abs((ga_past[gok] - gp_past[gok]) / ga_past[gok])) * 100) if gok.sum() >= 48 else 999
                global_mape = global_scores.get(name, 999)
                shrunk_scores[name] = (
                    (n_segment * segment_mape + shrinkage_hours * global_mape)
                    / (n_segment + shrinkage_hours)
                )

            selected = min(shrunk_scores, key=shrunk_scores.get)
            selected_score = shrunk_scores[selected]
            base_score = shrunk_scores.get(base_for_fold, selected_score)
            if (base_score - selected_score) < improvement_margin_pp:
                selected = base_for_fold
                reason = "below_margin"
                selected_score = base_score
            else:
                reason = "segment_history_best"

            event_switch.loc[segment_test_mask] = strategy_preds[selected].loc[segment_test_mask]
            decisions.append({
                "fold_id": int(fid) if isinstance(fid, (int, np.integer)) else str(fid),
                "event_window": str(label),
                "strategy": selected,
                "reason": reason,
                "history_hours": int(n_segment),
                "base_score": round(float(base_score), 6),
                "selected_score": round(float(selected_score), 6),
            })

    event_switch_mape = _calculate_mape(actual, event_switch)
    rolling_preds, rolling_metadata = _rolling_strategy_ridge(
        actual=actual,
        fold_ids=fold_ids,
        strategy_preds=strategy_preds,
        base_predictions=base_predictions,
        lookback_days=rolling_strategy_lookback_days,
        alpha=rolling_strategy_alpha,
        min_history_days=rolling_strategy_min_history_days,
    )
    rolling_mape = _calculate_mape(actual, rolling_preds)

    candidates = {
        "event_window_switch": (event_switch_mape, event_switch),
        "rolling_strategy_ridge": (rolling_mape, rolling_preds),
    }

    if ADAPTIVE_STRATEGY == "hourly_ridge":
        try:
            selector = HourlyAdaptiveSelector()
            selector.fit(strategy_preds, actual, index)
            hourly_preds = selector.predict(strategy_preds, index)
            hourly_mape = _calculate_mape(actual, hourly_preds)
            candidates["hourly_adaptive"] = (hourly_mape, hourly_preds)
        except Exception:
            pass
    selected_policy = min(candidates, key=lambda name: candidates[name][0])
    candidate_mape, adaptive = candidates[selected_policy]
    base_mape = _calculate_mape(actual, base_predictions.reindex(index).astype(float))
    accepted = candidate_mape + 1e-12 < base_mape
    final_method = f"Adaptive_{base_method}" if accepted else base_method

    if not accepted:
        adaptive = base_predictions.reindex(index).astype(float).copy()
        final_mape = base_mape
    else:
        final_mape = candidate_mape

    usage = pd.Series([d.get("strategy", base_method) for d in decisions]).value_counts().to_dict()
    metadata = {
        "type": "leak_safe_adaptive_policy",
        "base_method": base_method,
        "final_method": final_method,
        "accepted": accepted,
        "base_mape": round(float(base_mape), 6),
        "candidate_mape": round(float(candidate_mape), 6),
        "selected_policy": selected_policy,
        "event_window_switch_mape": round(float(event_switch_mape), 6),
        "rolling_strategy_ridge_mape": round(float(rolling_mape), 6),
        "final_mape": round(float(final_mape), 6),
        "improvement_pp": round(float(base_mape - candidate_mape), 6),
        "min_history_folds": min_history_folds,
        "min_segment_hours": min_segment_hours,
        "shrinkage_hours": shrinkage_hours,
        "improvement_margin_pp": improvement_margin_pp,
        "strategy_usage": {str(k): int(v) for k, v in usage.items()},
        "event_switch_decisions_tail": decisions[-25:],
        "rolling_strategy_ridge": rolling_metadata,
    }
    return AdaptiveStackingResult(
        predictions=adaptive,
        mape=final_mape,
        accepted=accepted,
        base_method=base_method,
        final_method=final_method,
        metadata=metadata,
    )


def select_t2_strategy(all_results, actuals, is_t2_mask):
    t2_mape = {}
    for name, (_, preds) in all_results.items():
        # Expanding-window stratejiler (Optimized_Weights, Inverse_MAPE, ...) warm-up
        # fold'larini tahminlerden disarida birakir; is_t2_mask'i sadece preds'in
        # kapsadigi index'e daraltarak hizala.
        common_idx = preds.index.intersection(is_t2_mask.index)
        mask = is_t2_mask.loc[common_idx]
        t2_preds = preds.loc[common_idx].loc[mask].astype(float)
        t2_actuals = actuals.loc[common_idx].loc[mask].astype(float)
        mape = np.mean(np.abs((t2_actuals - t2_preds) / t2_actuals)) * 100
        t2_mape[name] = mape
    # NaN'li adaylar min() karsilastirmasini yaniltir (nan hep False dondugu icin
    # hicbir zaman "yenilmez" ve yanlislikla secilebilir) — bunlari eleyip sec.
    valid_candidates = {k: v for k, v in t2_mape.items() if not np.isnan(v)}
    if not valid_candidates:
        best = next(iter(t2_mape))
    else:
        best = min(valid_candidates, key=valid_candidates.get)
    return best, t2_mape
