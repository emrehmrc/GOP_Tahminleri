"""Faz 2 2c-1/2c-2 (2026-07-13) — src/oof_feedback.py testleri.

1. get_inverse_mape_weights: chronos_fallback karantinasi gevsetildi -- artik
   sadece CHRONOS_Pred'in kendi MAPE hesabindan o gunler cikariliyor, XGB/LGBM/
   CAT o gunun OOF'unu hala kullanabiliyor (eskiden TUM satir dusuyordu).
2. get_segment_weights: hour_block x day_type_group kirilimi (scaffolding,
   henuz canliya baglanmadi).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oof_feedback import get_inverse_mape_weights, get_segment_weights


def _make_oof(n_days: int = 20, chronos_fallback_days: set[str] | None = None) -> pd.DataFrame:
    chronos_fallback_days = chronos_fallback_days or set()
    rows = []
    dates = pd.date_range("2026-06-01", periods=n_days, freq="D")
    rng = np.random.default_rng(42)
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        is_fallback = ds in chronos_fallback_days
        for h in range(24):
            actual = 1000.0 + h * 10
            xgb = actual + rng.normal(0, 5)     # iyi model
            lgbm = actual + rng.normal(0, 50)    # orta model
            chronos = xgb if is_fallback else actual + rng.normal(0, 200)  # fallback -> XGB kopyasi
            rows.append({
                "date": ds, "hour": h, "actual": actual,
                "XGB_Pred": xgb, "LGBM_Pred": lgbm, "CHRONOS_Pred": chronos,
                "chronos_fallback": is_fallback,
            })
    return pd.DataFrame(rows)


def test_inverse_mape_keeps_other_models_on_fallback_days(tmp_path):
    """Eskiden chronos_fallback gunun TUMU (XGB/LGBM dahil) egitimden dusuyordu.
    Artik sadece CHRONOS_Pred etkileniyor -- diger modellerin ornek sayisi
    fallback-gunsuz senaryoyla AYNI kalmali."""
    oof_path = tmp_path / "oof_history.parquet"

    clean = _make_oof(n_days=20, chronos_fallback_days=set())
    clean.to_parquet(oof_path, index=False)
    w_clean = get_inverse_mape_weights(oof_path, ["XGB_Pred", "LGBM_Pred", "CHRONOS_Pred"], min_days=14)

    with_fallback = _make_oof(n_days=20, chronos_fallback_days={"2026-06-05", "2026-06-10"})
    with_fallback.to_parquet(oof_path, index=False)
    w_fallback = get_inverse_mape_weights(oof_path, ["XGB_Pred", "LGBM_Pred", "CHRONOS_Pred"], min_days=14)

    assert w_clean is not None and w_fallback is not None
    # XGB agirligi fallback gunleri dahil olsa da hesaba katildigi icin
    # (whole-row-drop olsaydi ornek sayisi azalip agirlik daha oynak olurdu)
    # cok yakin kalmali.
    assert abs(w_clean["XGB_Pred"] - w_fallback["XGB_Pred"]) < 0.05
    # CHRONOS agirligi fallback kirliligi disariya cikarildigi icin (XGB kopyasi
    # -- yapay dusuk MAPE) sise/bozulmamali: fallback'siz senaryoya yakin kalmali.
    assert abs(w_clean["CHRONOS_Pred"] - w_fallback["CHRONOS_Pred"]) < 0.1


def test_inverse_mape_none_without_enough_days(tmp_path):
    oof_path = tmp_path / "oof_history.parquet"
    _make_oof(n_days=5).to_parquet(oof_path, index=False)
    assert get_inverse_mape_weights(oof_path, ["XGB_Pred", "LGBM_Pred"], min_days=14) is None


def test_segment_weights_returns_per_segment_dict(tmp_path):
    oof_path = tmp_path / "oof_history.parquet"
    _make_oof(n_days=35).to_parquet(oof_path, index=False)

    result = get_segment_weights(oof_path, ["XGB_Pred", "LGBM_Pred", "CHRONOS_Pred"],
                                  lookback_days=30, min_samples_per_segment=20)
    assert result is not None
    assert len(result) > 0
    for (hour_block, day_type_group), weights in result.items():
        assert hour_block in {"night", "morning", "pv", "evening"}
        assert day_type_group in {"hafta_ici", "cumartesi", "pazar", "ozel_gun"}
        assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_segment_weights_none_without_data(tmp_path):
    oof_path = tmp_path / "oof_history.parquet"
    assert get_segment_weights(oof_path, ["XGB_Pred", "LGBM_Pred"]) is None

    _make_oof(n_days=2).to_parquet(oof_path, index=False)
    assert get_segment_weights(oof_path, ["XGB_Pred", "LGBM_Pred"], min_samples_per_segment=30) is None
