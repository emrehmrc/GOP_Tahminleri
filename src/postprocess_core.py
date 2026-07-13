"""
postprocess_core.py — 05_postprocess.py ortak çekirdeği (Faz 3, 2026-07-13)

Ensemble bias-correction bloğu (gün-tipi duyarlı, T+1/T+2 ayrı ölçek) ADM ve
GDZ'de zaten satır satır aynıydı (GDZ'ye 2026-07-10'da ADM'den port edilmişti,
bkz. gdz talep/live/pipeline/05_postprocess.py docstring). Bu modül o tek
ortak bloğu barındırır.

Holiday-substitution ve PV-bias-correction katmanları KASITLI OLARAK buraya
taşınMADI — bunlar sadece ADM'ye özgü (GDZ'nin kendi eski bias_correction.py
katmanları A/B testte kötüleşince kapatıldı, bkz. docs/MASTER_PLAN.md Faz 3),
zorla ortaklaştırmak GDZ'ye hiç var olmayan bir davranış dayatmak olurdu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_ensemble_bias_correction(
    preds: pd.Series,
    horizon_day: np.ndarray,
    day_type: np.ndarray,
    *,
    t1_mwh: float,
    t2_mwh: float,
    weekend_scale_t1: float,
    weekend_scale_t2: float,
    sunday_scale_t1: float,
    sunday_scale_t2: float,
) -> tuple[pd.Series, np.ndarray, float]:
    """Sistematik under-estimation karşıtı, gün-tipi duyarlı ensemble bias.

    Pazar < Cumartesi < Hafta içi bias scaling (hafta sonu yükleri düşük,
    bias over-correction riski var).

    Returns:
        (corrected_preds, bias_arr, bias_total)
    """
    is_t2 = horizon_day == "T+2"
    bias_arr = np.where(is_t2, t2_mwh, t1_mwh)

    is_sunday = day_type == "pazar"
    is_weekend = (day_type == "cumartesi") | is_sunday

    scale_t1 = np.where(is_sunday, sunday_scale_t1,
                        np.where(is_weekend, weekend_scale_t1, 1.0))
    scale_t2 = np.where(is_sunday, sunday_scale_t2,
                        np.where(is_weekend, weekend_scale_t2, 1.0))
    bias_scale = np.where(is_t2, scale_t2, scale_t1)

    bias_arr = (bias_arr * bias_scale).astype(float)
    corrected = preds + bias_arr
    bias_total = float(bias_arr.sum())
    return corrected, bias_arr, bias_total
