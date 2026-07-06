"""recency_weight.py — Eğitim örneklerine yaşa göre üstel azalan ağırlık
=======================================================================
GBDT'ler (XGB/LGBM/CAT) 2.5 yıllık veriyi EŞİT ağırlıkla eğitiyordu; hızlı yük
rampasında (yaz soğutma) model geçmiş düşük seviyeye regularize olup sistematik
düşük-tahmin üretiyordu (oturum teşhisi). Bu yardımcı, en yeni örneği 1.0,
RECENCY_HALFLIFE_DAYS gün öncesini 0.5 ağırlıklayan bir sample_weight üretir.

Kullanım (manager'ların _fit_single_model'inde):
    sw = recency_sample_weight(X_train.index)
    model.fit(X, y, sample_weight=sw, ...)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config_live as C


def recency_sample_weight(index) -> np.ndarray | None:
    """DatetimeIndex → üstel yaş-ağırlığı (en yeni=1.0). Flag kapalıysa veya index
    tarih değilse None döner (fit sample_weight=None ile eşit ağırlığa düşer)."""
    if not getattr(C, "ENABLE_RECENCY_WEIGHTING", False):
        return None
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return None
    halflife = float(getattr(C, "RECENCY_HALFLIFE_DAYS", 60) or 60)
    if halflife <= 0:
        return None
    age_days = (index.max() - index).to_numpy(dtype="timedelta64[s]").astype("float64") / 86400.0
    return np.power(0.5, age_days / halflife)
