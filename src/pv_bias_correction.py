# -*- coding: utf-8 -*-
"""
PV Embedded-Generation Bias Corrector
======================================
Aydem dağıtım bölgesindeki gömülü PV üretiminden kaynaklanan öğlen saati
(10-14) sistematik aşırı tahmin hatasını OOF rezidüel lookup tablosuyla
düzeltir.

Hipotez:
    GHI yüksek olduğunda model, net yükü olduğundan yüksek tahmin eder çünkü
    GHI → sıcaklık artışı → AC yükü (pozitif sinyal) ile
    GHI → gömülü PV üretimi → net yük düşüşü (negatif sinyal) sinyallerini
    ayırt edemiyor.

Kullanım:
    corrector = PVBiasCorrector()
    corrector.fit(df_oof, actual_col='Gerçek_Tüketim', pred_col='Tahmin_Edilen',
                  ghi_col='GHI_ADM_Weighted')
    corrector.save('pv_bias_lookup.json')

    # Uygulama (post-process):
    corrector = PVBiasCorrector.load('pv_bias_lookup.json')
    corrected_pred = corrector.transform(pred_series, ghi_series)
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sadece bu saatler için düzeltme uygulanır (güneş penceresi)
SOLAR_HOURS = tuple(range(10, 15))  # 10, 11, 12, 13, 14

# Lookup boyutları
N_GHI_QUANTILES = 4   # GHI quartile: 0, 1, 2, 3


class PVBiasCorrector:
    """
    month × hour × GHI_quartile lookup tablosu ile bias düzeltmesi.

    fit() → OOF rezidüellerinden ortalama bias hesapla.
    transform() → Sadece solar_hours için düzeltme uygula.
    """

    def __init__(self, solar_hours: tuple[int, ...] = SOLAR_HOURS,
                 n_ghi_quantiles: int = N_GHI_QUANTILES) -> None:
        self.solar_hours = tuple(solar_hours)
        self.n_ghi_quantiles = n_ghi_quantiles
        # lookup: (month, hour, ghi_q) → mean_residual
        self._lookup: dict[tuple[int, int, int], float] = {}
        # GHI quartile boundaries per (month, hour) for inference-time binning
        self._ghi_boundaries: dict[tuple[int, int], list[float]] = {}
        # Sample counts per cell (for trust assessment)
        self._counts: dict[tuple[int, int, int], int] = {}
        # Global fallback per (month, hour)
        self._fallback: dict[tuple[int, int], float] = {}
        self.is_fitted = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        actual_col: str,
        pred_col: str,
        ghi_col: str,
        min_samples_per_cell: int = 5,
    ) -> "PVBiasCorrector":
        """
        OOF veri çerçevesinden bias lookup tablosunu oluşturur.

        Parameters
        ----------
        df : DataFrame with DatetimeIndex, containing actual, predicted and GHI.
        actual_col, pred_col, ghi_col : column names
        min_samples_per_cell : hücrede bu kadar örnek yoksa fallback kullan
        """
        df = df.copy()
        df["_month"] = df.index.month
        df["_hour"] = df.index.hour
        df["_residual"] = df[actual_col].astype(float) - df[pred_col].astype(float)

        # Sadece güneş penceresinde çalış
        solar_mask = df["_hour"].isin(self.solar_hours)
        dfs = df[solar_mask].copy()

        if dfs.empty:
            raise ValueError(f"Güneş saatlerinde ({self.solar_hours}) hiç veri yok.")

        # GHI quartile boundaries: her (month, hour) için ayrı
        for mo in range(1, 13):
            for hr in self.solar_hours:
                sel = dfs[(dfs["_month"] == mo) & (dfs["_hour"] == hr)][ghi_col].dropna()
                if len(sel) < 4:
                    # Yeterli veri yoksa, ay bazında genel sınırları kullan
                    sel_mo = dfs[dfs["_month"] == mo][ghi_col].dropna()
                    if len(sel_mo) < 4:
                        sel_mo = dfs[ghi_col].dropna()
                    boundaries = list(np.quantile(sel_mo, [0.25, 0.5, 0.75]))
                else:
                    boundaries = list(np.quantile(sel, [0.25, 0.5, 0.75]))
                self._ghi_boundaries[(mo, hr)] = boundaries

        # GHI quartile assignment
        def _assign_q(row: pd.Series) -> int:
            key = (int(row["_month"]), int(row["_hour"]))
            bounds = self._ghi_boundaries.get(key, [200.0, 400.0, 600.0])
            g = float(row[ghi_col]) if pd.notna(row[ghi_col]) else 0.0
            if g <= bounds[0]:
                return 0
            elif g <= bounds[1]:
                return 1
            elif g <= bounds[2]:
                return 2
            else:
                return 3

        dfs["_ghi_q"] = dfs.apply(_assign_q, axis=1)

        # Mean residual per cell
        grouped = dfs.groupby(["_month", "_hour", "_ghi_q"])["_residual"]
        for (mo, hr, q), grp in grouped:
            vals = grp.dropna().values
            n = len(vals)
            self._counts[(mo, hr, q)] = n
            if n >= min_samples_per_cell:
                self._lookup[(mo, hr, q)] = float(np.mean(vals))

        # Fallback: month × hour mean (ignoring quartile)
        for (mo, hr), grp in dfs.groupby(["_month", "_hour"])["_residual"]:
            vals = grp.dropna().values
            if len(vals) >= 2:
                self._fallback[(mo, hr)] = float(np.mean(vals))

        self.is_fitted = True
        logger.info(
            "[PVBias] fit complete — %d cells populated (min_samples=%d)",
            len(self._lookup), min_samples_per_cell,
        )
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def transform(
        self,
        pred: pd.Series,
        ghi: pd.Series,
        index: pd.DatetimeIndex | None = None,
    ) -> pd.Series:
        """
        Tahmin serisine bias düzeltmesi uygular.
        Sadece SOLAR_HOURS saatlerinde ve pozitif GHI değerlerinde aktif olur.

        Returns a Series with the same index as pred.
        """
        if not self.is_fitted:
            raise RuntimeError("PVBiasCorrector.fit() henüz çağrılmadı.")

        pred = pred.copy()
        idx = pred.index if index is None else index
        corrected = pred.values.copy().astype(float)

        for i, ts in enumerate(idx):
            ts = pd.Timestamp(ts)
            hr = ts.hour
            if hr not in self.solar_hours:
                continue
            mo = ts.month
            g = float(ghi.iloc[i]) if i < len(ghi) and pd.notna(ghi.iloc[i]) else 0.0
            if g <= 0:
                continue  # gece / sıfır GHI → düzeltme yok

            # GHI quartile
            bounds = self._ghi_boundaries.get((mo, hr), [200.0, 400.0, 600.0])
            if g <= bounds[0]:
                q = 0
            elif g <= bounds[1]:
                q = 1
            elif g <= bounds[2]:
                q = 2
            else:
                q = 3

            correction = self._lookup.get((mo, hr, q))
            if correction is None:
                correction = self._fallback.get((mo, hr), 0.0)

            corrected[i] += correction

        return pd.Series(corrected, index=pred.index, name=pred.name)

    # ------------------------------------------------------------------
    # Lookup quality summary
    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """Hücre başına örnek sayısı ve bias değeri tablosu."""
        rows = []
        for mo in range(1, 13):
            for hr in self.solar_hours:
                for q in range(self.n_ghi_quantiles):
                    rows.append({
                        "month": mo,
                        "hour": hr,
                        "ghi_quartile": q,
                        "n_samples": self._counts.get((mo, hr, q), 0),
                        "mean_residual": self._lookup.get((mo, hr, q), float("nan")),
                        "fallback": (mo, hr, q) not in self._lookup,
                    })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        payload: dict[str, Any] = {
            "solar_hours": list(self.solar_hours),
            "n_ghi_quantiles": self.n_ghi_quantiles,
            # JSON keys must be strings
            "lookup": {
                f"{mo}_{hr}_{q}": v
                for (mo, hr, q), v in self._lookup.items()
            },
            "ghi_boundaries": {
                f"{mo}_{hr}": bounds
                for (mo, hr), bounds in self._ghi_boundaries.items()
            },
            "counts": {
                f"{mo}_{hr}_{q}": n
                for (mo, hr, q), n in self._counts.items()
            },
            "fallback": {
                f"{mo}_{hr}": v
                for (mo, hr), v in self._fallback.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("[PVBias] Lookup kaydedildi: %s", path)

    @classmethod
    def load(cls, path: str) -> "PVBiasCorrector":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        solar_hours = tuple(payload.get("solar_hours", SOLAR_HOURS))
        n_quantiles = int(payload.get("n_ghi_quantiles", N_GHI_QUANTILES))
        obj = cls(solar_hours=solar_hours, n_ghi_quantiles=n_quantiles)
        for k, v in payload["lookup"].items():
            mo, hr, q = (int(x) for x in k.split("_"))
            obj._lookup[(mo, hr, q)] = v
        for k, v in payload["ghi_boundaries"].items():
            mo, hr = (int(x) for x in k.split("_"))
            obj._ghi_boundaries[(mo, hr)] = v
        for k, v in payload["counts"].items():
            mo, hr, q = (int(x) for x in k.split("_"))
            obj._counts[(mo, hr, q)] = int(v)
        for k, v in payload["fallback"].items():
            mo, hr = (int(x) for x in k.split("_"))
            obj._fallback[(mo, hr)] = v
        obj.is_fitted = True
        return obj
