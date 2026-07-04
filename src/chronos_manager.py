"""
Chronos-2 (isteğe bağlı LoRA) tek sefer yükleme + predict_df ile kısa horizon tahmin.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd



if TYPE_CHECKING:
    from chronos import Chronos2Pipeline

DEFAULT_MODEL_ID = "amazon/chronos-2"


class ChronosInferenceWrapper:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        adapter_path: Optional[str] = None,
        device_map: Optional[str] = None,
        context_length: int = 1024,
    ):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.context_length = int(context_length)
        self._pipeline = None
        if device_map is None:
            try:
                import torch

                self.device_map = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device_map = "cpu"
        else:
            self.device_map = device_map

    @property
    def pipeline(self) -> "Chronos2Pipeline":
        if self._pipeline is None:
            self._load()
        return self._pipeline

    def _load(self) -> None:
        import torch
        from chronos import Chronos2Pipeline

        # Monkey-patch Chronos freq bug: validate_inputs=False'da assert ediyor
        import chronos.df_utils as _cdu, pandas as _pd
        _orig = _cdu.convert_df_input_to_list_of_dicts_input
        def _fixed(df, *args, **kwargs):
            if not kwargs.get('validate_inputs', True):
                if 'ds' in df.columns:
                    try:
                        df = df.copy()
                        df['ds'] = _pd.DatetimeIndex(df['ds'])
                        if df['ds'].inferred_freq is None:
                            try:
                                df['ds'] = _pd.DatetimeIndex(df['ds'], freq=_pd.infer_freq(df['ds']) or 'h')
                            except Exception:
                                df['ds'] = _pd.DatetimeIndex(df['ds'], freq='h')
                    except Exception:
                        pass
            return _orig(df, *args, **kwargs)
        _cdu.convert_df_input_to_list_of_dicts_input = _fixed

        print(
            f"[Chronos] Yükleniyor: {self.model_id} | cihaz={self.device_map} | "
            f"context_length<={self.context_length}"
        )
        self._pipeline = Chronos2Pipeline.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            torch_dtype=torch.float32,
        )
        if self.adapter_path and os.path.isdir(self.adapter_path):
            cfg = os.path.join(self.adapter_path, "adapter_config.json")
            if os.path.isfile(cfg):
                from peft import PeftModel

                print(f"[Chronos] LoRA adaptör: {self.adapter_path}")
                self._pipeline.model = PeftModel.from_pretrained(
                    self._pipeline.model, self.adapter_path
                )
            else:
                print(f"[Chronos] UYARI: adapter_config.json yok, taban model kullanılıyor: {self.adapter_path}")

    def predict_horizon(
        self,
        context_df: pd.DataFrame,
        future_df: pd.DataFrame,
        prediction_length: int,
        use_covariates: Optional[bool] = None,
    ) -> np.ndarray:
        """Median (0.5) quantile tahmini, shape (prediction_length,).

        Args:
            context_df: Geçmiş zaman serisi (unique_id, ds, y + kovaryan sütunları).
            future_df: Tahmin penceresi future-known kovaryanları (unique_id, ds + kovaryanlar).
            prediction_length: Tahmin edilecek adım sayısı.
            use_covariates: None ise config.CHRONOS_USE_COVARIATES değeri kullanılır.
                True  → future_df kovaryatları Chronos'a geçilir (aktif kovaryat modu).
                False → future_df=None, saf zaman serisi tahmini (ablasyon modu).
        """
        if use_covariates is None:
            try:
                from config_live import CHRONOS_USE_COVARIATES
                use_covariates = CHRONOS_USE_COVARIATES
            except ImportError:
                use_covariates = True  # fallback: mevcut davranışı koru

        _future = future_df if use_covariates else None
        if not use_covariates:
            print("[Chronos] CHRONOS_USE_COVARIATES=False — future_df kovaryatsız (ablasyon modu)")

        p = self.pipeline
        pred_df = p.predict_df(
            context_df,
            future_df=_future,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="unique_id",
            timestamp_column="ds",
            target="y",
            validate_inputs=False,
        )
        if "predictions" in pred_df.columns:
            vals = pred_df["predictions"].values
        else:
            vals = pred_df["0.5"].values
        vals = np.asarray(vals, dtype=np.float64).ravel()
        if len(vals) < prediction_length:
            raise RuntimeError(f"Chronos çıktı uzunluğu {len(vals)} < {prediction_length}")
        return vals[:prediction_length]
