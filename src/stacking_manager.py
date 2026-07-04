"""

SORUN NE İDİ?
─────────────
Eski kod, 3 modelin (XGB, LGBM, CAT) CV tahminlerini topluyor, sonra bir 
Ridge meta-model'i BU TAHMİNLER ÜZERİNDE eğitiyordu. Ardından AYNI VERİ 
üzerinde tahmin yapıp "Hibrit MAPE" olarak raporluyordu.

Bu, tren=test sızıntısıdır (data leakage). Model kendi eğitim verisini 
tahmin ettiği için MAPE yapay olarak düşük çıkıyordu.

ÇÖZÜM NE?
─────────
3 farklı sızıntısız (leakage-free) strateji:

1. EXPANDING-WINDOW STACKING (Genişleyen Pencere)
   ├── İlk 10 fold: Sadece veri topla (meta-model için yeterli veri yok)
   ├── Fold 11: Fold 1-10'daki tahminlerle Ridge eğit → Fold 11'i tahmin et
   ├── Fold 12: Fold 1-11'deki tahminlerle Ridge eğit → Fold 12'yi tahmin et
   └── ...böyle devam eder. Her fold'un tahmini GERÇEKTEN out-of-sample'dır.

2. INVERSE-MAPE WEIGHTING (Ters MAPE Ağırlıklandırma)
   ├── Her fold için, önceki fold'lardaki model bazlı MAPE'leri hesapla
   ├── Hangi model daha düşük MAPE → ona daha fazla ağırlık ver
   ├── Formül: ağırlık = (1/MAPE_model) / toplam(1/MAPE_hepsi)
   └── Örnek: XGB %3, LGBM %2, CAT %2.5 ise → LGBM en çok ağırlık alır

3. CONSTRAINED OPTIMIZATION (Kısıtlı Optimizasyon)
   ├── scipy.optimize ile optimal ağırlıkları bul
   ├── Kısıtlar: tüm ağırlıklar ≥ 0 ve toplamları = 1
   └── Hedef fonksiyon: önceki fold'lardaki MAPE'yi minimize et

Sistem 3 stratejiyi + basit ortalamayı karşılaştırır ve EN İYİSİNİ seçer.

STACKING MANAGER

Taban modellerin OOF tahminleri uzerinde birden fazla meta-strateji dener.
CONTEXTUAL RIDGE: Tahminlere ek olarak saat, haftanin gunu, bayram/HDD vb.
(sadece o an bilinen baglam) ile Ridge.

Stratejiler:
  1) Basit ortalama
  2) Expanding-fold Ridge (sadece taban tahminleri)
  3) Ters MAPE agirliklandirma
  4) Kisitli optimizasyon (agirliklar)
  5) Expanding-fold Ridge + baglam ozellikleri
  6) Rolling (kayan pencere) Ridge + baglam
  7) Conditional Ridge (segment-aware interaction features)
  8) Meta-Ridge (expanding over strategies)
"""

from __future__ import annotations

import os
import joblib
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.linear_model import Ridge, RidgeCV
from scipy.optimize import minimize
from src.metrics import calculate_mape
from config_live import ENABLE_SEASONAL_STACKING_INTERACTIONS, ENABLE_HOUR_STACKING_INTERACTIONS

META_MERGE_CANDIDATES = [
    "Ramazan_Bayram", "Kurban_Bayram", "After_Bayram", "Is_Ramadan", "Is_Eve", "Is_Sahur",
    "Yilbasi", "before_yilbasi", "Milli_Bayram", "Is_lockdown", "Secim_Gunu",
    "HDD_Heating_Stress", "CDD_Cooling_Stress", "GHI_ADM_Weighted",
]

HOLIDAY_SEGMENT_SET = frozenset({
    "religious_pre_2", "religious_pre_1",
    "religious_day_1", "religious_day_2", "religious_day_3",
    "official_pre_1", "official_day", "de_facto_bridge",
})

POST_HOLIDAY_SEGMENT_SET = frozenset({
    "religious_post_1", "religious_post_2_3",
    "official_post_1",
})


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour.to_numpy(dtype=np.float64)
    # Gece saatlerini (00:00 - 05:00) aktivite olarak bir önceki güne bağlıyoruz
    shifted_index = index - pd.Timedelta(hours=6)
    dow = shifted_index.dayofweek.to_numpy(dtype=np.float64)
    return pd.DataFrame(
        {
            "meta__hour": hour,
            "meta__dow": dow,
            "meta__month": index.month.to_numpy(dtype=np.float64),
            "meta__is_weekend": (dow >= 5).astype(np.float64),
            "meta__sin_hour": np.sin(2 * np.pi * hour / 24.0),
            "meta__cos_hour": np.cos(2 * np.pi * hour / 24.0),
            "meta__sin_dow": np.sin(2 * np.pi * dow / 7.0),
            "meta__cos_dow": np.cos(2 * np.pi * dow / 7.0),
        },
        index=index,
    )



class SimpleAverageEnsemble:
    def __init__(self, pred_cols):
        self.pred_cols = pred_cols

    def fit(self, df, y=None):
        pass

    def predict(self, df):
        return df[self.pred_cols].mean(axis=1).values


class StaticWeightedEnsemble:
    def __init__(self, pred_cols, weights):
        self.pred_cols = pred_cols
        self.weights = weights  # dict: col -> weight

    def fit(self, df, y=None):
        pass

    def predict(self, df):
        preds = np.zeros(len(df))
        for col in self.pred_cols:
            w = self.weights.get(col, 0.0)
            preds += df[col].values * w
        return preds


class HourlyRidgeEnsemble:
    def __init__(self, pred_cols, alpha=100.0):
        self.pred_cols = pred_cols
        self.alpha = alpha
        self.models = {}

    def fit(self, df, y=None):
        hours = df.index.hour if isinstance(df.index, pd.DatetimeIndex) else df["meta__hour"]
        actuals = df["Actual"].values
        for h in range(24):
            h_mask = hours == h
            if h_mask.sum() > 10:
                ridge = Ridge(alpha=self.alpha, fit_intercept=True)
                ridge.fit(df.loc[h_mask, self.pred_cols].values, actuals[h_mask])
                self.models[h] = ridge
            else:
                self.models[h] = None

    def predict(self, df):
        hours = df.index.hour if isinstance(df.index, pd.DatetimeIndex) else df["meta__hour"]
        preds = np.zeros(len(df))
        for h in range(24):
            h_mask = hours == h
            idx = np.where(h_mask)[0]
            if len(idx) == 0:
                continue
            df_h = df.iloc[idx]
            model = self.models.get(h)
            if model is not None:
                preds[idx] = model.predict(df_h[self.pred_cols].values)
            else:
                preds[idx] = df_h[self.pred_cols].mean(axis=1).values
        return preds


class InteractionRidgeEnsemble:
    def __init__(self, pred_cols, alpha=500.0):
        self.pred_cols = pred_cols
        self.alpha = alpha
        self.model = None

    def _build_features(self, df):
        X = df[self.pred_cols].copy()
        features = [X[col].values for col in self.pred_cols]
        
        cdd = df["CDD_Cooling_Stress"].values if "CDD_Cooling_Stress" in df.columns else np.zeros(len(df))
        hdd = df["HDD_Heating_Stress"].values if "HDD_Heating_Stress" in df.columns else np.zeros(len(df))
        
        for col in self.pred_cols:
            features.append(X[col].values * cdd)
            features.append(X[col].values * hdd)
            
        return np.column_stack(features)

    def fit(self, df, y=None):
        X_feats = self._build_features(df)
        y_val = df["Actual"].values
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model.fit(X_feats, y_val)

    def predict(self, df):
        X_feats = self._build_features(df)
        return self.model.predict(X_feats)


class FWLSRidgeEnsemble:
    def __init__(self, pred_cols, alpha=1000.0):
        self.pred_cols = pred_cols
        self.alpha = alpha
        self.model = None

    def _build_features(self, df):
        X = df[self.pred_cols].copy()
        
        # Normalize continuous features for Ridge stability
        cdd = df["CDD_Cooling_Stress"].values / 20.0 if "CDD_Cooling_Stress" in df.columns else np.zeros(len(df))
        hdd = df["HDD_Heating_Stress"].values / 20.0 if "HDD_Heating_Stress" in df.columns else np.zeros(len(df))
        solar = df["Solar_Shaving_Proxy"].values / 1000.0 if "Solar_Shaving_Proxy" in df.columns else np.zeros(len(df))
        
        sin_hour = df["meta__sin_hour"].values if "meta__sin_hour" in df.columns else np.zeros(len(df))
        cos_hour = df["meta__cos_hour"].values if "meta__cos_hour" in df.columns else np.zeros(len(df))
        is_weekend = df["meta__is_weekend"].values if "meta__is_weekend" in df.columns else np.zeros(len(df))
        
        meta_feats = {
            'sin_hour': sin_hour,
            'cos_hour': cos_hour,
            'is_weekend': is_weekend,
            'cdd': cdd,
            'hdd': hdd,
            'solar': solar
        }
        
        features = []
        for col in self.pred_cols:
            features.append(X[col].values)
            
        # Cross product interactions
        for col in self.pred_cols:
            p_val = X[col].values
            for mf_name, mf_val in meta_feats.items():
                features.append(p_val * mf_val)
                
        return np.column_stack(features)

    def fit(self, df, y=None):
        X_feats = self._build_features(df)
        y_val = df["Actual"].values
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model.fit(X_feats, y_val)

    def predict(self, df):
        X_feats = self._build_features(df)
        return self.model.predict(X_feats)


class LightGBMStackingEnsemble:
    def __init__(self, pred_cols):
        self.pred_cols = pred_cols
        self.model = None

    def _build_features(self, df):
        meta_cols = ['meta__hour', 'meta__dow', 'meta__month', 'meta__is_weekend', 
                     'HDD_Heating_Stress', 'CDD_Cooling_Stress', 'Solar_Shaving_Proxy']
        existing_meta = [c for c in meta_cols if c in df.columns]
        X = df[self.pred_cols + existing_meta].copy()
        
        for col in X.columns:
            if X[col].dtype.name == 'category':
                X[col] = X[col].cat.codes
        return X

    def fit(self, df, y=None):
        import lightgbm as lgb
        X_feats = self._build_features(df)
        y_val = df["Actual"].values
        
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=8,
            min_child_samples=50,
            reg_alpha=10.0,
            reg_lambda=10.0,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        self.model.fit(X_feats, y_val)

    def predict(self, df):
        X_feats = self._build_features(df)
        return self.model.predict(X_feats)


class HourlyFWLSRidgeEnsemble:
    def __init__(self, pred_cols, alpha=100.0):
        self.pred_cols = pred_cols
        self.alpha = alpha
        self.models = {}

    def _build_features(self, df):
        X = df[self.pred_cols].copy()
        
        is_weekend = df["meta__is_weekend"].values if "meta__is_weekend" in df.columns else np.zeros(len(df))
        solar = df["Solar_Shaving_Proxy"].values / 1000.0 if "Solar_Shaving_Proxy" in df.columns else np.zeros(len(df))
        cdd = df["CDD_Cooling_Stress"].values / 20.0 if "CDD_Cooling_Stress" in df.columns else np.zeros(len(df))
        hdd = df["HDD_Heating_Stress"].values / 20.0 if "HDD_Heating_Stress" in df.columns else np.zeros(len(df))
        
        # Sunset hours 15–20 => electrical load transition period
        hour_raw = df.index.hour.values if isinstance(df.index, pd.DatetimeIndex) else np.zeros(len(df))
        is_sunset = ((hour_raw >= 15) & (hour_raw <= 20)).astype(np.float64)
        
        meta_feats = {
            'is_weekend': is_weekend,
            'solar': solar,
            'cdd': cdd,
            'hdd': hdd,
            'sunset': is_sunset,
        }
        
        features = []
        for col in self.pred_cols:
            features.append(X[col].values)
            
        for col in self.pred_cols:
            p_val = X[col].values
            for mf_name, mf_val in meta_feats.items():
                features.append(p_val * mf_val)
                
        return np.column_stack(features)

    def fit(self, df, y=None):
        hours = df.index.hour if isinstance(df.index, pd.DatetimeIndex) else df["meta__hour"]
        actuals = df["Actual"].values
        X_feats = self._build_features(df)
        
        for h in range(24):
            h_mask = hours == h
            if h_mask.sum() > 10:
                model = RidgeCV(
                    alphas=[1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0],
                    fit_intercept=True
                )
                model.fit(X_feats[h_mask], actuals[h_mask])
                self.models[h] = model
            else:
                self.models[h] = None

    def predict(self, df):
        hours = df.index.hour if isinstance(df.index, pd.DatetimeIndex) else df["meta__hour"]
        X_feats = self._build_features(df)
        preds = np.zeros(len(df))
        
        for h in range(24):
            h_mask = hours == h
            idx = np.where(h_mask)[0]
            if len(idx) == 0:
                continue
            model = self.models.get(h)
            if model is not None:
                preds[idx] = model.predict(X_feats[idx])
            else:
                preds[idx] = df.iloc[idx][self.pred_cols].mean(axis=1).values
        return preds



class StackingManager:

    MIN_WARMUP_FOLDS = 10
    DEFAULT_PRED_COLS = ["XGB_Pred", "LGBM_Pred", "CAT_Pred"]
    DEFAULT_ROLLING_LOOKBACK_HOURS = 720

    def __init__(
        self,
        project_root=".",
        pred_cols=None,
        rolling_lookback_hours=None,
        output_dir=None,
    ):
        root = output_dir if output_dir is not None else project_root
        self.PRED_COLS = list(pred_cols) if pred_cols is not None else list(self.DEFAULT_PRED_COLS)
        self.rolling_lookback_hours = (
            int(rolling_lookback_hours)
            if rolling_lookback_hours is not None
            else self.DEFAULT_ROLLING_LOOKBACK_HOURS
        )
        self.model_dir = os.path.join(root, "Model")
        os.makedirs(self.model_dir, exist_ok=True)
        self.meta_model = None
        self.best_method = None
        self.best_weights = None
        self.meta_feature_cols = None
        self.artifact_metadata = {}

    def _ensure_datetime_index(self, full_df: pd.DataFrame) -> pd.DataFrame:
        out = full_df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors="coerce")
        out = out.sort_index()
        if out.index.has_duplicates:
            out = out[~out.index.duplicated(keep="last")]
        return out

    def _build_enriched(self, full_df: pd.DataFrame, context_source: Optional[pd.DataFrame]):
        base = self._ensure_datetime_index(full_df)
        cal = _calendar_features(base.index)
        enriched = pd.concat([base, cal], axis=1)
        merged = []
        if context_source is not None:
            ctx = context_source.copy()
            if not isinstance(ctx.index, pd.DatetimeIndex):
                ctx.index = pd.to_datetime(ctx.index, errors="coerce")
            ctx = ctx.sort_index()
            if ctx.index.has_duplicates:
                ctx = ctx[~ctx.index.duplicated(keep="last")]
            aligned = ctx.reindex(base.index)
            for c in META_MERGE_CANDIDATES:
                if c not in aligned.columns:
                    continue
                enriched[c] = pd.to_numeric(aligned[c], errors="coerce").fillna(0.0).astype(np.float64)
                merged.append(c)
        cal_cols = [c for c in cal.columns if c.startswith("meta__")]
        meta_cols = list(self.PRED_COLS) + cal_cols + merged
        # event_window for segment-aware stacking
        if context_source is not None and "event_window" in context_source.columns:
            ew_series = context_source["event_window"].reindex(base.index)
            enriched["event_window"] = ew_series.fillna("normal").astype(str)
        return enriched, meta_cols

    def _apply_holiday_override(self, df: pd.DataFrame, preds: pd.Series) -> pd.Series:
        if "CAT_Pred" not in self.PRED_COLS or "CAT_Pred" not in df.columns:
            return preds

        # Get holiday flags
        is_holiday_flag = np.zeros(len(df), dtype=bool)
        if "Yilbasi" in df.columns:
            is_holiday_flag |= (df["Yilbasi"] == 1).to_numpy()
        if "Milli_Bayram" in df.columns:
            is_holiday_flag |= (df["Milli_Bayram"] == 1).to_numpy()
        if "Ramazan_Bayram" in df.columns:
            is_holiday_flag |= (df["Ramazan_Bayram"] == 1).to_numpy()
        if "Kurban_Bayram" in df.columns:
            is_holiday_flag |= (df["Kurban_Bayram"] == 1).to_numpy()


        # Get day of week
        if isinstance(df.index, pd.DatetimeIndex):
            dow_arr = df.index.dayofweek.to_numpy()
        else:
            dow_arr = df["meta__dow"].to_numpy()

        is_weekday_holiday = is_holiday_flag & (dow_arr < 5)

        if not is_weekday_holiday.any():
            return preds

        out = preds.copy()
        cat_preds = df["CAT_Pred"].to_numpy()

        # Safely assign using positional indexing via .iloc
        out.iloc[np.where(is_weekday_holiday)[0]] = cat_preds[is_weekday_holiday]
        print(f"  [Override] Overrode {is_weekday_holiday.sum()} weekday holiday hours with CatBoost solo.")
        return out

    def run_ensemble(self, full_df, context_source=None):
        print("\n" + "=" * 60)
        print("  SIZINTISIZ ENSEMBLE KARSILASTIRMASI (+ baglam)")
        print("=" * 60)

        enriched, meta_cols = self._build_enriched(full_df, context_source)
        self.meta_feature_cols = meta_cols
        n_ctx = len(meta_cols) - len(self.PRED_COLS)
        print(f"  [Meta] Taban: {len(self.PRED_COLS)} | Ek baglam ozelligi: {n_ctx}")

        results = {}

        if "fold_id" in enriched.columns:
            fold_ids = sorted(enriched["fold_id"].unique())
            if len(fold_ids) >= self.MIN_WARMUP_FOLDS + 1:
                valid_mask = enriched["fold_id"].isin(fold_ids[self.MIN_WARMUP_FOLDS:])
                valid_idx = enriched.index[valid_mask]
            else:
                valid_idx = enriched.index
        else:
            valid_idx = enriched.index

        simple_preds = enriched[self.PRED_COLS].mean(axis=1)
        simple_mape = calculate_mape(enriched.loc[valid_idx, "Actual"], simple_preds.loc[valid_idx])
        results["Simple_Average"] = (simple_mape, simple_preds)
        print(f"\n  [1] Basit Ortalama MAPE:              %{simple_mape:.4f}")

        stacking_preds, stacking_mape = self._expanding_window_stacking(enriched)
        if stacking_preds is not None:
            results["Stacking_Ridge"] = (stacking_mape, stacking_preds)
            print(f"  [2] Expanding Stacking (pred) MAPE:   %{stacking_mape:.4f}")
        else:
            print("  [2] Expanding Stacking (pred):        ATLANDI (yeterli fold yok)")

        inv_preds, inv_mape = self._inverse_mape_weighting(enriched)
        if inv_preds is not None:
            results["Inverse_MAPE"] = (inv_mape, inv_preds)
            print(f"  [3] Ters MAPE Agirliklandirma MAPE:   %{inv_mape:.4f}")
        else:
            print("  [3] Ters MAPE Agirliklandirma:        ATLANDI (yeterli fold yok)")

        opt_preds, opt_mape = self._constrained_optimization(enriched)
        if opt_preds is not None:
            results["Optimized_Weights"] = (opt_mape, opt_preds)
            print(f"  [4] Kisitli Optimizasyon MAPE:        %{opt_mape:.4f}")
        else:
            print("  [4] Kisitli Optimizasyon:             ATLANDI (yeterli fold yok)")

        ctx_preds, ctx_mape = self._expanding_window_contextual_ridge(enriched, meta_cols)
        if ctx_preds is not None:
            results["Stacking_Ridge_Context"] = (ctx_mape, ctx_preds)
            print(f"  [5] Expanding Ridge + baglam MAPE:    %{ctx_mape:.4f}")
        else:
            print("  [5] Expanding Ridge + baglam:         ATLANDI")

        roll_preds, roll_mape = self._rolling_ridge_contextual(enriched, meta_cols)
        if roll_preds is not None:
            results["Rolling_Ridge_Context"] = (roll_mape, roll_preds)
            lb = self.rolling_lookback_hours
            print(f"  [6] Rolling Ridge + baglam ({lb}h) MAPE: %{roll_mape:.4f}")
        else:
            print("  [6] Rolling Ridge + baglam:           ATLANDI")

        cond_preds, cond_mape, cond_meta_cols = self._rolling_ridge_conditional(enriched, meta_cols, return_meta_cols=True)
        if cond_preds is not None:
            results["Conditional_Ridge"] = (cond_mape, cond_preds)
            lb = self.rolling_lookback_hours
            print(f"  [7] Conditional Ridge ({lb}h) MAPE:      %{cond_mape:.4f}")
        else:
            print("  [7] Conditional Ridge:                ATLANDI")
            cond_meta_cols = meta_cols

        meta_preds, meta_mape = self._expanding_meta_ridge(enriched, results)
        if meta_preds is not None:
            results["Meta_Ridge"] = (meta_mape, meta_preds)
            print(f"  [8] Meta-Ridge (strategies) MAPE:      %{meta_mape:.4f}")
        else:
            print("  [8] Meta-Ridge:                       ATLANDI")

        seg_meta_preds, seg_meta_mape = self._segment_meta_ridge(enriched, results)
        if seg_meta_preds is not None:
            results["Segment_Meta_Ridge"] = (seg_meta_mape, seg_meta_preds)
            print(f"  [9] Segment Meta-Ridge MAPE:           %{seg_meta_mape:.4f}")
        else:
            print("  [9] Segment Meta-Ridge:                ATLANDI (event_window yok)")

        hourly_preds, hourly_mape = self._expanding_window_hourly_ridge(enriched)
        if hourly_preds is not None:
            results["Hourly_Ridge"] = (hourly_mape, hourly_preds)
            print(f"  [10] Hourly Ridge Stacking MAPE:        %{hourly_mape:.4f}")
        else:
            print("  [10] Hourly Ridge Stacking:             ATLANDI")

        int_preds, int_mape = self._expanding_window_interaction_ridge(enriched)
        if int_preds is not None:
            results["Interaction_Ridge"] = (int_mape, int_preds)
            print(f"  [11] Interaction Ridge Stacking MAPE:   %{int_mape:.4f}")
        else:
            print("  [11] Interaction Ridge Stacking:        ATLANDI")

        fwls_preds, fwls_mape = self._expanding_window_fwls_ridge(enriched)
        if fwls_preds is not None:
            results["FWLS_Ridge"] = (fwls_mape, fwls_preds)
            print(f"  [12] FWLS Stacking (Ridge) MAPE:         %{fwls_mape:.4f}")
        else:
            print("  [12] FWLS Stacking (Ridge):              ATLANDI")

        gbdt_preds, gbdt_mape = self._expanding_window_gbdt_stacking(enriched)
        if gbdt_preds is not None:
            results["GBDT_Stacking"] = (gbdt_mape, gbdt_preds)
            print(f"  [13] GBDT Stacking (LightGBM) MAPE:      %{gbdt_mape:.4f}")
        else:
            print("  [13] GBDT Stacking (LightGBM):           ATLANDI")

        hourly_fwls_preds, hourly_fwls_mape = self._expanding_window_hourly_fwls_ridge(enriched)
        if hourly_fwls_preds is not None:
            results["Hourly_FWLS_Ridge"] = (hourly_fwls_mape, hourly_fwls_preds)
            print(f"  [14] Hourly FWLS Stacking MAPE:          %{hourly_fwls_mape:.4f}")
        else:
            print("  [14] Hourly FWLS Stacking:               ATLANDI")

        # Apply holiday override to all strategies in results before selecting the winner
        for name in list(results.keys()):
            mape, preds = results[name]
            overridden_preds = self._apply_holiday_override(enriched, preds)
            new_mape = calculate_mape(enriched["Actual"], overridden_preds)
            results[name] = (new_mape, overridden_preds)

        best_name = min(results, key=lambda k: results[k][0])
        best_mape, best_preds = results[best_name]
        self.best_method = best_name

        print(f"\n  [KAZANAN] {best_name}  (MAPE = %{best_mape:.4f})")
        print("=" * 60 + "\n")

        self._train_final_meta_model(enriched, meta_cols, cond_meta_cols=cond_meta_cols, results=results)

        return best_preds, results

    def _expanding_window_stacking(self, full_df):
        if "fold_id" not in full_df.columns:
            return None, None

        fold_ids = sorted(full_df["fold_id"].unique())
        if len(fold_ids) < self.MIN_WARMUP_FOLDS + 1:
            return None, None

        stacking_parts = []
        for i, fid in enumerate(fold_ids):
            if i < self.MIN_WARMUP_FOLDS:
                continue
            train_mask = full_df["fold_id"].isin(fold_ids[:i])
            test_mask = full_df["fold_id"] == fid
            X_meta_train = full_df.loc[train_mask, self.PRED_COLS]
            y_meta_train = full_df.loc[train_mask, "Actual"]
            X_meta_test = full_df.loc[test_mask, self.PRED_COLS]
            ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
            ridge.fit(X_meta_train, y_meta_train)
            preds = ridge.predict(X_meta_test)
            stacking_parts.append(pd.Series(preds, index=X_meta_test.index))

        all_stacking_preds = pd.concat(stacking_parts)
        valid_idx = all_stacking_preds.index
        mape = calculate_mape(
            full_df.loc[valid_idx, "Actual"],
            all_stacking_preds.loc[valid_idx],
        )
        full_preds = full_df[self.PRED_COLS].mean(axis=1).copy()
        full_preds.loc[valid_idx] = all_stacking_preds.loc[valid_idx]
        return full_preds, mape

    def _expanding_window_hourly_ridge(self, full_df):
        if "fold_id" not in full_df.columns:
            return None, None

        fold_ids = sorted(full_df["fold_id"].unique())
        if len(fold_ids) < self.MIN_WARMUP_FOLDS + 1:
            return None, None

        stacking_parts = []
        for i, fid in enumerate(fold_ids):
            if i < self.MIN_WARMUP_FOLDS:
                continue
            train_mask = full_df["fold_id"].isin(fold_ids[:i])
            test_mask = full_df["fold_id"] == fid
            
            df_train = full_df.loc[train_mask]
            df_test = full_df.loc[test_mask]
            
            model = HourlyRidgeEnsemble(pred_cols=self.PRED_COLS, alpha=100.0)
            model.fit(df_train)
            preds = model.predict(df_test)
            stacking_parts.append(pd.Series(preds, index=df_test.index))

        all_stacking_preds = pd.concat(stacking_parts)
        valid_idx = all_stacking_preds.index
        mape = calculate_mape(
            full_df.loc[valid_idx, "Actual"],
            all_stacking_preds.loc[valid_idx],
        )
        full_preds = full_df[self.PRED_COLS].mean(axis=1).copy()
        full_preds.loc[valid_idx] = all_stacking_preds.loc[valid_idx]
        return full_preds, mape

    def _expanding_window_interaction_ridge(self, full_df):
        if "fold_id" not in full_df.columns:
            return None, None

        fold_ids = sorted(full_df["fold_id"].unique())
        if len(fold_ids) < self.MIN_WARMUP_FOLDS + 1:
            return None, None

        stacking_parts = []
        for i, fid in enumerate(fold_ids):
            if i < self.MIN_WARMUP_FOLDS:
                continue
            train_mask = full_df["fold_id"].isin(fold_ids[:i])
            test_mask = full_df["fold_id"] == fid
            
            df_train = full_df.loc[train_mask]
            df_test = full_df.loc[test_mask]
            
            model = InteractionRidgeEnsemble(pred_cols=self.PRED_COLS, alpha=500.0)
            model.fit(df_train)
            preds = model.predict(df_test)
            stacking_parts.append(pd.Series(preds, index=df_test.index))

        all_stacking_preds = pd.concat(stacking_parts)
        valid_idx = all_stacking_preds.index
        mape = calculate_mape(
            full_df.loc[valid_idx, "Actual"],
            all_stacking_preds.loc[valid_idx],
        )
        full_preds = full_df[self.PRED_COLS].mean(axis=1).copy()
        full_preds.loc[valid_idx] = all_stacking_preds.loc[valid_idx]
        return full_preds, mape

    def _expanding_window_fwls_ridge(self, full_df):
        if "fold_id" not in full_df.columns:
            return None, None

        fold_ids = sorted(full_df["fold_id"].unique())
        if len(fold_ids) < self.MIN_WARMUP_FOLDS + 1:
            return None, None

        stacking_parts = []
        for i, fid in enumerate(fold_ids):
            if i < self.MIN_WARMUP_FOLDS:
                continue
            train_mask = full_df["fold_id"].isin(fold_ids[:i])
            test_mask = full_df["fold_id"] == fid
            
            df_train = full_df.loc[train_mask]
            df_test = full_df.loc[test_mask]
            
            model = FWLSRidgeEnsemble(pred_cols=self.PRED_COLS, alpha=1000.0)
            model.fit(df_train)
            preds = model.predict(df_test)
            stacking_parts.append(pd.Series(preds, index=df_test.index))

        all_stacking_preds = pd.concat(stacking_parts)
        valid_idx = all_stacking_preds.index
        mape = calculate_mape(
            full_df.loc[valid_idx, "Actual"],
            all_stacking_preds.loc[valid_idx],
        )
        full_preds = full_df[self.PRED_COLS].mean(axis=1).copy()
        full_preds.loc[valid_idx] = all_stacking_preds.loc[valid_idx]
        return full_preds, mape

    def _expanding_window_gbdt_stacking(self, full_df):
        if "fold_id" not in full_df.columns:
            return None, None

        fold_ids = sorted(full_df["fold_id"].unique())
        if len(fold_ids) < self.MIN_WARMUP_FOLDS + 1:
            return None, None

        stacking_parts = []
        for i, fid in enumerate(fold_ids):
            if i < self.MIN_WARMUP_FOLDS:
                continue
            train_mask = full_df["fold_id"].isin(fold_ids[:i])
            test_mask = full_df["fold_id"] == fid
            
            df_train = full_df.loc[train_mask]
            df_test = full_df.loc[test_mask]
            
            model = LightGBMStackingEnsemble(pred_cols=self.PRED_COLS)
            model.fit(df_train)
            preds = model.predict(df_test)
            stacking_parts.append(pd.Series(preds, index=df_test.index))

        all_stacking_preds = pd.concat(stacking_parts)
        valid_idx = all_stacking_preds.index
        mape = calculate_mape(
            full_df.loc[valid_idx, "Actual"],
            all_stacking_preds.loc[valid_idx],
        )
        full_preds = full_df[self.PRED_COLS].mean(axis=1).copy()
        full_preds.loc[valid_idx] = all_stacking_preds.loc[valid_idx]
        return full_preds, mape

    def _expanding_window_hourly_fwls_ridge(self, full_df):
        if "fold_id" not in full_df.columns:
            return None, None

        fold_ids = sorted(full_df["fold_id"].unique())
        if len(fold_ids) < self.MIN_WARMUP_FOLDS + 1:
            return None, None

        stacking_parts = []
        for i, fid in enumerate(fold_ids):
            if i < self.MIN_WARMUP_FOLDS:
                continue
            train_mask = full_df["fold_id"].isin(fold_ids[:i])
            test_mask = full_df["fold_id"] == fid
            
            df_train = full_df.loc[train_mask]
            df_test = full_df.loc[test_mask]
            
            model = HourlyFWLSRidgeEnsemble(pred_cols=self.PRED_COLS, alpha=100.0)
            model.fit(df_train)
            preds = model.predict(df_test)
            stacking_parts.append(pd.Series(preds, index=df_test.index))

        all_stacking_preds = pd.concat(stacking_parts)
        valid_idx = all_stacking_preds.index
        mape = calculate_mape(
            full_df.loc[valid_idx, "Actual"],
            all_stacking_preds.loc[valid_idx],
        )
        full_preds = full_df[self.PRED_COLS].mean(axis=1).copy()
        full_preds.loc[valid_idx] = all_stacking_preds.loc[valid_idx]
        return full_preds, mape

    def _expanding_window_contextual_ridge(self, enriched, meta_cols):
        if "fold_id" not in enriched.columns:
            return None, None
        fold_ids = sorted(enriched["fold_id"].unique())
        if len(fold_ids) < self.MIN_WARMUP_FOLDS + 1:
            return None, None

        parts = []
        for i, fid in enumerate(fold_ids):
            if i < self.MIN_WARMUP_FOLDS:
                continue
            train_mask = enriched["fold_id"].isin(fold_ids[:i])
            test_mask = enriched["fold_id"] == fid
            X_tr = enriched.loc[train_mask, meta_cols]
            y_tr = enriched.loc[train_mask, "Actual"]
            X_te = enriched.loc[test_mask, meta_cols]
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0])
            ridge.fit(X_tr, y_tr)
            parts.append(pd.Series(ridge.predict(X_te), index=X_te.index))

        all_p = pd.concat(parts)
        valid_idx = all_p.index
        mape = calculate_mape(enriched.loc[valid_idx, "Actual"], all_p.loc[valid_idx])
        full_preds = enriched[self.PRED_COLS].mean(axis=1).copy()
        full_preds.loc[valid_idx] = all_p.loc[valid_idx]
        return full_preds, mape

    def _rolling_ridge_batch(self, df: pd.DataFrame, meta_cols: list[str], alpha: float = 100.0, use_ridgecv: bool = False):
        n = len(df)
        lb = self.rolling_lookback_hours
        if n <= lb + 1:
            return None, None

        X_all = df[meta_cols].to_numpy(dtype=np.float64)
        y_all = df["Actual"].to_numpy(dtype=np.float64)
        base_mean = df[self.PRED_COLS].mean(axis=1).to_numpy(dtype=np.float64)
        preds = np.zeros(n, dtype=np.float64)
        preds[:lb] = base_mean[:lb]

        dates = df.index.normalize()
        unique_days = dates[lb:].unique()

        for day_ts in unique_days:
            mask_day = dates == day_ts
            day_idx = np.where(mask_day)[0]
            if len(day_idx) == 0 or day_idx[0] < lb:
                continue
            i_first = day_idx[0]
            X_tr = X_all[i_first - lb : i_first]
            y_tr = y_all[i_first - lb : i_first]

            if use_ridgecv:
                ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0])
            else:
                ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_tr, y_tr)
            preds[day_idx] = ridge.predict(X_all[day_idx])

        ser = pd.Series(preds, index=df.index)
        mape = calculate_mape(df["Actual"], ser)
        return ser, mape

    def _rolling_ridge_contextual(self, enriched, meta_cols):
        return self._rolling_ridge_batch(enriched, meta_cols, alpha=100.0)

    def _build_conditional_features(self, enriched: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        df = enriched.copy()
        new_cols = []

        if isinstance(df.index, pd.DatetimeIndex):
            hour_arr = df.index.hour.to_numpy(dtype=np.float64)
        else:
            hour_arr = df["meta__hour"].to_numpy(dtype=np.float64)

        is_night = ((hour_arr >= 0) & (hour_arr <= 5)).astype(np.float64)
        is_morning_ramp = ((hour_arr >= 6) & (hour_arr <= 9)).astype(np.float64)

        holiday_cols = [c for c in ["Ramazan_Bayram", "Kurban_Bayram", "Milli_Bayram",
                                    "Yilbasi", "Is_Eve"] if c in df.columns]
        if holiday_cols:
            is_holiday_any = df[holiday_cols].max(axis=1).clip(0, 1).to_numpy(dtype=np.float64)
        else:
            is_holiday_any = np.zeros(len(df), dtype=np.float64)

        if 'fold_id' in df.columns:
            fold_ids_sorted = sorted(df['fold_id'].unique())
            fold_position = np.zeros(len(df), dtype=int)
            for fid in fold_ids_sorted:
                mask = df['fold_id'] == fid
                fold_position[mask.values] = np.arange(mask.sum())
            is_t2_flag = fold_position >= 24
        else:
            is_t2_flag = (np.arange(len(df)) % 48) >= 24

        is_t2 = is_t2_flag.astype(np.float64)

        for col in self.PRED_COLS:
            short = col.replace("_Pred", "")

            feat_night = f"{short}_x_night"
            df[feat_night] = df[col].to_numpy(dtype=np.float64) * is_night
            new_cols.append(feat_night)

            feat_morning = f"{short}_x_morning_ramp"
            df[feat_morning] = df[col].to_numpy(dtype=np.float64) * is_morning_ramp
            new_cols.append(feat_morning)

            feat_hol = f"{short}_x_holiday"
            df[feat_hol] = df[col].to_numpy(dtype=np.float64) * is_holiday_any
            new_cols.append(feat_hol)

            feat_t2 = f"{short}_x_t2"
            df[feat_t2] = df[col].to_numpy(dtype=np.float64) * is_t2
            new_cols.append(feat_t2)

            feat_t2_hol = f"{short}_x_t2_holiday"
            df[feat_t2_hol] = df[col].to_numpy(dtype=np.float64) * is_t2 * is_holiday_any
            new_cols.append(feat_t2_hol)

        if ENABLE_SEASONAL_STACKING_INTERACTIONS:
            df, seasonal_cols = self._build_seasonal_interactions(df)
            new_cols.extend(seasonal_cols)

        if ENABLE_HOUR_STACKING_INTERACTIONS:
            df, hour_cols = self._build_hour_interaction_features(df)
            new_cols.extend(hour_cols)

        return df, new_cols

    def _build_hour_interaction_features(self, df):
        """Chronos x hour group interaction features for meta-model"""
        if isinstance(df.index, pd.DatetimeIndex):
            hour_arr = df.index.hour.to_numpy(dtype=np.float64)
        else:
            hour_arr = df["meta__hour"].to_numpy(dtype=np.float64)

        is_night = ((hour_arr >= 0) & (hour_arr <= 7)).astype(np.float64)
        is_morning = ((hour_arr >= 8) & (hour_arr <= 11)).astype(np.float64)
        is_afternoon = ((hour_arr >= 12) & (hour_arr <= 17)).astype(np.float64)
        is_evening = ((hour_arr >= 18) & (hour_arr <= 23)).astype(np.float64)

        new_cols = []
        for col in self.PRED_COLS:
            if 'CHRONOS' not in col.upper():
                continue
            short = col.replace("_Pred", "")
            df[f"{short}_x_night_w"] = df[col].to_numpy(dtype=np.float64) * is_night
            new_cols.append(f"{short}_x_night_w")
            df[f"{short}_x_morning_w"] = df[col].to_numpy(dtype=np.float64) * is_morning
            new_cols.append(f"{short}_x_morning_w")
            df[f"{short}_x_afternoon_w"] = df[col].to_numpy(dtype=np.float64) * is_afternoon
            new_cols.append(f"{short}_x_afternoon_w")
            df[f"{short}_x_evening_w"] = df[col].to_numpy(dtype=np.float64) * is_evening
            new_cols.append(f"{short}_x_evening_w")

        return df, new_cols

    def _build_seasonal_interactions(self, df):
        season_map = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        }

        if isinstance(df.index, pd.DatetimeIndex):
            month_series = pd.Series(df.index.month, index=df.index)
        else:
            month_series = df["meta__month"]

        new_cols = []
        for season in ['winter', 'spring', 'summer', 'fall']:
            is_season = (month_series.map(season_map) == season).astype(np.float64)
            for col in self.PRED_COLS:
                short = col.replace("_Pred", "")
                df[f"{short}_x_{season}"] = df[col].to_numpy(dtype=np.float64) * is_season
                new_cols.append(f"{short}_x_{season}")

        return df, new_cols

    def _rolling_ridge_conditional(self, enriched: pd.DataFrame, meta_cols: list[str], return_meta_cols: bool = False):
        cond_enriched, interaction_cols = self._build_conditional_features(enriched)
        cond_meta_cols = meta_cols + interaction_cols
        ser, mape_val = self._rolling_ridge_batch(cond_enriched, cond_meta_cols, alpha=1000.0)
        if return_meta_cols:
            return ser, mape_val, cond_meta_cols
        return ser, mape_val

    def _expanding_meta_ridge(self, enriched: pd.DataFrame, results: dict):
        if "fold_id" not in enriched.columns:
            return None, None
        fold_ids = sorted(enriched["fold_id"].unique())
        if len(fold_ids) < self.MIN_WARMUP_FOLDS + 1:
            return None, None

        strategy_cols = []
        for name, (_, preds) in results.items():
            col = f"strat__{name}"
            enriched[col] = preds.reindex(enriched.index)
            strategy_cols.append(col)

        # Holiday-boosted Chronos weight (Fix 2.3)
        if "CHRONOS_Pred" in enriched.columns:
            from src.holiday_calendar import build_event_window_map
            idx = enriched.index
            if isinstance(idx, pd.DatetimeIndex):
                years = range(idx.year.min() - 1, idx.year.max() + 2)
                event_map = build_event_window_map(years=years)
                HOLIDAY_SET = {
                    "religious_pre_2", "religious_pre_1",
                    "religious_day_1", "religious_day_2", "religious_day_3",
                    "religious_post_1",
                    "official_pre_1", "official_day", "official_post_1",
                }
                is_hol = np.array([
                    1.0 if event_map.get(ts.date(), "normal") in HOLIDAY_SET else 0.0
                    for ts in idx
                ], dtype=np.float64)
            else:
                is_hol = np.zeros(len(enriched), dtype=np.float64)
            chronos_boost = enriched["CHRONOS_Pred"].to_numpy(dtype=np.float64) * is_hol
            enriched["strat__Chronos_boost"] = chronos_boost
            strategy_cols.append("strat__Chronos_boost")

        base_mean = enriched[self.PRED_COLS].mean(axis=1)
        for col in strategy_cols:
            enriched[col] = enriched[col].fillna(base_mean)

        parts = []
        for i, fid in enumerate(fold_ids):
            if i < self.MIN_WARMUP_FOLDS:
                continue
            train_mask = enriched["fold_id"].isin(fold_ids[:i])
            test_mask = enriched["fold_id"] == fid
            X_tr = enriched.loc[train_mask, strategy_cols]
            y_tr = enriched.loc[train_mask, "Actual"]
            X_te = enriched.loc[test_mask, strategy_cols]
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0])
            ridge.fit(X_tr, y_tr)
            parts.append(pd.Series(ridge.predict(X_te), index=X_te.index))

        all_p = pd.concat(parts)
        valid_idx = all_p.index
        mape = calculate_mape(enriched.loc[valid_idx, "Actual"], all_p.loc[valid_idx])
        full_preds = enriched[self.PRED_COLS].mean(axis=1).copy()
        full_preds.loc[valid_idx] = all_p.loc[valid_idx]
        return full_preds, mape

    def _segment_meta_ridge(self, enriched: pd.DataFrame, results: dict):
        """Strategy 9: Segment-aware Meta-Ridge — normal/holiday/post_holiday icin ayri RidgeCV."""
        if "fold_id" not in enriched.columns:
            return None, None
        fold_ids = sorted(enriched["fold_id"].unique())
        if len(fold_ids) < self.MIN_WARMUP_FOLDS + 1:
            return None, None
        if "event_window" not in enriched.columns:
            return None, None

        strategy_cols = []
        for name, (_, preds) in results.items():
            col = f"strat__{name}"
            enriched[col] = preds.reindex(enriched.index)
            strategy_cols.append(col)

        if "CHRONOS_Pred" in enriched.columns:
            from src.holiday_calendar import build_event_window_map
            idx = enriched.index
            if isinstance(idx, pd.DatetimeIndex):
                years = range(idx.year.min() - 1, idx.year.max() + 2)
                event_map = build_event_window_map(years=years)
                is_hol = np.array([
                    1.0 if event_map.get(ts.date(), "normal") in (HOLIDAY_SEGMENT_SET | POST_HOLIDAY_SEGMENT_SET) else 0.0
                    for ts in idx
                ], dtype=np.float64)
            else:
                is_hol = np.zeros(len(enriched), dtype=np.float64)
            chronos_boost = enriched["CHRONOS_Pred"].to_numpy(dtype=np.float64) * is_hol
            enriched["strat__Chronos_boost"] = chronos_boost
            strategy_cols.append("strat__Chronos_boost")

        base_mean = enriched[self.PRED_COLS].mean(axis=1)
        for col in strategy_cols:
            enriched[col] = enriched[col].fillna(base_mean)

        ew_series = enriched["event_window"]
        normal_mask = ~ew_series.isin(HOLIDAY_SEGMENT_SET | POST_HOLIDAY_SEGMENT_SET)
        holiday_mask = ew_series.isin(HOLIDAY_SEGMENT_SET)
        post_holiday_mask = ew_series.isin(POST_HOLIDAY_SEGMENT_SET)

        segment_defs = [
            ("normal", normal_mask, 100.0),
            ("holiday", holiday_mask, 500.0),
            ("post_holiday", post_holiday_mask, 1000.0),
        ]

        X_all = enriched[strategy_cols].to_numpy(dtype=np.float64)
        y_all = enriched["Actual"].to_numpy(dtype=np.float64)
        mask_all = {seg: msk.to_numpy(dtype=bool) for seg, msk, _ in segment_defs}

        parts = []
        for i, fid in enumerate(fold_ids):
            if i < self.MIN_WARMUP_FOLDS:
                continue
            train_mask = enriched["fold_id"].isin(fold_ids[:i]).to_numpy(dtype=bool)
            test_mask = (enriched["fold_id"] == fid).to_numpy(dtype=bool)
            test_rows = enriched.loc[test_mask]
            fold_preds = pd.Series(np.nan, index=test_rows.index, dtype=np.float64)

            for seg_name, seg_full_mask, alpha in segment_defs:
                seg_train = train_mask & mask_all[seg_name]
                seg_test = test_mask & mask_all[seg_name]
                if seg_train.sum() < 24:
                    continue
                if not seg_test.any():
                    continue
                ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0])
                ridge.fit(X_all[seg_train], y_all[seg_train])
                seg_idx = enriched.index[seg_test]
                fold_preds.loc[seg_idx] = ridge.predict(X_all[seg_test])

            # Fallback: global Ridge for rows not covered by any segment
            missing = fold_preds.isna()
            if missing.any():
                train_all = enriched.loc[train_mask]
                test_miss = enriched.loc[test_mask][missing.values]
                ridge_g = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0])
                ridge_g.fit(train_all[strategy_cols], train_all["Actual"])
                fold_preds.loc[missing[missing].index] = ridge_g.predict(test_miss[strategy_cols])

            parts.append(fold_preds)

        all_p = pd.concat(parts)
        valid_idx = all_p.index
        mape = calculate_mape(enriched.loc[valid_idx, "Actual"], all_p.loc[valid_idx])
        full_preds = enriched[self.PRED_COLS].mean(axis=1).copy()
        full_preds.loc[valid_idx] = all_p.loc[valid_idx]
        return full_preds, mape

    def _inverse_mape_weighting(self, full_df):
        if "fold_id" not in full_df.columns:
            return None, None

        fold_ids = sorted(full_df["fold_id"].unique())
        if len(fold_ids) < self.MIN_WARMUP_FOLDS + 1:
            return None, None

        inv_parts = []
        for i, fid in enumerate(fold_ids):
            if i < self.MIN_WARMUP_FOLDS:
                continue
            prior_mask = full_df["fold_id"].isin(fold_ids[:i])
            prior = full_df.loc[prior_mask]
            test_mask = full_df["fold_id"] == fid
            test = full_df.loc[test_mask]

            model_mapes = []
            for col in self.PRED_COLS:
                m = calculate_mape(prior["Actual"], prior[col])
                model_mapes.append(max(m, 1e-6))

            inv = np.array([1.0 / m for m in model_mapes])
            weights = inv / inv.sum()
            preds = sum(w * test[col].values for w, col in zip(weights, self.PRED_COLS))
            inv_parts.append(pd.Series(preds, index=test.index))

        all_inv_preds = pd.concat(inv_parts)
        valid_idx = all_inv_preds.index
        mape = calculate_mape(
            full_df.loc[valid_idx, "Actual"],
            all_inv_preds.loc[valid_idx],
        )

        final_mapes = [max(calculate_mape(full_df["Actual"], full_df[c]), 1e-6) for c in self.PRED_COLS]
        inv_final = np.array([1.0 / m for m in final_mapes])
        self.best_weights = dict(zip(self.PRED_COLS, np.round(inv_final / inv_final.sum(), 4)))
        print(f"   Ters MAPE Agirliklari: {self.best_weights}")

        full_preds = full_df[self.PRED_COLS].mean(axis=1).copy()
        full_preds.loc[valid_idx] = all_inv_preds.loc[valid_idx]
        return full_preds, mape

    def _constrained_optimization(self, full_df):
        if "fold_id" not in full_df.columns:
            return None, None

        fold_ids = sorted(full_df["fold_id"].unique())
        if len(fold_ids) < self.MIN_WARMUP_FOLDS + 1:
            return None, None

        opt_parts = []
        n_models = len(self.PRED_COLS)

        for i, fid in enumerate(fold_ids):
            if i < self.MIN_WARMUP_FOLDS:
                continue
            prior_mask = full_df["fold_id"].isin(fold_ids[:i])
            prior = full_df.loc[prior_mask]
            test_mask = full_df["fold_id"] == fid
            test = full_df.loc[test_mask]

            prior_actuals = prior["Actual"].values
            prior_preds_matrix = prior[self.PRED_COLS].values
            test_preds_matrix = test[self.PRED_COLS].values

            def mape_objective(w):
                blended = prior_preds_matrix @ w
                return calculate_mape(prior_actuals, blended)

            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
            bounds = [(0, 1)] * n_models
            x0 = np.ones(n_models) / n_models

            res = minimize(
                mape_objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            w_opt = res.x
            preds = test_preds_matrix @ w_opt
            opt_parts.append(pd.Series(preds, index=test.index))

        all_opt_preds = pd.concat(opt_parts)
        valid_idx = all_opt_preds.index
        mape = calculate_mape(
            full_df.loc[valid_idx, "Actual"],
            all_opt_preds.loc[valid_idx],
        )

        prior_actuals = full_df["Actual"].values
        prior_preds_matrix = full_df[self.PRED_COLS].values

        def final_objective(w):
            return calculate_mape(prior_actuals, prior_preds_matrix @ w)

        res_final = minimize(
            final_objective,
            np.ones(n_models) / n_models,
            method="SLSQP",
            bounds=[(0, 1)] * n_models,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            options={"maxiter": 500, "ftol": 1e-10},
        )
        self.best_weights = dict(zip(self.PRED_COLS, np.round(res_final.x, 4)))
        print(f"   Optimizasyon Agirliklari: {self.best_weights}")

        full_preds = full_df[self.PRED_COLS].mean(axis=1).copy()
        full_preds.loc[valid_idx] = all_opt_preds.loc[valid_idx]
        return full_preds, mape

    def _train_final_meta_model(self, enriched, meta_cols, cond_meta_cols=None, results=None):
        if self.best_method == "Segment_Meta_Ridge" and results is not None:
            strategy_cols = [c for c in enriched.columns if c.startswith("strat__")]
            X_meta = enriched[strategy_cols].fillna(enriched[self.PRED_COLS].mean(axis=1), axis=0)
            y_meta = enriched["Actual"]
            ew_series = enriched.get("event_window", pd.Series("normal", index=enriched.index))

            self.meta_model = {}
            self.meta_feature_cols = strategy_cols

            segments = [
                ("normal", ~ew_series.isin(HOLIDAY_SEGMENT_SET | POST_HOLIDAY_SEGMENT_SET)),
                ("holiday", ew_series.isin(HOLIDAY_SEGMENT_SET)),
                ("post_holiday", ew_series.isin(POST_HOLIDAY_SEGMENT_SET)),
            ]
            for seg_name, seg_mask in segments:
                if seg_mask.sum() < 24:
                    continue
                ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0])
                ridge.fit(X_meta[seg_mask], y_meta[seg_mask])
                self.meta_model[seg_name] = ridge
                print(f"[StackingManager] Uretim Segment Meta-Ridge [{seg_name}] "
                      f"n={seg_mask.sum()} bias={ridge.intercept_:.4f}")
            if not self.meta_model:
                self.best_method = "Meta_Ridge"
                self._train_final_meta_model(enriched, meta_cols, cond_meta_cols, results)
                return

        elif self.best_method == "Meta_Ridge" and results is not None:
            strategy_cols = [c for c in enriched.columns if c.startswith("strat__")]
            X_meta = enriched[strategy_cols].fillna(enriched[self.PRED_COLS].mean(axis=1), axis=0)
            y_meta = enriched["Actual"]
            self.meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0])
            self.meta_model.fit(X_meta, y_meta)
            self.meta_feature_cols = strategy_cols
            print(f"[StackingManager] Uretim Meta-Ridge — {len(strategy_cols)} strateji ozelligi, "
                  f"bias={self.meta_model.intercept_:.4f}")

        elif self.best_method in ("Rolling_Ridge_Context", "Conditional_Ridge"):
            use_meta = cond_meta_cols if self.best_method == "Conditional_Ridge" and cond_meta_cols is not None else meta_cols
            lb = self.rolling_lookback_hours
            if self.best_method == "Conditional_Ridge" and cond_meta_cols is not None:
                cond_enriched, _ = self._build_conditional_features(enriched)
                X_meta = cond_enriched[use_meta].iloc[-lb:]
                y_meta = cond_enriched["Actual"].iloc[-lb:]
            else:
                X_meta = enriched[use_meta].iloc[-lb:]
                y_meta = enriched["Actual"].iloc[-lb:]
            alpha = 1000.0 if self.best_method == "Conditional_Ridge" else 100.0
            self.meta_model = Ridge(alpha=alpha, fit_intercept=True)
            self.meta_model.fit(X_meta, y_meta)
            self.meta_feature_cols = use_meta
            pred_part = {c: round(float(self.meta_model.coef_[use_meta.index(c)]), 4)
                         for c in self.PRED_COLS if c in use_meta}
            print(f"[StackingManager] Uretim Rolling Ridge — taban katsayilar: {pred_part}")
            print(f"[StackingManager] Uretim Rolling Ridge — alpha={alpha}, bias={self.meta_model.intercept_:.4f}")

        elif self.best_method == "Hourly_Ridge":
            self.meta_model = HourlyRidgeEnsemble(pred_cols=self.PRED_COLS, alpha=100.0)
            self.meta_model.fit(enriched)
            self.meta_feature_cols = self.PRED_COLS
            print("[StackingManager] Uretim Hourly Ridge — 24 model egitildi.")

        elif self.best_method == "Interaction_Ridge":
            self.meta_model = InteractionRidgeEnsemble(pred_cols=self.PRED_COLS, alpha=500.0)
            self.meta_model.fit(enriched)
            self.meta_feature_cols = self.PRED_COLS
            print("[StackingManager] Uretim Interaction Ridge — etkilesimli model egitildi.")

        elif self.best_method == "FWLS_Ridge":
            self.meta_model = FWLSRidgeEnsemble(pred_cols=self.PRED_COLS, alpha=1000.0)
            self.meta_model.fit(enriched)
            self.meta_feature_cols = self.PRED_COLS
            print("[StackingManager] Uretim FWLS Ridge — model egitildi.")

        elif self.best_method == "GBDT_Stacking":
            self.meta_model = LightGBMStackingEnsemble(pred_cols=self.PRED_COLS)
            self.meta_model.fit(enriched)
            self.meta_feature_cols = self.PRED_COLS
            print("[StackingManager] Uretim GBDT Stacking — model egitildi.")

        elif self.best_method == "Hourly_FWLS_Ridge":
            self.meta_model = HourlyFWLSRidgeEnsemble(pred_cols=self.PRED_COLS, alpha=100.0)
            self.meta_model.fit(enriched)
            self.meta_feature_cols = self.PRED_COLS
            print("[StackingManager] Uretim Hourly FWLS Ridge — model egitildi.")

        elif self.best_method == "Simple_Average":
            self.meta_model = SimpleAverageEnsemble(pred_cols=self.PRED_COLS)
            self.meta_feature_cols = self.PRED_COLS
            print("[StackingManager] Uretim Simple Average — katsayilar esit.")

        elif self.best_method in ("Inverse_MAPE", "Optimized_Weights"):
            self.meta_model = StaticWeightedEnsemble(pred_cols=self.PRED_COLS, weights=self.best_weights)
            self.meta_feature_cols = self.PRED_COLS
            print(f"[StackingManager] Uretim {self.best_method} — katsayilar: {self.best_weights}")

        else:
            X_meta = enriched[meta_cols]
            y_meta = enriched["Actual"]
            self.meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0])
            self.meta_model.fit(X_meta, y_meta)
            self.meta_feature_cols = meta_cols
            coef_arr = self.meta_model.coef_
            pred_part = {c: round(float(coef_arr[meta_cols.index(c)]), 4) for c in self.PRED_COLS}
            print(f"[StackingManager] Uretim Ridge — taban katsayilar: {pred_part}")
            print(
                f"[StackingManager] Uretim Ridge — +{len(meta_cols) - len(self.PRED_COLS)} baglam ozelligi, "
                f"bias={self.meta_model.intercept_:.4f}"
            )

    def save_model(self, filename="stacking_ridge.joblib"):
        if self.meta_model is None:
            print("[StackingManager] Kaydedilecek model yok.")
            return
        path = os.path.join(self.model_dir, filename)
        save_data = {
            "meta_model": self.meta_model,
            "meta_feature_cols": self.meta_feature_cols,
            "pred_cols": self.PRED_COLS,
            "best_method": self.best_method,
            "best_weights": self.best_weights,
            "artifact_metadata": self.artifact_metadata,
        }
        joblib.dump(save_data, path)
        print(f"[StackingManager] Meta-model kaydedildi: {path}")
