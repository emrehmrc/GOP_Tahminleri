"""
stacking_strategies.py — Ensemble meta-model stratejileri
===========================================================
Expanding-window cross-validation icin kullanilan bagimsiz ensemble stratejileri.
Her strateji fit(X, y) / predict(X) arayuzune sahiptir.

StackingManager (stacking_manager.py) bunlari import edip expanding-window
uzerinde karsilastirir, en iyisini secer ve uretime gonderir.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV


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
        self.weights = weights
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
        cdd = df["CDD_Cooling_Stress"].values / 20.0 if "CDD_Cooling_Stress" in df.columns else np.zeros(len(df))
        hdd = df["HDD_Heating_Stress"].values / 20.0 if "HDD_Heating_Stress" in df.columns else np.zeros(len(df))
        solar = df["Solar_Shaving_Proxy"].values / 1000.0 if "Solar_Shaving_Proxy" in df.columns else np.zeros(len(df))
        sin_hour = df["meta__sin_hour"].values if "meta__sin_hour" in df.columns else np.zeros(len(df))
        cos_hour = df["meta__cos_hour"].values if "meta__cos_hour" in df.columns else np.zeros(len(df))
        is_weekend = df["meta__is_weekend"].values if "meta__is_weekend" in df.columns else np.zeros(len(df))
        meta_feats = {'sin_hour': sin_hour, 'cos_hour': cos_hour, 'is_weekend': is_weekend, 'cdd': cdd, 'hdd': hdd, 'solar': solar}
        features = []
        for col in self.pred_cols:
            features.append(X[col].values)
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
            n_estimators=100, learning_rate=0.05, max_depth=3,
            num_leaves=8, min_child_samples=50, reg_alpha=10.0,
            reg_lambda=10.0, random_state=42, verbose=-1, n_jobs=-1,
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
        hour_raw = df.index.hour.values if isinstance(df.index, pd.DatetimeIndex) else np.zeros(len(df))
        is_sunset = ((hour_raw >= 15) & (hour_raw <= 20)).astype(np.float64)
        meta_feats = {'is_weekend': is_weekend, 'solar': solar, 'cdd': cdd, 'hdd': hdd, 'sunset': is_sunset}
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
                model = RidgeCV(alphas=[1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0], fit_intercept=True)
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