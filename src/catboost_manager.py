import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config_live import MODEL_NAME
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

class HybridCatBoostModel:
    def __init__(self, model_wd_sat, model_we):
        self.model_wd_sat = model_wd_sat
        self.model_we = model_we

    def predict(self, X):
        preds = np.zeros(len(X))
        
        # Get day of week
        if isinstance(X.index, pd.DatetimeIndex):
            dow = X.index.dayofweek
        elif 'Haftanın_Günü' in X.columns:
            dow = X['Haftanın_Günü']
        else:
            return self.model_wd_sat.predict(X)
            
        wd_sat_mask = dow <= 5
        sun_mask = dow == 6
        
        if isinstance(wd_sat_mask, pd.Series):
            wd_sat_mask = wd_sat_mask.values
        if isinstance(sun_mask, pd.Series):
            sun_mask = sun_mask.values
            
        if wd_sat_mask.any():
            preds[wd_sat_mask] = self.model_wd_sat.predict(X[wd_sat_mask])
        if sun_mask.any():
            preds[sun_mask] = self.model_we.predict(X[sun_mask])
            
        return preds

    @property
    def feature_names_(self):
        return self.model_wd_sat.feature_names_

    def get_feature_importance(self):
        return (self.model_wd_sat.get_feature_importance() + self.model_we.get_feature_importance()) / 2.0


class CatBoostManager:
    def __init__(self):
        self.model = None
        self.model_dir = os.path.join(project_root, 'models')
        os.makedirs(self.model_dir, exist_ok=True)

    def train_model(self, X_train, y_train, X_test, y_test):
        from config_live import MAX_TRAIN_SIZE
        if MAX_TRAIN_SIZE is not None and len(X_train) > MAX_TRAIN_SIZE:
            print(f"[CatBoostManager] Capping training set size to last {MAX_TRAIN_SIZE} samples (was {len(X_train)})")
            X_train = X_train.iloc[-MAX_TRAIN_SIZE:]
            y_train = y_train.iloc[-MAX_TRAIN_SIZE:]

        print("[CatBoostManager] Initializing CatBoost Regressor...")
        
        cat_features_names = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if cat_features_names:
            print(f"[CatBoostManager] Kategorik sutunlar tespit edildi: {cat_features_names}")
        
        import json
        from config_live import HPO_PARAMS_SUFFIX, ENABLE_WEEKEND_SPLIT_CAT
        
        params = {
            'iterations': 1500,
            'learning_rate': 0.1308660259549872,
            'depth': 6,
            'l2_leaf_reg': 2,
            'random_strength': 1.5130937203539108,
            'bagging_temperature': 0.6835231104564964,
            'min_data_in_leaf': 16,
            'loss_function': 'MAE',
            'eval_metric': 'MAPE',
            'random_seed': 42,
            'verbose': 100,
            'early_stopping_rounds': 50,
            'allow_writing_files': False,
            'cat_features': cat_features_names
        }
        
        suffix = HPO_PARAMS_SUFFIX if HPO_PARAMS_SUFFIX else ""
        
        # Load weekday (general) parameters
        params_wd = params.copy()
        param_file_wd = os.path.join(project_root, f"best_params_cat_general{suffix}.json")
        if os.path.exists(param_file_wd):
            print(f"[CatBoostManager] Loading optimized weekday parameters from: {param_file_wd}")
            try:
                with open(param_file_wd, "r") as f:
                    opt_params_wd = json.load(f)
                params_wd.update(opt_params_wd)
            except Exception as e:
                print(f"[CatBoostManager] UYARI: Weekday parametre dosyası okunamadı: {e}")
        else:
            print(f"[CatBoostManager] Weekday parametre dosyası bulunamadı ({param_file_wd}). Varsayılanlar kullanılacak.")
            
        # Load weekend parameters
        params_we = params_wd.copy()
        param_file_we = os.path.join(project_root, f"best_params_cat_weekend{suffix}.json")
        if os.path.exists(param_file_we):
            print(f"[CatBoostManager] Loading optimized weekend parameters from: {param_file_we}")
            try:
                with open(param_file_we, "r") as f:
                    opt_params_we = json.load(f)
                params_we.update(opt_params_we)
            except Exception as e:
                print(f"[CatBoostManager] UYARI: Weekend parametre dosyası okunamadı: {e}")
        else:
            print(f"[CatBoostManager] Weekend parametre dosyası bulunamadı ({param_file_we}). Hafta içi parametreleri kullanılacak.")
            
        # Override for fast mode if active
        # OE_FULL_STRENGTH (OpenEvolve sadik proxy): stride uygulanir ama modeller TAM GUC kalir.
        import config_live as config
        if getattr(config, 'FAST_MODE', False) and not os.environ.get('OE_FULL_STRENGTH'):
            fast_max_iter = getattr(config, 'FAST_MAX_ITER', 150)
            print(f"[CatBoostManager] Hızlı mod aktif. iterations değeri {fast_max_iter} olarak güncelleniyor.")
            for p_dict in [params_wd, params_we]:
                p_dict['iterations'] = fast_max_iter
                p_dict['depth'] = min(6, p_dict.get('depth', 6))
                p_dict['early_stopping_rounds'] = min(15, p_dict.get('early_stopping_rounds', 50))

        # Check for dayofweek to determine split availability
        if isinstance(X_train.index, pd.DatetimeIndex):
            dow_train = X_train.index.dayofweek
        elif 'Haftanın_Günü' in X_train.columns:
            dow_train = X_train['Haftanın_Günü']
        else:
            dow_train = None

        if ENABLE_WEEKEND_SPLIT_CAT and dow_train is not None:
            print("[CatBoostManager] Hybrid Split Model training active (Mon-Sat vs Sat-Sun).")
            
            # Model A (WD+Sat): Mon-Sat (dow <= 5)
            wd_sat_mask = dow_train <= 5
            if isinstance(wd_sat_mask, pd.Series):
                wd_sat_mask = wd_sat_mask.values
            X_train_wd_sat = X_train[wd_sat_mask]
            y_train_wd_sat = y_train[wd_sat_mask]
            
            # Model B (WE): Sat-Sun (dow >= 5)
            we_mask = dow_train >= 5
            if isinstance(we_mask, pd.Series):
                we_mask = we_mask.values
            X_train_we = X_train[we_mask]
            y_train_we = y_train[we_mask]
            
            print(f"[CatBoostManager] Training Model A (WD+Sat) on {len(X_train_wd_sat)} samples...")
            model_wd_sat = self._fit_single_model(X_train_wd_sat, y_train_wd_sat, params_wd)
            
            print(f"[CatBoostManager] Training Model B (WE) on {len(X_train_we)} samples...")
            model_we = self._fit_single_model(X_train_we, y_train_we, params_we)
            
            self.model = HybridCatBoostModel(model_wd_sat, model_we)
            print("[CatBoostManager] Hybrid training finished successfully.")
        else:
            print(f"[CatBoostManager] Training standard single model on {len(X_train)} samples...")
            self.model = self._fit_single_model(X_train, y_train, params_wd)


    def _fit_single_model(self, X_train, y_train, params):
        from config_live import ENABLE_GBDT_REFIT
        
        # Reserve last 20% of training data (chronologically) for early stopping validation.
        split_idx = int(len(X_train) * 0.8)
        if split_idx < len(X_train) and split_idx > 10:
            X_train_fit = X_train.iloc[:split_idx]
            y_train_fit = y_train.iloc[:split_idx]
            X_val = X_train.iloc[split_idx:]
            y_val = y_train.iloc[split_idx:]
        else:
            X_train_fit, y_train_fit = X_train, y_train
            X_val, y_val = None, None

        from src.recency_weight import recency_sample_weight
        sw_fit = recency_sample_weight(X_train_fit.index)

        model = CatBoostRegressor(**params)

        if X_val is not None:
            model.fit(
                X_train_fit, y_train_fit,
                sample_weight=sw_fit,
                eval_set=(X_val, y_val),
                use_best_model=True,
                verbose=False
            )
        else:
            model.fit(
                X_train_fit, y_train_fit,
                sample_weight=sw_fit,
                use_best_model=True,
                verbose=False
            )

        if ENABLE_GBDT_REFIT and X_val is not None:
            best_iter = model.get_best_iteration()
            if best_iter is None or best_iter <= 0:
                best_iter = params.get('iterations', 1500)
            
            print(f"[CatBoostManager] Erken durdurma ile en iyi iterasyon: {best_iter}. Model tüm veriyle (%100) yeniden eğitiliyor...")
            
            refit_params = params.copy()
            refit_params['iterations'] = best_iter
            if 'early_stopping_rounds' in refit_params:
                del refit_params['early_stopping_rounds']
                
            model_refit = CatBoostRegressor(**refit_params)
            model_refit.fit(
                X_train, y_train,
                sample_weight=recency_sample_weight(X_train.index),
                verbose=False
            )
            return model_refit
        else:
            return model

    def evaluate(self, X_test, y_test):
        if self.model is None:
            print("Model is not trained yet!")
            return None

        print("[CatBoostManager] Predicting...")
        preds = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = calculate_mape(y_test, preds)
        
        print("\n--- CatBoost Performance ---")
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"MAPE : %{mape:.2f}") 
        print("----------------------------\n")
        
        return preds

    def get_feature_importance(self, max_num=20):
        if self.model is None:
            print("Model egitilmedi!")
            return None
        
        if isinstance(self.model, HybridCatBoostModel):
            print("[CatBoostManager] Hybrid Split: Showing feature importances from Model A (WD+Sat).")
            model_to_plot = self.model.model_wd_sat
        else:
            model_to_plot = self.model

        imp_values = model_to_plot.get_feature_importance()
        feature_names = model_to_plot.feature_names_
        
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': imp_values})
        df_imp = df_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        plt.figure(figsize=(10, 8))
        top_df = df_imp.head(max_num).sort_values(by='Importance', ascending=True)
        plt.barh(top_df['Feature'], top_df['Importance'], color='skyblue')
        plt.xlabel('Importance Score')
        plt.title('CatBoost Feature Importance')
        plt.tight_layout()
        plt.show()
        
        return df_imp

    def save_model(self, filename='best_catboost_model.cbm'):
        if self.model is None:
            print("Model egitilmedi, kayit yapilamiyor.")
            return

        save_path = os.path.join(self.model_dir, filename)
        
        if isinstance(self.model, HybridCatBoostModel):
            # Save both submodels
            wd_sat_path = save_path.replace('.cbm', '_wd_sat.cbm')
            we_path = save_path.replace('.cbm', '_we.cbm')
            self.model.model_wd_sat.save_model(wd_sat_path)
            self.model.model_we.save_model(we_path)
            
            # Write a pointer file to the main location
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("HYBRID_CAT_SPLIT\n")
                f.write(f"wd_sat_model: {os.path.basename(wd_sat_path)}\n")
                f.write(f"we_model: {os.path.basename(we_path)}\n")
            print(f"[CatBoostManager] Hybrid CatBoost models saved to: {wd_sat_path} and {we_path}")
        else:
            self.model.save_model(save_path)
            print(f"[CatBoostManager] Model saved to: {save_path}")

    def load_model(self, filename='best_catboost_model.cbm'):
        load_path = os.path.join(self.model_dir, filename)
        if os.path.exists(load_path):
            is_hybrid = False
            try:
                with open(load_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line == "HYBRID_CAT_SPLIT":
                        is_hybrid = True
                        wd_sat_line = f.readline().strip()
                        we_line = f.readline().strip()
                        wd_sat_file = wd_sat_line.split(': ')[1]
                        we_file = we_line.split(': ')[1]
            except Exception:
                pass
                
            if is_hybrid:
                model_wd_sat = CatBoostRegressor()
                model_wd_sat.load_model(os.path.join(self.model_dir, wd_sat_file))
                model_we = CatBoostRegressor()
                model_we.load_model(os.path.join(self.model_dir, we_file))
                self.model = HybridCatBoostModel(model_wd_sat, model_we)
                print(f"[CatBoostManager] Hybrid CatBoost models loaded from pointer: {load_path}")
            else:
                self.model = CatBoostRegressor()
                self.model.load_model(load_path)
                print(f"[CatBoostManager] Model loaded from: {load_path}")
        else:
            print(f"Model file not found at: {load_path}")

