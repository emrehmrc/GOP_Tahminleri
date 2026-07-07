import lightgbm as lgb
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

import config_live
from config_live import MODEL_NAME
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

class HybridLightGBMModel:
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
    def feature_name_(self):
        return self.model_wd_sat.feature_name_

    @property
    def feature_importances_(self):
        return (self.model_wd_sat.feature_importances_ + self.model_we.feature_importances_) / 2.0


class LightGBMManager:
    def __init__(self):
        self.model = None
        self.model_dir = os.path.join(project_root, 'models')
        os.makedirs(self.model_dir, exist_ok=True)

    def train_model(self, X_train, y_train, X_test, y_test):
        from config_live import MAX_TRAIN_SIZE
        if MAX_TRAIN_SIZE is not None and len(X_train) > MAX_TRAIN_SIZE:
            print(f"[LightGBMManager] Capping training set size to last {MAX_TRAIN_SIZE} samples (was {len(X_train)})")
            X_train = X_train.iloc[-MAX_TRAIN_SIZE:]
            y_train = y_train.iloc[-MAX_TRAIN_SIZE:]

        print("[LightGBMManager] Initializing LightGBM Regressor...")
        
        import json
        from config_live import HPO_PARAMS_SUFFIX, ENABLE_WEEKEND_SPLIT_LGBM
        
        params = {
            'n_estimators': 2000,
            'objective': 'regression',
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            'importance_type': 'gain',
            'learning_rate': 0.0895602324603589,
            'num_leaves': 40,
            'max_depth': 4,
            'min_child_samples': 33,
            'subsample': 0.6702191322327771,
            'colsample_bytree': 0.6740711121764694,
            'reg_alpha': 1.003975348777657,
            'reg_lambda': 6.587716807268391
        }
        
        suffix = HPO_PARAMS_SUFFIX if HPO_PARAMS_SUFFIX else ""
        
        # Load weekday (general) parameters
        params_wd = params.copy()
        param_file_wd = os.path.join(project_root, f"best_params_lgbm_general{suffix}.json")
        if os.path.exists(param_file_wd):
            print(f"[LightGBMManager] Loading optimized weekday parameters from: {param_file_wd}")
            try:
                with open(param_file_wd, "r") as f:
                    opt_params_wd = json.load(f)
                params_wd.update(opt_params_wd)
            except Exception as e:
                print(f"[LightGBMManager] UYARI: Weekday parametre dosyası okunamadı: {e}")
        else:
            print(f"[LightGBMManager] Weekday parametre dosyası bulunamadı ({param_file_wd}). Varsayılanlar kullanılacak.")
            
        # Load weekend parameters
        params_we = params_wd.copy()
        param_file_we = os.path.join(project_root, f"best_params_lgbm_weekend{suffix}.json")
        if os.path.exists(param_file_we):
            print(f"[LightGBMManager] Loading optimized weekend parameters from: {param_file_we}")
            try:
                with open(param_file_we, "r") as f:
                    opt_params_we = json.load(f)
                params_we.update(opt_params_we)
            except Exception as e:
                print(f"[LightGBMManager] UYARI: Weekend parametre dosyası okunamadı: {e}")
        else:
            print(f"[LightGBMManager] Weekend parametre dosyası bulunamadı ({param_file_we}). Hafta içi parametreleri kullanılacak.")
            
        # Override for fast mode if active
        # OE_FULL_STRENGTH (OpenEvolve sadik proxy): stride uygulanir ama modeller TAM GUC kalir.
        import config_live as config
        if getattr(config, 'FAST_MODE', False) and not os.environ.get('OE_FULL_STRENGTH'):
            fast_max_iter = getattr(config, 'FAST_MAX_ITER', 150)
            print(f"[LightGBMManager] Hızlı mod aktif. n_estimators değeri {fast_max_iter} olarak güncelleniyor.")
            for p_dict in [params_wd, params_we]:
                p_dict['n_estimators'] = fast_max_iter
                p_dict['early_stopping_rounds'] = min(15, p_dict.get('early_stopping_rounds', 50))
                
                # Scale learning rate in fast mode if low to prevent underfitting
                orig_lr = p_dict.get('learning_rate', 0.1)
                scaled_lr = min(0.12, orig_lr * (1000.0 / fast_max_iter))
                if scaled_lr > orig_lr:
                    p_dict['learning_rate'] = scaled_lr

        # Check for dayofweek to determine split availability
        if isinstance(X_train.index, pd.DatetimeIndex):
            dow_train = X_train.index.dayofweek
        elif 'Haftanın_Günü' in X_train.columns:
            dow_train = X_train['Haftanın_Günü']
        else:
            dow_train = None

        if ENABLE_WEEKEND_SPLIT_LGBM and dow_train is not None:
            print("[LightGBMManager] Hybrid Split Model training active (Mon-Sat vs Sat-Sun).")
            
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
            
            print(f"[LightGBMManager] Training Model A (WD+Sat) on {len(X_train_wd_sat)} samples...")
            model_wd_sat = self._fit_single_model(X_train_wd_sat, y_train_wd_sat, params_wd)
            
            print(f"[LightGBMManager] Training Model B (WE) on {len(X_train_we)} samples...")
            # Sunday boost: WE model Sat+Sun birlikte eğitir ama Saturday ~5x daha
            # fazla sample → Sunday under-predict (MAPE %7.4 vs %5.8). Sunday örneklerine
            # extra weight ver.
            from config_live import LGBM_SUNDAY_WEIGHT_BOOST
            we_boost = np.ones(len(X_train_we))
            if isinstance(dow_train, pd.Series):
                we_dow = dow_train[we_mask] if isinstance(we_mask, pd.Series) else dow_train[np.array(we_mask)]
            else:
                we_dow = dow_train[np.array(we_mask)]
            we_boost[we_dow == 6] = 1.0 + LGBM_SUNDAY_WEIGHT_BOOST
            model_we = self._fit_single_model(X_train_we, y_train_we, params_we, sample_weight=we_boost)
            
            self.model = HybridLightGBMModel(model_wd_sat, model_we)
            print("[LightGBMManager] Hybrid training finished successfully.")
        else:
            print(f"[LightGBMManager] Training standard single model on {len(X_train)} samples...")
            self.model = self._fit_single_model(X_train, y_train, params_wd)


    def _fit_single_model(self, X_train, y_train, params, sample_weight=None):
        from config_live import ENABLE_GBDT_REFIT
        
        # Reserve last 20% of training data (chronologically) for early stopping validation.
        split_idx = int(len(X_train) * 0.8)
        if split_idx < len(X_train) and split_idx > 10:
            X_train_fit = X_train.iloc[:split_idx]
            y_train_fit = y_train.iloc[:split_idx]
            X_val = X_train.iloc[split_idx:]
            y_val = y_train.iloc[split_idx:]
            sw_split = sample_weight[:split_idx] if sample_weight is not None else None
            sw_val = sample_weight[split_idx:] if sample_weight is not None else None
        else:
            X_train_fit, y_train_fit = X_train, y_train
            X_val, y_val = None, None
            sw_split = sample_weight
            sw_val = None

        eval_sets = [(X_train_fit, y_train_fit)]
        if X_val is not None:
            eval_sets.append((X_val, y_val))

        from src.recency_weight import recency_sample_weight
        sw_fit = recency_sample_weight(X_train_fit.index)
        if sw_split is not None:
            sw_fit = sw_fit * sw_split

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train_fit, y_train_fit,
            sample_weight=sw_fit,
            eval_set=eval_sets,
            eval_metric='mae',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )

        if ENABLE_GBDT_REFIT and X_val is not None:
            best_iter = model.best_iteration_
            if best_iter is None or best_iter <= 0:
                best_iter = params.get('n_estimators', 2000)
            
            print(f"[LightGBMManager] Erken durdurma ile en iyi iterasyon: {best_iter}. Model tüm veriyle (%100) yeniden eğitiliyor...")
            
            refit_params = params.copy()
            refit_params['n_estimators'] = best_iter
            if 'early_stopping_rounds' in refit_params:
                del refit_params['early_stopping_rounds']
                
            sw_refit = recency_sample_weight(X_train.index)
            if sample_weight is not None:
                sw_refit = sw_refit * sample_weight
            model_refit = lgb.LGBMRegressor(**refit_params)
            model_refit.fit(
                X_train, y_train,
                sample_weight=sw_refit,
                eval_metric='mae',
                callbacks=[]
            )
            return model_refit
        else:
            return model

    def evaluate(self, X_test, y_test):
        if self.model is None:
            print("Model is not trained yet!")
            return

        preds = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = calculate_mape(y_test, preds)
        
        print("\n--- LightGBM Performance ---")
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"MAPE : %{mape:.2f}") 
        print("----------------------------\n")
        
        return preds

    def get_feature_importance(self, max_num=20):
        if self.model is None:
            print("Model egitilmedi!")
            return None
        
        if isinstance(self.model, HybridLightGBMModel):
            # Plot for Model A as representation or both separately
            print("[LightGBMManager] Hybrid Split: Showing feature importances from Model A (WD+Sat).")
            model_to_plot = self.model.model_wd_sat
        else:
            model_to_plot = self.model

        df_imp = pd.DataFrame({
            'Feature': model_to_plot.feature_name_,
            'Importance': model_to_plot.feature_importances_
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        plt.figure(figsize=(10, 8))
        lgb.plot_importance(model_to_plot, max_num_features=max_num, importance_type='gain', 
                           title='LightGBM Feature Importance (Gain)')
        plt.show()
        
        return df_imp

    def save_model(self, filename='model_lgbm.txt'):
        if self.model is None:
            print("Model egitilmedi, kayit yapilamiyor.")
            return

        save_path = os.path.join(self.model_dir, filename)
        
        try:
            if isinstance(self.model, HybridLightGBMModel):
                # Save both submodels
                wd_sat_path = save_path.replace('.txt', '_wd_sat.txt')
                we_path = save_path.replace('.txt', '_we.txt')
                
                with open(wd_sat_path, 'w', encoding='utf-8') as f:
                    f.write(self.model.model_wd_sat.booster_.model_to_string())
                with open(we_path, 'w', encoding='utf-8') as f:
                    f.write(self.model.model_we.booster_.model_to_string())
                    
                # Write a pointer file to the main location
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write("HYBRID_LGBM_SPLIT\n")
                    f.write(f"wd_sat_model: {os.path.basename(wd_sat_path)}\n")
                    f.write(f"we_model: {os.path.basename(we_path)}\n")
                print(f"[LightGBMManager] Hybrid models successfully saved to {wd_sat_path} and {we_path}")
            else:
                model_str = self.model.booster_.model_to_string()
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(model_str)
                print(f"[LightGBMManager] Model basariyla kaydedildi: {save_path}")
                
        except Exception as e:
            print(f"[LightGBMManager] Kayit sirasinda hata olustu: {e}")

