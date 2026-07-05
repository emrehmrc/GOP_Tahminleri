"""
config_live.py — ADM Canlı Pipeline Yapılandırması
====================================================
Boray/p0_updated config.py'den türetilmiş, canlı servis için sabitler.
Kopyalanan src/ modülleri (config_live.py'den import eder) ile çalışır.
"""

import os
from pathlib import Path

# ── Dizin yapısı ──────────────────────────────────────────────────────────────
LIVE_DIR = Path(__file__).parent

DATA_DIR         = LIVE_DIR / "data"
WEATHER_CACHE_DIR = DATA_DIR / "weather_cache"
MODELS_DIR       = LIVE_DIR / "models"
OUTPUT_DIR       = LIVE_DIR / "output"
ARCHIVE_DIR      = OUTPUT_DIR / "archive"
LOGS_DIR         = LIVE_DIR / "logs"
SRC_DIR          = LIVE_DIR / "src"

OOF_HISTORY_PATH = DATA_DIR / "oof_history.parquet"

# ── Run-context / reprodüksiyon (Faz -1) ──────────────────────────────────────
EDAS_ID              = "ADM"                    # multi-tenant (Faz 5) için şimdiden
MODEL_ARCHIVE_DIR    = MODELS_DIR / "archive"   # her run'ın modelleri: archive/<run_id>/
RUN_CONTEXT_PATH     = DATA_DIR / "run_context.json"
ARCHIVE_RETENTION_DAYS = 90

# ── Forecast/actuals log deposu (Faz 0) ────────────────────────────────────────
# OneDrive DIŞI: parquet append + DuckDB tek-dosya, OneDrive senkronuyla aynı
# anda yazılırsa kilitlenme/bozulma riski var. Bu yüzden canlı log verisi
# %LOCALAPPDATA% altında (senkronlanmıyor); OneDrive'a sadece günlük zip yedeği
# gider (LOG_BACKUP_DIR, aşağıda — proje dizini altında, git-ignored).
LOG_ROOT           = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "adm_live_logs"
FORECAST_LOG_DIR   = LOG_ROOT / "forecast_log"
ACTUALS_LOG_DIR    = LOG_ROOT / "actuals_log"
MONITORING_DB      = LOG_ROOT / "monitoring.duckdb"    # türetilmiş, parquet'ten rebuild edilebilir
KNOWN_EVENTS_CSV   = LOG_ROOT / "known_events.csv"     # Faz 2'de aktifleşir
LOG_BACKUP_DIR     = LOGS_DIR / "backup"                # OneDrive altı — günlük zip yedeği

# ── daily_scorecard (Faz 1) ────────────────────────────────────────────────────
# Tasarım: stlf_faz1_scorecard_tasarim.md §1 (K1-K5).
HEADLINE_HORIZON            = "T+2"    # 06_deliver'ın müşteriye teslim ettiği gün (roadmap §1.1)
SCORECARD_REBUILD_WINDOW_DAYS = 400    # her run'da trailing kaç gün yeniden hesaplanır
SCORECARD_WINDOWS           = (7, 30, 365)   # window_report pencereleri
Z_THRESHOLD                 = 3.0      # robust_z alarm eşiği
Z_BASELINE_WINDOW_DAYS       = 30      # median/MAD pencere genişliği
Z_WARMUP_MIN_DAYS           = 30       # bu kadar temiz gün dolmadan 'warmup' modu (mutlak p95 eşik)
ALERTS_DIR                  = LOGS_DIR / "alerts"   # z>3 / kapsam eksikliği -> <target_date>.json

MASTER_PARQUET   = DATA_DIR / "master.parquet"
WEATHER_HISTORY_PARQUET = DATA_DIR / "weather_history.parquet"
WEATHER_FC_PARQUET = DATA_DIR / "weather_cache" / "weather_fc_live.parquet"
FEATURE_MATRIX_PATH = DATA_DIR / "weather_cache" / "feature_matrix.parquet"

# ── Günlük veri kaynağı (OneDrive DD.MM subfolder yapısı) ─────────────────────
LIVE_DATA_DIR = LIVE_DIR.parent.parent.parent / "02_Alınan Veriler" / "gdz-adm live" / "talep"

# ── Hedef kolon (Boray'dan birebir aynı) ─────────────────────────────────────
RAW_TARGET_COL = "ADM_Dağıtılan_Enerji_(MWh)"
RAW_DATE_COL   = "Tarih"
RAW_HOUR_COL   = "Saat"

# ── CV/model parametreleri (Boray config'inden kopyalandı) ────────────────────
TEST_SIZE       = 48       # 48h forecast horizon (T+1=24h + T+2=24h)
WARMUP_PERIOD   = 504      # Lag504h için NaN ısınma
MAX_TRAIN_SIZE  = 22000    # ~2.5 yıl, concept drift kapağı
DATA_START_DATE = None
DATA_END_DATE   = None

# ── DataManager için INPUT_FILE_PATH (çalışma zamanında override edilir) ──────
INPUT_FILE_PATH = str(DATA_DIR / "weather_cache" / "_tmp_combined.xlsx")

# ── Drop listesi (DataManager COLS_TO_DROP) ───────────────────────────────────
COLS_TO_DROP = [
    "After_Bayram", "Haftanin_gunu_Sin", "Haftanin_gunu_Cos",
    "Gun_Sin", "Gun_Cos", "Ay_Sin", "Ay_Cos", "Saat_Sin", "Saat_Cos",
    "Rolling_Mean_3h", "Rolling_Mean_168h", "ÖzelGün_Adı",
]

# ── A-FAMILY FEATURES (A1: momentum, A2: volatility, A3: ratio) ──────────────
ENABLE_A1_FEATURES = True
ENABLE_A2_FEATURES = False
ENABLE_A3_FEATURES = True

# ── Feature toggle'ları ───────────────────────────────────────────────────────
ENABLE_HOLIDAY_LAG_CLEAN = True
ENABLE_THERMAL_FEATURES  = False

# ── Model dosyaları ───────────────────────────────────────────────────────────
MODEL_XGB_PATH      = MODELS_DIR / "live_xgboost.json"
MODEL_LGBM_PATH     = MODELS_DIR / "live_lightgbm.txt"
MODEL_LGBM_WD_SAT   = MODELS_DIR / "live_lightgbm_wd_sat.txt"
MODEL_LGBM_WE       = MODELS_DIR / "live_lightgbm_we.txt"
MODEL_XGB_WD_SAT    = MODELS_DIR / "live_xgboost_wd_sat.json"
MODEL_XGB_WE        = MODELS_DIR / "live_xgboost_we.json"
MODEL_STACKER_PATH  = MODELS_DIR / "stacking_ridge.joblib"

MODEL_NAME          = "live_xgboost.json"

# ── HPO / Training params ─────────────────────────────────────────────────────
HPO_PARAMS_SUFFIX = "_sagemaker_hpo"
ENABLE_WEEKEND_SPLIT_XGB  = True
ENABLE_WEEKEND_SPLIT_LGBM = True
ENABLE_WEEKEND_SPLIT_CAT  = False
ENABLE_GBDT_REFIT         = True
FAST_MODE   = False
FAST_MAX_ITER = 150

# Chronos T+2 lag strategy
CHRONOS_T2_LAG_STRATEGY = "recursive"

# ── Chronos ────────────────────────────────────────────────────────────────────
CHRONOS_ADAPTER_PATH  = str(MODELS_DIR / "chronos_lora_turkforecast")
CHRONOS_MODEL_ID      = "amazon/chronos-2"
CHRONOS_CONTEXT_LENGTH = 2048
CHRONOS_FORCE_CPU     = True
CHRONOS_USE_COVARIATES = True

# ── Post-process artefaktları (donmuş) ────────────────────────────────────────
PV_BIAS_LOOKUP_T1   = MODELS_DIR / "pv_bias_lookup.json"
PV_BIAS_LOOKUP_T2   = MODELS_DIR / "pv_bias_lookup_t2.json"
HOLIDAY_BLEND_ALPHAS    = MODELS_DIR / "holiday_blend_alphas.json"
HOLIDAY_BLEND_ALPHAS_T2 = MODELS_DIR / "holiday_blend_alphas_t2.json"
HOLIDAY_BLEND_ALPHAS_JSON = str(HOLIDAY_BLEND_ALPHAS)
HOLIDAY_BLEND_ALPHAS_T2_JSON = str(HOLIDAY_BLEND_ALPHAS_T2)

# Post-process flag'leri
ENABLE_HOLIDAY_SUBSTITUTION = True
HOLIDAY_SUBSTITUTION_TEMP_COL = "Hissedilen_Sıcaklık_Mean_MUGLA"
ENABLE_PV_BIAS_CORRECTION = True
PV_BIAS_SOLAR_HOURS = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
PV_BIAS_MIN_SAMPLES_PER_CELL = 5
PV_BIAS_FIT_EXCLUDE_HOLIDAYS = True
PV_BIAS_FALLBACK_ENABLED = True

POST_HOLIDAY_MULTIPLIERS_T1 = {"religious_post_1": 1.0540, "religious_post_2_3": 0.9890}
POST_HOLIDAY_MULTIPLIERS_T2 = {"religious_post_1": 1.1370, "religious_post_2_3": 1.0460}

ENABLE_T2_BIAS_CORRECTION  = False
ENABLE_T2_RIDGE_CORRECTION = False
T2_STRATEGY_MODE = "separate"
ADAPTIVE_STRATEGY = "rolling_ridge"

# ── Stacking interaction toggle'ları (devre dışı) ─────────────────────────────
ENABLE_SEASONAL_STACKING_INTERACTIONS = False
ENABLE_HOUR_STACKING_INTERACTIONS     = False

# ── Hava verisi (Open-Meteo Forecast API) ─────────────────────────────────────
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_FC_DAYS = 3    # bugün (T+0, Lag kaynağı) + T+1 + T+2 — 3-step recursion için
WEATHER_TIMEZONE = "Europe/Istanbul"
WEATHER_MODEL = "best_match"  # ICON-seamless fallback dahil

WEATHER_STATIONS = {
    "MUGLA_MenteseCenter":     (37.23520, 28.43968),
    "MUGLA_MilasIndustrial":   (37.31164, 27.78080),
    "MUGLA_YataganIndustrial": (37.33970, 28.14950),
    "MUGLA_SandrasHighAlt":    (37.08125, 28.83792),
    "MUGLA_DalamanPlain":      (36.76700, 28.80000),
    "MUGLA_BodrumCenter":      (37.03647, 27.42547),
    "DENIZLI_Honaz":           (37.68222, 29.28083),
    "DENIZLI_OSB":             (37.80758, 29.24372),
    "DENIZLI_Merkez":          (37.78333, 29.09639),
    "DENIZLI_IsikliCivril":    (38.22806, 29.90333),
    "AYDIN_Merkez":            (37.84806, 27.84528),
    "AYDIN_OSB":               (37.86573, 27.98502),
    "AYDIN_BuyukMenderes":     (37.75196, 27.40567),
    "AYDIN_BozdoganMadran":    (37.64615, 28.24152),
}
WEATHER_GHI_WEIGHTS = {"MUGLA": 0.25, "DENIZLI": 0.40, "AYDIN": 0.35}

# ── Çıktı ──────────────────────────────────────────────────────────────────────
OUTPUT_FILENAME_TEMPLATE = "{date}_forecast.xlsx"

# ── Arşivlenecek / hash'lenecek artefaktlar (Faz -1) ──────────────────────────
# Günlük yeniden eğitilen modeller: her run archive/<run_id>/'a kopyalanır (~7.5 MB).
DAILY_RETRAINED_MODELS = [
    MODEL_XGB_PATH,
    MODEL_LGBM_PATH,
    MODELS_DIR / "live_catboost.cbm",
    MODEL_XGB_WD_SAT,
    MODEL_XGB_WE,
    MODEL_LGBM_WD_SAT,
    MODEL_LGBM_WE,
]
# Donmuş kalibrasyon artefaktları: git-tracked, kopyalanmaz — sadece manifest'e hash'i.
FROZEN_ARTEFACTS = [
    HOLIDAY_BLEND_ALPHAS,
    HOLIDAY_BLEND_ALPHAS_T2,
    PV_BIAS_LOOKUP_T1,
    PV_BIAS_LOOKUP_T2,
    MODELS_DIR / "stacking_ridge.joblib",
    MODELS_DIR / "stacking_ridge_chronos.joblib",
    MODELS_DIR / "stacking_meta_model.joblib",
]
# Donmuş ama büyük (git dışı) — reprodüksiyon için sadece manifest'e hash/mtime.
CHRONOS_ADAPTER_DIR = MODELS_DIR / "chronos_lora_turkforecast"
