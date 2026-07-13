# STLF LIVE ADM/GDZ — Agent Reference

## 1. Project Overview

ADM ve GDZ bolgeleri icin **48 saat ileri** (T+1 + T+2) saatlik dagitilan enerji tahmini.
Her sabah (veya istenildiginde) calisan 9-adimli pipeline:
**Ingest → Weather → Features → Predict → PostProcess → Deliver → Excel → HTML → Email**

- **EDAS**: ADM (Aydem), GDZ (Gediz) — iki paralel proje
- **Horizon**: T+1 (24h) + T+2 (24h) = 48 saat
- **Modeller**: XGBoost, LightGBM, CatBoost, Chronos-2 → Ridge ensemble
- **Output**: `output/YYYY-MM-DD_forecast.xlsx` (24 satir, T+2 teslim)
- **Dil**: Python 3.11, pandas/numpy, openpyxl, Streamlit (UI)

---

## 2. CRITICAL RULES

### NEVER do these without explicit instruction:

1. **NEVER run `04_predict_48h.py` standalone** — overwrites production model files.
   Use `python asof_regen.py` for backtesting (backup/restore pattern).

2. **NEVER modify `master.parquet`** without first backing up.
   The file is overwritten atomically now (`.tmp` + rename), but still:
   backup = `master.parquet.bak` before any bulk operation.

3. **NEVER delete `weather_history.parquet`** — it contains Archive API backfill
   data that takes 30+ minutes to regenerate (14 stations × 8 days).

4. **NEVER hardcode paths** with usernames. GDZ path uses
   `ROOT.parent.parent / "cagatay" / "gdz talep" / "live"`. This is fragile —
   the `07_report_excel.py` already has this issue.

### ALWAYS do these:

1. **Import pattern**: `import config_live as C` then `C.CONSTANT_NAME`.
2. **Pipeline step return**: `{"status": "ok", ...}` or `{"status": "error", "err": "..."}`.
3. **Logging**: Use `logging.getLogger("adm_live")`, NOT `print()` in src/ modules.
4. **Error wrapping**: Steps 07-09 (report, diagnostic, email) MUST be in try/except
   — pipeline devami icin.
5. **MAPE**: Always import from `src.metrics.calculate_mape()` — 3 duplicate copies
   were removed, don't re-add.

---

## 3. File Map (52 Python files)

### config & entry
```
config_live.py          — ALL constants (276 lines). Single source of truth.
run_daily.py            — Main orchestrator (183 lines). 9 pipeline steps + logging.
```

### pipeline/ (9 steps)
```
01_ingest_actual.py     — master.parquet upsert. -> data/master.parquet
02_fetch_weather.py     — Open-Meteo forecast + archive backfill. -> weather_fc_live.parquet
03_build_features.py    — Feature matrix. DataManager + holiday lag. -> feature_matrix.parquet
04_predict_48h.py       — 4 models + Ridge stacking. -> raw_predictions.parquet
05_postprocess.py       — Holiday sub + PV bias + bias correction. -> postprocessed.parquet
06_deliver.py           — Sanity check + Excel output. -> YYYY-MM-DD_forecast.xlsx
07_report_excel.py      — STLF_LIVE_RAPOR.xlsx (5 tables, ADM+GDZ). Auto append.
08_diagnostic_html.py   — diagnostic_YYYY-MM-DD.html (Chart.js, 7 tabs). Thin shell:
                          maps ADM cols to canonical names, calls src/diagnostic_core.py.
                          GDZ's own 08_diagnostic_html.py imports the SAME core module
                          (sys.path trick) — never edit tabs/JS in this file directly,
                          edit diagnostic_core.py so both EDAS stay identical.
09_email_report.py      — Email report (SMTP env var). Skipped if not configured.
```

### src/ (model managers & helpers)
```
model_manager.py        — XGBoost (HybridXGBModel). Weekend split. GBDT refit.
lightgbm_manager.py     — LightGBM (HybridLightGBMModel). Same split.
catboost_manager.py     — CatBoost (single model). cat_features detected.
chronos_manager.py      — Amazon Chronos-2 LoRA adapter. GPU/CPU toggle.
chronos_bridge.py        — Panel preparation for Chronos.
data_manager.py         — Boray legacy. Feature engineering + holiday lag clean.
stacking_manager.py     — (1369 lines) 9 stacking strategies. Leak-safe expanding window.
adaptive_stacking.py    — Leak-safe adaptive stacking. Event-window switch.
oof_feedback.py         — OOF history + Chronos fallback detection.
forecast_logger.py      — Faz 0: forecast/actual parquet log + DuckDB views.
scorecard.py            — Faz 1: daily MAPE, z-score alerts.
diagnostic_core.py      — Shared ADM+GDZ diagnostic engine (compute() + render()).
                          GDZ imports this file directly (path: ../adm live/src).
                          Single source of truth for the interactive HTML — do not
                          fork per-EDAS logic here, add canonical-column mapping in
                          the wrapper (pipeline/08_diagnostic_html.py) instead.
run_context.py          — Run identity, model archive, config hash.
holiday_calendar.py     — Turkish holidays 2018-2026. Manual update needed.
holiday_substitution.py — (789 lines) Holiday blend alpha + profile substitution.
holiday_lag_clean.py    — Holiday-aware lag cleaning.
pv_bias_correction.py   — Solar PV bias correction lookup table.
recency_weight.py       — Exponential decay sample weights.
smart_features.py       — Profile-based lag features.
thermal_features.py     — Thermal inertia features (A/B/C groups).
metrics.py              — Canonical calculate_mape().
common.py               — Shared utilities (add_local_src_path).
data_scanner.py         — OneDrive CSV scanner (YYYY.MM/DD; DD.MM legacy fallback).
t2_post_process.py      — T+2 Ridge correction (currently disabled).
```

### ui/ (Streamlit Dashboard)
```
dashboard.py            — Entry point. 3 tabs.
common.py               — Shared UI components. GDZ subprocess pattern.
tab_veri_durumu.py      — Data status tab.
tab_veri_yukleme.py     — Data upload tab.
tab_tahmin_uret.py      — Forecast generation tab (mirrors run_daily.py).
tab_izleme.py           — Monitoring tab.
```

### Root scripts (manual tools)
```
asof_regen.py           — As-of backtest engine. Backup/restore safety.
backtest_30d.py         — 30-day backtest (calls asof_regen).
backtest_7d.py          — 7-day T+2 backtest.
backtest_tomorrow.py    — T+1 delivered day backtest.
backtest_walkforward.py — Walk-forward with forecast log backfill.
analyze_models_30d.py   — Per-model error analysis (MAPE, bias, hour breakdown).
fix_weather_history.py  — Archive API weather gap backfill.
export_hourly_mape_7d.py— DuckDB -> Excel export.
perfect_prog_rerun.py   — Perfect-prog weather rerun.
backfill_logs.py        — Forecast log backfill from archive.
```

### tests/ (pipeline koruma testleri — yeni)
```
tests/test_pipeline_core.py — 21 test: split, lag, ensemble, bias, csv, config, recency, dropna.
```
- ADM: `cd [adm live] && pytest tests/ -v` (21 test)
- GDZ: `cd [gdz live] && pytest tests/ -v` (14 test)
- Pipeline degisikligi öncesi/sonrasi test PASS olmali

---

## 4. Data Flow Diagram

```
[OneDrive CSV] → 01_ingest → master.parquet (atomic write)
[Open-Meteo]   → 02_weather → weather_fc_live.parquet
                                    ↘
                              weather_history.parquet (Archive backfill)
                                    ↙
master.parquet ─→ 03_features → feature_matrix.parquet
weather_history.parquet ──↗           ↕
weather_fc_live.parquet ──↗    DataManager (legacy) + NaN guard
                                    ↓
feature_matrix.parquet → 04_predict → raw_predictions.parquet
    ↕         ↕          ↕          ↕
  XGBoost   LightGBM   CatBoost   Chronos-2
    ↕         ↕          ↕          ↕
                         Ridge Ensemble
                              ↓
raw_predictions → 05_postprocess → postprocessed_predictions.parquet
postprocessed → 06_deliver → YYYY-MM-DD_forecast.xlsx (T+2)
postprocessed → 07_report → STLF_LIVE_RAPOR.xlsx (5 tables, ADM+GDZ)
postprocessed → 08_diagnostic → diagnostic_YYYY-MM-DD.html (Chart.js)
                                → 09_email → emre.hangul@mrc-tr.com
                                              cagatay.bayrak@mrc-tr.com
```

### Data Safety Rules
- **master.parquet**: Atomic write via `.tmp` → rename. Has `.bak` copies.
- **weather_history.parquet**: Never delete. Fix via `fix_weather_history.py`.
- **Model files** (`models/live_*`): Replaced daily. Archived in `models/archive/<run_id>/`.
- **Output files**: Git-ignored. Stored in OneDrive-synced `output/`.

---

## 5. Key Configuration (`config_live.py`)

### Paths
```python
LIVE_DIR = Path(__file__).parent                     # Proje root
DATA_DIR = LIVE_DIR / "data"                          # master.parquet, weather_history
OUTPUT_DIR = LIVE_DIR / "output"                      # forecast.xlsx, reports
MODELS_DIR = LIVE_DIR / "models"                      # live_*.json/.txt/.cbm
```

### Model Toggles
```python
ENABLE_WEEKEND_SPLIT_XGB  = True  # Aydem A.S. split (Mon-Sat vs Sat-Sun)
ENABLE_WEEKEND_SPLIT_LGBM = True
ENABLE_WEEKEND_SPLIT_CAT  = False # Single model (weekend split hurts CAT)
ENABLE_RECENCY_WEIGHTING  = True  # Exponential decay with halflife=60 days
ENABLE_GBDT_REFIT         = True  # Early stop -> 100% data refit
```

### Ensemble (ADM)
```python
CALIBRATED_ENSEMBLE_WEIGHTS = {"XGB_Pred": 0.40, "LGBM_Pred": 0.10, "CAT_Pred": 0.05, "CHRONOS_Pred": 0.45}
ENSEMBLE_BIAS_CORRECTION_T1_MWH = 10  # T+1: +10 MWh
ENSEMBLE_BIAS_CORRECTION_T2_MWH = 15  # T+2: +15 MWh
# Weekend/sunday bias scaled: cumartesi T+2*0.20, pazar T+2*0.50
# OOF birikince Rolling Ridge (Tier 1), yoksa Inverse-MAPE (Tier 1b),
# sonra kalibre statik agirlik (Tier 2), frozen stacker (3), mean (4).
```

### CAT HPO
```python
# File: best_params_cat_general_sagemaker_hpo.json
# CRITICAL: l2_leaf_reg=3, min_data_in_leaf=1, loss_function=RMSE, iterations=600
# Previously: l2_leaf_reg=45 (over-regularized), loss=MAE, iter=350
```

---

## 6. Common Pitfalls & Fixes

### P1: weather_history NaN -> 03_build_features crash
**Symptom**: `CRITICAL: XXXX NaN cells in training data` in step 03.
**Root cause**: `02_fetch_weather._update_weather_history()` was using
`keep="last"` on concat, overwriting clean `_actual` columns with NaN
from forecast `_fc` columns (column names differ).
**Fix**: Run `python fix_weather_history.py` (fills 14 stations from Archive API).
Then clear cache: `Remove-Item data/weather_cache/_tmp_combined.*`.
**Permanent fix**: Applied (commit 63ac1b5, 00b9bc8) — only append new rows.

### P2: master.parquet corrupted/truncated
**Symptom**: Dashboard shows fewer days than expected.
**Root cause**: Backtest script (`backtest_30d.py`) timeout before `_restore()`.
**Fix**: Restore from backup:
```python
import shutil; shutil.copy2("data/master.parquet.bak", "data/master.parquet")
```
Or rebuild from OneDrive:
```python
python run_daily.py --skip-weather --skip-ingest --dry-run  # check
python run_daily.py                                          # full pipeline
```

### P3: Chronos crash -> XGB double-counting
**Symptom**: Ensemble suspiciously close to XGB, `chronos_ok=False` in log.
**Root cause**: `04_predict_48h.py` fallback: `gbdt_preds["CHRONOS_Pred"] = gbdt_preds["XGB_Pred"]`.
**Fix**: Check `chronos_ok` in run summary. If false, the ensemble has XGB double-weighted.
Chronos requires ~4GB RAM. If crashes persist, disable Chronos and rebalance weights.

### P4: GDZ forecast not appearing in report
**Symptom**: Only ADM sheet in STLF_LIVE_RAPOR.xlsx.
**Root cause**: Hardcoded path `ROOT.parent.parent / "cagatay" / "gdz talep" / "live"`.
**Fix**: The path depends on the developer's username — verify the GDZ project location.

### P5: 03_build_features NaN guard fails after data changes
**Symptom**: `NaN cells in training data` even after fix_weather_history.
**Root cause**: `_suppress_dropna()` monkey-patch was previously a full `noop` —
it preserved ALL NaN rows including training rows. Current `_smart_dropna_patch()` correctly
drops training NaN but keeps forecast rows. If new feature engineering introduces NaN,
this fails.
**Diagnostic**: Check `feature_df.loc[train_mask].isna().sum().sum()` after step 03.

### P6: GDZ ensemble weights vs actual skill misalignment
**Symptom**: GDZ under-prediction continues even after bias correction.
**Root cause**: GDZ ensemble weights ({LGBM: 0.27, XGB: 0.27, CAT: 0.24, CHRONOS: 0.22}) are
from 90-day backtest. If recent model skill has shifted, rolling inverse-MAPE becomes stale
until enough live log data accumulates (`ROLLING_WEIGHT_MIN_ROWS = 14*24`).
**Fix**: Check `src/ensemble_weights.py` weight source in log. If "default" persists beyond
14 days, verify forecast_log population. Reset: `Remove-Item $env:LOCALAPPDATA/gdz_live_logs/monitoring.duckdb`.

### P7: ADM Inverse-MAPE activates before Rolling Ridge
**Symptom**: Log shows "Inverse-MAPE adaptive" but no "Rolling Ridge".
**Root cause**: Rolling Ridge requires `ROLLING_RIDGE_MIN_SAMPLES=168` clean OOF samples.
Until then, Tier 1b (inverse-MAPE) runs. This is by design — adaptive but more conservative.
**Diagnostic**: Check OOF history size: `len(pd.read_parquet('data/oof_history.parquet'))`.

---

## 7. Backtest & Analysis Commands

```bash
# 30-day backtest (as-of, perfect-prog weather)
python backtest_30d.py                   # 2026-06-06 .. 2026-07-05
python backtest_30d.py 2026-06-01 2026-06-15   # specific range

# Model analysis (after backtest files exist)
python analyze_models_30d.py

# Fix weather gaps
python fix_weather_history.py

# Standalone pipeline steps
python pipeline/03_build_features.py      # requires cache
python pipeline/06_deliver.py             # requires postprocessed

# Full pipeline
python run_daily.py                       # normal
python run_daily.py --skip-weather        # use cached weather
python run_daily.py --skip-ingest         # skip data ingestion
python run_daily.py --dry-run             # steps 01-03 only

# Unit tests (ADM — 21 test)
python -m pytest tests/ -v

# Unit tests (GDZ — 14 test)
python -m pytest ../"gdz talep/live/tests"/ -v
```

---

## 8. GDZ Parallel Project

- **Location**: `../gdz talep/live/` relative to this repo
- **Config**: `config_live_gdz.py` (GDZ_MASTER_PARQUET, own constants)
- **Station naming**: IZMIR_*, MANISA_* (not MUGLA/DENIZLI/AYDIN)
- **GDZ improvements (2026-07-10)**:
  - **Bias correction**: Added ensemble bias correction (haftaici T1=+10, T2=+15 MWh,
    cumartesi 0.30x, pazar 0.00x). ADM'den port edildi — ME=-42 MWh under-prediction fix'i.
  - **Recency weighting**: `ENABLE_RECENCY_WEIGHTING=True`, halflife=60g. GBDT eğitiminde
    sample_weight olarak uygulanır (ADM parity).
  - **Tests**: `tests/test_gdz_core.py` — 14 test (bias, config, recency, postprocess).
- **Integration points**:
  - `07_report_excel.py` — tries to import config_live_gdz dynamically
  - `ui/common.py` — runs GDZ pipeline via subprocess (sys.modules collision avoidance)
  - `klima_analizi.py` — reads both ADM and GDZ masters
- **Critical**: When importing GDZ modules in the same Python process as ADM,
  `sys.modules` collision occurs (both have `run_context`, `forecast_logger`).
  Use `subprocess` for cross-project calls.

---

## 9. Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| CAT MAPE ~5.4% (35g backtest) vs Chronos 4.5% (best) | MEDIUM | WATCH (CAT weight=0.05 korunuyor; segment edge var) |
| GDZ model under-prediction (~10% MAPE) | HIGH | FIXED (bias correction eklendi 2026-07-10) |
| Chronos session crash -> XGB double-count | MEDIUM | WORKAROUND (rolling inverse-MAPE + renormalize fix) |
| Holiday calendar needs 2027+ entries | MEDIUM | TODO (manual update in holiday_calendar.py) |
| Hardcoded GDZ paths with username | LOW | KNOWN (07_report_excel.py) |
| No unit tests (0 test files) | DONE | 35 test (ADM 21 + GDZ 14) |
| stacking_manager.py 1369 lines | LOW | REFACTOR CANDIDATE |
| GDZ holiday/PV post-process kapali | MEDIUM | TODO (lookup tablolari gerekli) |

---

## 10. Git History (recent commits)

```
920351d feat: Email rapor otomasyonu (SMTP template) — Faz D
cb6e79c feat: P95 prediction intervals + Cloud cross-check + Scenario engine (Faz C)
ef40cf5 feat: STLF DIAGNOSTIC HTML — interaktif Chart.js dashboard (Faz B)
c30126f feat: STLF LIVE RAPOR — 5 tablolu otomatik Excel (Faz A)
6265b35 fix: master bayat hava kolonu düsürme + holiday flag NaN fill
2dcfa4a fix: _suppress_dropna forecast row korumasi — training NaN temizligini engelliyordu
00b9bc8 fix: numpy.datetime64 has no .date() — pd.Timestamp(d).date()
63ac1b5 fix: _update_weather_history NaN bug — keep=last temiz datayi eziyordu
49f9452 fix: CAT parametre düzeltme — l2_leaf_reg 45→3, min_data_in_leaf 65→1, loss MAE→RMSE
a697ce0 fix: 3 paket iyilestirme — bias day-aware, LGBM Sunday boost, CAT cat_features
87e8fe1 fix: optimize ensemble weights + bias correction (MAPE 7-12% -> 2.0%)
73ba409 Faz 2: rampa-donemi bug fix paketi
```

---

## 11. Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `STLF_SMTP_HOST` | SMTP server for email reports | (none, skip if missing) |
| `STLF_SMTP_PORT` | SMTP port | 587 |
| `STLF_SMTP_USER` | SMTP username | (none, skip if missing) |
| `STLF_SMTP_PASS` | SMTP password | (none, skip if missing) |
| `LOCALAPPDATA` | Log storage path (Windows) | %LOCALAPPDATA%/adm_live_logs |
| `OE_FULL_STRENGTH` | Bypass FAST_MODE | (optional) |
