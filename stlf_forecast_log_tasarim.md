# STLF İzleme Katmanı — Loglama Şeması Tasarımı (Faz 0)

> Amaç: `stlf_izleme_deney_metodolojisi.md` §1'in `forecast_log` / `actuals_log` / `daily_scorecard`
> şemalarını, **mevcut canlı pipeline'ın gerçekte ürettiği ara çıktılara** dayandırarak somutlaştırmak.
> Bu doküman onaylandıktan sonra kod yazılır. Geri dönüşü zor olan tek şey veri şeması olduğu için önce bunu sabitliyoruz.

---

## 0. Tasarım İlkeleri

1. **Multi-tenant baştan.** Her tabloda `edas_id` kolonu + partition anahtarı. Bugün `ADM`, yarın `GDZ` ve diğer EDAŞ'lar. Şema hiç değişmeden yeni tenant eklenebilir.
2. **Tahmin anında yakala, sonra türet.** Saatlik granülerlik. Meta ağırlık, bias delta, kullanılan hava, config_hash — bunlar tahmin anında yazılmazsa **geri dönülemez kaybolur**. Günlük scorecard bunlardan *hesaplanır*, ayrı loglanmaz.
3. **Gerçek düzeltme adımlarını ayrı ayrı logla.** Metodolojinin jenerik "general + pv" ikilisi yerine pipeline'ın gerçek zinciri: stacking → holiday override → holiday substitution → PV bias. Her birinin per-saat MW katkısı ayrı kolonda.
4. **Fallback'leri açığa çıkar.** Chronos XGB'ye düştüğünde veya CatBoost atlandığında bunu flag'le — yoksa attribution yalan söyler.
5. **Küçük veri, basit depo.** Tek EDAŞ × saatlik = günde 48 satır. Parquet (partition) + tek DuckDB dosyası yeter; ağır MLOps yok.

---

## 1. Depolama & Partition Düzeni

```
logs/
  forecast_log/
    edas_id=ADM/
      target_date=2026-07-04/
        run_2026-07-03.parquet          # o teslim gününü üreten run
  actuals_log/
    edas_id=ADM/
      target_date=2026-07-04.parquet    # D+1/D+2'de doldurulur
  monitoring.duckdb                       # daily_scorecard + view'lar (türetilmiş)
```

- **forecast_log**: `target_date` partition. Bir run 48h = 2 teslim günü (T+1 + T+2) yazar → 2 partition'a düşer, dosya adı `run_<issue_date>` ile çakışma engellenir. Sorgu deseni "X gününde ne oldu" olduğu için partition anahtarı `target_date`.
- **actuals_log**: aynı partition düzeni, ayrı tablo. Gerçekleşme D+1'de gelir, bu yüzden ayrı tablo — forecast_log immutable kalır, actuals sonradan join edilir.
- **daily_scorecard**: DuckDB tablosu. forecast_log ⋈ actuals_log join'inden her gün türetilir. MLflow'a değil buraya yazılır (metodoloji §1 önerisi).
- **Hacim notu:** Multi-tenant büyürse `edas_id` zaten partition; ay bazlı rollup gerekirse `target_date`→`target_month` kolayca genişler.

---

## 2. Tablo: `forecast_log` (saatlik, tahmin anında yazılır)

Grain: bir satır = (`edas_id`, `run_id`, `target_ts`). Aynı `target_ts` iki farklı run'da (T+1 ve T+2 olarak) loglanabilir; `horizon_day` ayırır.

### 2.1 Kimlik & zaman
| Kolon | Tip | Kaynak | Not |
|---|---|---|---|
| `edas_id` | str | config (`EDAS_ID`) | Partition anahtarı. Şimdilik `"ADM"` |
| `run_id` | str | `run_daily` üretir | `<issue_date>_<config_hash8>` |
| `issue_ts` | datetime | `run_daily` başlangıç `datetime.now()` | Lead time için |
| `target_ts` | datetime | `raw_predictions.Datetime` | Tahmin edilen saat |
| `horizon_day` | str | `04` is_t2 maskesi (`TEST_SIZE//2`) | `"T+1"` / `"T+2"` |
| `lead_time_h` | int | `target_ts − issue_ts` (saat) | Türetilir |

### 2.2 Alt-model ham tahminleri
| Kolon | Tip | Kaynak | Not |
|---|---|---|---|
| `y_pred_xgb` | float | `raw_predictions.XGB_Pred` | |
| `y_pred_lgbm` | float | `raw_predictions.LGBM_Pred` | |
| `y_pred_cat` | float? | `raw_predictions.CAT_Pred` | **Nullable** — CatBoost atlanabilir (`04:248`) |
| `y_pred_chronos` | float? | `raw_predictions.CHRONOS_Pred` | Aşağıdaki `chronos_ok` ile birlikte oku |
| `cat_present` | bool | `04` — CAT_Pred üretildi mi | |
| `chronos_ok` | bool | `04` — Chronos gerçekten koştu mu | **Kritik:** False ise `y_pred_chronos` aslında XGB kopyası (`04:400`). Bunu loglamazsan `single_model_failure` tespit edilemez |

### 2.3 Stacking / meta katman
| Kolon | Tip | Kaynak | Not |
|---|---|---|---|
| `y_pred_ens_raw` | float | `04` stacking çıktısı (holiday override & postproc ÖNCESİ) | Metodolojideki `y_pred_ens_raw`. **Not:** bugün `raw_predictions.Ensemble_Pred` override *sonrası* yazılıyor — override öncesini de tutmak için `04`'te snapshot gerekli (bkz. §5) |
| `meta_method` | str | `stack_predictions` | `rolling_ridge` / `frozen_ridge` / `simple_mean` |
| `meta_w_xgb` | float? | Ridge `.coef_` | Nullable — `simple_mean`'de eşit ağırlık, non-lineer'de boş |
| `meta_w_lgbm` | float? | Ridge `.coef_` | |
| `meta_w_cat` | float? | Ridge `.coef_` | |
| `meta_w_chronos` | float? | Ridge `.coef_` | |
| `meta_intercept` | float? | Ridge `.intercept_` | |

### 2.4 Düzeltme zinciri (her adımın per-saat MW katkısı)
Zincir: `y_pred_ens_raw` → (+override) → (+substitution) → (+pv_bias) = `y_pred_final`

| Kolon | Tip | Kaynak | Not |
|---|---|---|---|
| `override_active` | bool | `04:_apply_holiday_override` | Hafta içi tatilde CatBoost solo devreye girdi mi |
| `override_delta` | float | override sonrası − öncesi | 0 = devreye girmedi |
| `subst_active` | bool | `05:apply_holiday_substitution` | Bu saatte substitution uygulandı mı |
| `subst_source_date` | str? | `05` sub_stats | Hangi kaynak gün kullanıldı (triyaj Adım 4d) |
| `subst_delta` | float | substitution sonrası − öncesi | |
| `pv_bias_delta` | float | `05:apply_pv_bias` sonrası − öncesi | PV corrector'ın eklediği/çıkardığı MW (triyaj Adım 4a) |
| `y_pred_final` | float | `postprocessed_predictions.Final_Pred` | Müşteriye giden |

### 2.5 Kullanılan dış girdiler (tahmin anındaki hava — reanalysis DEĞİL)
> **Doğrulandı:** `03_build_features`, gelecek satırlarda forecast havayı `_fc` değil **`_actual`-adlı kolonlara** yazıyor (uniform şema). Bu yüzden tahmin anında `feature_matrix` gelecek satırlarındaki hava = *kullanılan forecast*'tır. `wx_*_fcst`'i buradan yakala.

| Kolon | Tip | Kaynak | Not |
|---|---|---|---|
| `wx_temp_fcst` | float | `feature_matrix`'teki 14 istasyon `*_app_temp_actual` (gelecek satır = forecast) ortalaması @ predict_idx | `wx_temp_actual` ile **aynı** aggregasyon (apples-to-apples) |
| `wx_ghi_fcst` | float | `feature_matrix["GHI_ADM_Weighted"]` @ predict_idx | Kullanılan ışınım tahmini |

### 2.6 Takvim & segment
| Kolon | Tip | Kaynak | Not |
|---|---|---|---|
| `day_type` | str | `feature_matrix` takvim kolonları → türetilir | `hafta_ici`/`cmt`/`paz`/`resmi`/`dini`/`kopru` |
| `flag_holiday` | bool | `feature_matrix` (`Yilbasi`/`Milli_Bayram`/`Ramazan_Bayram`/`Kurban_Bayram`) | `04:350` aynı kolonları kullanıyor |
| `flag_bridge` | bool | takvim | Köprü günü |
| `flag_ramadan` | bool | takvim | Ramazan ayı |

> **Doğrulanacak:** `feature_matrix`'teki tam takvim kolon adları implementasyonda teyit edilecek (`03_build_features` çıktısı). Şema sabit, mapping esnek.

### 2.7 Sürüm & tekrarlanabilirlik
| Kolon | Tip | Kaynak | Not |
|---|---|---|---|
| `config_hash` | str | `config_live` içeriğinin hash'i | Changepoint analizinde config değişimi işareti (§5.3) |
| `model_versions` | str (json) | `models/` artefakt mtime/hash'leri | Hangi model dosyaları canlıydı |
| `feature_snapshot_ref` | str | O run'ın `feature_matrix.parquet` arşiv yolu | Debug için (metodoloji §1.1) |

---

## 3. Tablo: `actuals_log` (D+1/D+2'de doldurulur)

Grain: bir satır = (`edas_id`, `target_ts`).

| Kolon | Tip | Kaynak | Not |
|---|---|---|---|
| `edas_id` | str | config | Partition |
| `target_ts` | datetime | `master.parquet` Tarih+Saat | Join anahtarı |
| `y_actual` | float | `master[RAW_TARGET_COL]` | Gerçekleşen yük. **D+1**'de gelir |
| `wx_temp_actual` | float? | `weather_history` 14 istasyon `*_app_temp_actual` ortalaması | `wx_temp_fcst` ile aynı aggregasyon. **~D+6** gecikmeyle dolar |
| `wx_ghi_actual` | float? | `weather_history["GHI_ADM_Weighted"]` | Fcst ile aynı ad → bire bir eşleşir. **~D+6** gecikme |
| `data_quality_flag` | str? | `01_ingest_actual` türetir | Eksik/gecikmeli/interpolasyon/imkânsız değer (negatif/spike/düz çizgi) |
| `known_event` | str? | Manuel giriş — başta `known_events.csv`, sonra UI form | Dış şok: arıza, kesinti, büyük müşteri, etkinlik. Serbest metin + kategori kodu |

> **Gecikme gerçeği:** Yük gerçekleşmesi D+1'de, **hava gerçekleşmesi ~D+6**'da (`weather_history` reanalysis gecikmesi) gelir. Sonuç: scorecard yük hatası (MAPE/ME) D+1'de hazır; **hava-attribution ve perfect-prog rerun ~D+6'da** tamamlanır. Bu yüzden actuals_log iki dalgada dolar (`y_actual` erken, `wx_*_actual` geç).
>
> `y_actual` zaten `oof_history.parquet`'te var — actuals_log onu genişletir. oof_history stacker eğitimi için kalır; actuals_log izlemenin kanonik kaynağı olur.

---

## 4. Tablo: `daily_scorecard` (türetilmiş, DuckDB — Faz 1'de dolar)

forecast_log ⋈ actuals_log join'inden günlük hesaplanır. Şemayı şimdiden sabitliyoruz; hesap Faz 1.

| Kolon | Açıklama |
|---|---|
| `edas_id`, `target_date` | Anahtar |
| `mape`, `wape`, `rmse`, `me` | Günlük hata metrikleri (`me` = işaretli bias) |
| `max_ape_hour` | En kötü saat |
| `mape_xgb..chronos`, `mape_ens_raw`, `mape_final` | Bileşen attribution — "corrector kaç bps kazandırdı" doğrudan okunur |
| `mape_night`/`mape_morning`/`mape_pv`/`mape_evening` | Saat-blok (00-06/06-10/10-16/17-22) |
| `temp_fcst_error`, `ghi_fcst_error` | Hava tahmin hatası (actuals_log dolunca) |
| `verdict_code` | Triyaj çıktısı (Faz 2): `DATA_QUALITY`/`WEATHER_DRIVEN`/... |
| `robust_z` | (MAPE − median_30d)/(1.4826·MAD_30d) |
| `n_hours`, `data_quality_flag_count` | Kapsam & sağlık |

---

## 5. Pipeline Değişiklik Yüzeyi (onaydan sonra yapılacak)

| Dosya | Değişiklik | Boyut |
|---|---|---|
| `config_live.py` | `EDAS_ID`, `FORECAST_LOG_DIR`, `ACTUALS_LOG_DIR`, `MONITORING_DB` sabitleri | Küçük |
| `src/forecast_logger.py` (yeni) | `write_forecast_log()`, `update_actuals_log()`, `build_daily_scorecard()` | Ana iş |
| `04_predict_48h.py` | (a) override *öncesi* ensemble snapshot; (b) `stack_predictions` meta ağırlık/method döndürsün; (c) `chronos_ok`/`cat_present` flag'leri result'a | Orta |
| `05_postprocess.py` | Substitution & PV aşamaları arası `preds` snapshot → `override_delta`/`subst_delta`/`pv_bias_delta`/`subst_source_date` kolonlarını postprocessed parquet'e yaz | Orta |
| `run_daily.py` | `run_id`/`config_hash` üret, aşağı geçir; `05` sonrası `write_forecast_log()`; `01` sonrası `update_actuals_log()` çağır | Küçük |
| `01_ingest_actual.py` | `data_quality_flag` türetimi (eksik/spike/düz çizgi kontrolü) | Küçük |

Prensip: matematik `04`/`05`'te kalır (delta hesabı = iki snapshot farkı), IO ise `forecast_logger.py`'de toplanır. `04`/`05` sadece ara çıktıyı parquet'e ek kolon olarak bırakır.

---

## 6. Kilitlenmiş Kararlar (2026-07-04 onaylı)

1. **Her adım loglanır — kazanç/kayıp nerede olursa görünür.** `y_pred_ens_raw` = holiday override *öncesi* saf stacking çıktısı. Zincirin her adımı (`override_delta` → `subst_delta` → `pv_bias_delta`) ayrı kolon; hangi düzeltme kaç MW ekledi/çıkardı ve o gün hata hangi yöndeydi → doğrudan okunur. ✅
2. **`known_event`:** başta elle düzenlenen `known_events.csv` (`edas_id, target_ts_start, target_ts_end, kategori, not`), Faz 3'te UI form. ✅
3. **Hava gerçekleşmesi:** `weather_history.parquet` **gerçekleşmeyi tutuyor** (`_actual` kolonlar, ~D+6 gecikme). Forecast ise `feature_matrix`'in gelecek satırlarında (`_actual`-adlı kolonlara yazılmış). İkisi apples-to-apples → **perfect-prog rerun mümkün.** ✅
4. **DuckDB.** Parquet'i doğrudan sorgular; scorecard join'i tek SQL. ✅
5. **Günlük partition** (`target_date`). Hacim küçük, sorgu deseni günlük. ✅

---

## 7. Sonraki Adım

Bu doküman onaylanınca sıra:
1. `config_live.py` sabitleri + `src/forecast_logger.py` iskeleti + şema (pyarrow schema tanımı).
2. `04`/`05` snapshot & flag eklemeleri.
3. `run_daily.py` entegrasyonu — bir sonraki canlı run'da forecast_log dolmaya başlar.
4. Geçmiş `output/archive/*` dosyalarından **kısmi backfill** (elde olan kolonlar kadar — meta ağırlık/delta geçmişe dönük yok, ama y_pred'ler ve actual'lar backfill edilebilir).
