# STLF Faz 1 — `daily_scorecard` Tasarımı

> Kapsam: [STLF_LIVE_OPS_ROADMAP.md](STLF_LIVE_OPS_ROADMAP.md) §3 Faz 1. Girdi: Faz 0'ın ürettiği
> `forecast_log` / `actuals_log` (bkz. [stlf_forecast_log_tasarim.md](stlf_forecast_log_tasarim.md), kod:
> `src/forecast_logger.py`, entegre: `run_daily.py:117-123`, `pipeline/04_predict_48h.py`,
> `pipeline/05_postprocess.py`). Faz 0 kod tarafı bitmiş durumda (04/05 snapshot/delta kolonlarını
> zaten yazıyor) — Faz 1 bunun üzerine türetilmiş katmanı kurar.

---

## 1. Kilitlenmiş Kararlar (2026-07-05)

**K1 — Headline horizon = T+2.** `06_deliver` müşteriye T+2 gününü teslim ediyor (roadmap §1.1);
alarm ve çoklu-pencere raporu T+2 üzerinden hesaplanır. T+1 de aynı şemada hesaplanıp saklanır
(ikincil sinyal — T+1→T+2 dejenerasyonunu görünür kılmak için), ama tetikleyici headline değildir.

**K2 — `daily_scorecard` materialize DuckDB tablosu**, view değil. Her run sonrası trailing pencere
(`SCORECARD_REBUILD_WINDOW_DAYS`, öntanımlı 400 gün) `CREATE OR REPLACE TABLE` ile yeniden hesaplanır
→ **tamamen türetilmiş, silinip yeniden kurulabilir** (Faz 0'ın `monitoring.duckdb` ilkesiyle tutarlı).
Manuel `verdict_code` (Faz 2) ayrı küçük bir CSV/tabloda tutulur ve join'le eklenir — rebuild onu asla
silmez.

**K3 — ME işaret konvansiyonu:** `me = mean(y_pred_final − y_actual)`. **Pozitif = fazla tahmin
(over-forecast).** Faz 4 CUSUM bu konvansiyonu birebir devralır — bir daha değişmez.

**K4 — WAPE headline'da MAPE'nin yanında:** `wape = Σ|pred−actual| / Σ|actual|`. Düşük yük gece
saatlerinde MAPE'nin şiştiği durumlarda WAPE daha güvenilir okunur (Kaynakça #6).

**K5 — Alarm kanalı v1 = dosya + scorecard kolonu.** `logs/alerts/<target_date>.json` (küçük hacim,
OneDrive altında sorun değil — forecast_log/actuals_log'un aksine append-heavy değil) + scorecard'da
`alert_flag`. E-posta/Teams webhook Faz 3 dashboard'una ertelendi (roadmap Açık Soru #3 hâlâ açık,
ama v1 kanalı böylece devrede).

---

## 2. `daily_scorecard` Şeması

Grain: `(edas_id, target_date, horizon_day)`.

| Kolon | Açıklama |
|---|---|
| `edas_id`, `target_date`, `horizon_day` | Anahtar |
| `n_hours` | Eşleşen saat sayısı (beklenen: 24) |
| `mape`, `wape`, `rmse`, `me` | Headline hata metrikleri (final tahmin, K3/K4) |
| `max_ape_hour`, `max_ape_value` | En kötü saat |
| `mape_xgb`, `mape_lgbm`, `mape_cat`, `mape_chronos` | Alt-model attribution (backfill'lenmiş günlerde NULL — meta/delta geriye dönük yok) |
| `mape_ens_raw`, `mape_final` | Corrector öncesi/sonrası — corrector net kazancı `mape_ens_raw − mape_final` (bps) |
| `mape_night`/`mape_morning`/`mape_pv`/`mape_evening` | Saat-blok: 00-06 / 06-10 / 10-16 / 17-22 |
| `temp_fcst_error`, `ghi_fcst_error` | `mean(wx_*_fcst − wx_*_actual)` — actuals_log'un hava dalgası (~D+6) dolunca hesaplanır, önce NULL |
| `robust_z` | `(mape − median_30d) / (1.4826·MAD_30d)`, tatil-hariç baseline |
| `baseline_mode` | `warmup` (< `Z_WARMUP_MIN_DAYS` temiz gün) / `robust` |
| `alert_flag` | robust modda `robust_z > Z_THRESHOLD`; warmup modda `mape > p95(son 60g)` |
| `verdict_code` | Faz 2 — şimdilik NULL, ayrı tablodan join |
| `data_quality_flag_count`, `known_event_present` | Sağlık / dış-şok işareti |
| `built_at`, `actuals_wave` | `load_only` (D+1) / `complete` (D+6, hava da dolu) |

`flag_holiday` baseline partition anahtarına dahildir (`edas_id, horizon_day, flag_holiday`) — tatil
günleri kendi seyrek baseline'ında kalır, hafta-içi z-score'u kirletmez.

---

## 3. Hesap Akışı (`src/scorecard.py`)

```
build_daily_scorecard(window_days=400)
  1. forecast_log_v ⋈ actuals_log_v  (edas_id, target_ts eşleşmesi, INNER — y_actual dolu olmalı)
  2. Saatlik → günlük agg (pandas groupby: edas_id, target_date, horizon_day)
  3. robust_z: pandas rolling(30) median/MAD, gruplanmış (edas_id, horizon_day, flag_holiday),
     .shift(1) ile bugünü kendi baseline'ından hariç tutarak
  4. baseline_mode + alert_flag hesap
  5. verdict tablosu varsa LEFT JOIN (manuel kod korunur)
  6. CREATE OR REPLACE TABLE daily_scorecard AS SELECT * FROM df  (DuckDB'ye yaz)

window_report(windows=(7,30,365))    -> {7: {...}, 30: {...}, 365: {...}} agregatlar (headline horizon)
check_alerts(z_threshold=3.0)         -> son target_date'in alert_flag=true satırları -> logs/alerts/<date>.json
latest_scorecard(edas_id, horizon)    -> pd.DataFrame (dashboard/CLI için)
```

Matematik pandas'ta (rolling median/MAD DuckDB'de daha kırılgan); IO DuckDB'de. Join iki view
üzerinden `duckdb.sql(...).df()` ile tek sorguda.

---

## 4. Perfect-Prog Rerun (v1, manuel tetiklenen)

Amaç: kötü günde hata **model mi meteoroloji mi** ayrımı (GEFCom geleneği, roadmap Kaynakça #8).

**Güvenlik kısıtı (kritik):** `04_predict_48h.run()` GBDT modellerini **her çağrıldığında yeniden
eğitip canlı model dosyalarının üzerine yazıyor** (`MODEL_XGB_PATH` vb., `04:236,242,251`). Perfect-prog
rerun bu fonksiyonu doğrudan çağıramaz — çağırırsa canlı modelleri sessizce bozar. Bu yüzden v1:

1. `forecast_log`'dan `target_date`'i teslim eden `run_id`'yi bulur.
2. `models/archive/<run_id>/feature_matrix.parquet` (Faz -1 `archive_models()` tarafından zaten
   kopyalanıyor, `run_context.py:162-167`) + **o run'ın arşivlenmiş model dosyalarını** yükler —
   yeniden eğitmez, sadece `.predict()` çağırır.
3. Feature matrisinde gelecek satırlardaki forecast hava kolonlarını (`*_app_temp_actual`,
   `GHI_ADM_Weighted` — bunlar `03`'ün uniform şeması gereği forecast'ı da `_actual` adıyla tutuyor,
   bkz. tasarım dok. §2.5) `weather_history.parquet`'teki gerçekleşen değerle値iştirir (sadece o saat
   için reanalysis doluysa — ~D+6 sonrası).
4. **T+1** için doğrudan tahmin: T+1 satırlarının lag feature'ları (`Lag24h`+) her zaman gerçek geçmiş
   yüke dayanır, modelin kendi çıktısına değil → recursive zincir gerekmez, hava ikamesi saf.
5. **T+2** için bilinen yaklaşıklık: `_recompute_lags_for_t2` zinciri orijinal (canlı) T+1
   tahminleriyle kurulur (yeniden hesaplanmaz) — yalnızca hava etkisini izole eder, T+1 model
   belirsizliğinin T+2'ye ikinci-derece sızıntısını yok saymış olur. Bu, docstring'de açıkça
   belgelenir; tam sadakat (T+1'i de perfect-prog ile yeniden zincirlemek) Faz 4/backlog.
6. **v1 sınırı (kod ile doğrulandı):** ensemble = arşivlenmiş GBDT modellerinin (XGB+LGBM+CAT)
   basit ortalaması. Chronos-2 (LoRA inference) ve gerçek meta-stacker (rolling ridge) v1'de YOK —
   ekleme maliyeti (Chronos context hazırlığı + ağır inference, stacker'ın ayrı OOF geçmişi istemesi)
   bu fazın kapsamını aşıyordu. Sonuç: `perfect_mape` canlı `mape_final` ile birebir kıyaslanamaz;
   `weather_attribution_pp = live_mape − perfect_mape` **kaba bir üst-sınır işareti**, kesin sayı değil.
7. Çıktı: `perfect_mape` vs canlı `mape_final` (aynı target_date) → yukarıdaki kısıtla birlikte okunur.

CatBoost opsiyonel (arşivde yoksa atlanır, `cat_present=False` ile aynı davranış).

**Doğrulandı (2026-07-05):** gerçek arşivlenmiş bir run (`2026-07-05_329ea241`) üzerinde uçtan uca
çalıştırıldı. İki gerçek hata bulunup düzeltildi: (a) T+1 penceresi `all_nan[:24]` sanılmıştı, gerçekte
72-satırlık NaN bloğunda ilk 24 satır teslim edilmeyen "T+0 valid-lag" adımı — asıl T+1 `all_nan[24:48]`
(`04_predict_48h.py:split_train_predict` ile birebir); (b) `src/lightgbm_manager.py:save_model`
booster metnini Python text-mode `open(path,'w')` ile yazıyor, Windows'ta `\n`→`\r\n` çevirisi
LightGBM'in kendi metin formatını bozuyor (`lgb.Booster(model_file=...)` "Model format error, expect
a tree here" ile patlıyor) — production bunu hiç görmedi çünkü `04` asla kayıtlı bir LGBM'i geri
yüklemiyor, hep sıfırdan eğitiyor; bu script arşivden yükleyen ilk kod yolu. v1'de salt-okunur
normalize-et-geçici-dosyaya-yaz atlatmasıyla çözüldü; kalıcı düzeltme (`open(..., newline="\n")`)
`lightgbm_manager.py`'de — Faz 1 kapsamı dışı, ayrı **teknik borç** olarak not edilir.

---

## 5. Dosya değişiklik yüzeyi

| Dosya | Değişiklik |
|---|---|
| `src/scorecard.py` (yeni) | `build_daily_scorecard`, `window_report`, `check_alerts`, `latest_scorecard` |
| `perfect_prog_rerun.py` (yeni) | v1 — arşivlenmiş modellerle hava-ikame rerun (yeniden eğitim YOK) |
| `config_live.py` | `HEADLINE_HORIZON`, `SCORECARD_REBUILD_WINDOW_DAYS`, `Z_THRESHOLD`, `Z_BASELINE_WINDOW_DAYS`, `Z_WARMUP_MIN_DAYS`, `ALERTS_DIR` |
| `run_daily.py` | `build_daily_scorecard()` + `check_alerts()` çağrısı (forecast_log bloğundan sonra); `log_daily_mape` çağrısı kaldırıldı (borç #2 — kümülatif MAPE, scorecard onu emekliye ayırıyor) |

---

## 6. Açık Sorular (devam eden)

- **AS-1:** Isınma penceresi kaç gün sürecek — mevcut backfill (4 arşiv run'ı) çok kısa; ilk birkaç
  hafta `baseline_mode=warmup` kalacak, bu beklenen bir durum.
- **AS-2:** `gain_*_bps` (corrector net kazancı) backfill'lenmiş günlerde hesaplanamıyor (meta/delta
  NULL) — sadece Faz 0 sonrası ileriye dönük günlerde dolu olacak; raporlarda ayırt edilmeli.
- **AS-3:** Alarm kanalı v1 sadece dosya — e-posta/Teams (roadmap Açık Soru #3) hâlâ karar bekliyor.
