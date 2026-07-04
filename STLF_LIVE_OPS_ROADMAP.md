# STLF Live-Ops Yol Haritası — ADM Canlı Tahmin Ürünü

> **Amaç:** Deneysel/notebook kökenli, canlıya yeni geçmiş STLF sistemini sistematik, izlenebilir,
> aksiyon alınabilir bir live-ops ürününe dönüştürmek. Ana KPI: günlük MAPE'nin düşürülmesi —
> ama asıl hedef, *MAPE neden yüksekti* sorusunun her gün 15 dakikada cevaplanabilir olması.
>
> Bu doküman üç kaynağı birleştirir: (1) kod tabanı keşfi (2026-07-04), (2) literatür/best-practice
> taraması, (3) mevcut iki tasarım dokümanı — [stlf_izleme_deney_metodolojisi.md](stlf_izleme_deney_metodolojisi.md)
> ve [stlf_forecast_log_tasarim.md](stlf_forecast_log_tasarim.md) (Faz 0 şeması, §6 kararları kilitli).

---

## 1. Mevcut Durum Özeti (Kod Tabanı Envanteri)

### 1.1 Pipeline mimarisi

`run_daily.py` orkestratörü 6 adımı sırayla çalıştırır (her adım idempotent, ara çıktılar parquet):

```
01_ingest_actual   OneDrive DD.MM klasöründen müşteri CSV'si → master.parquet (upsert)
                   └─ update_oof_history(): dünkü arşiv tahmini ⋈ bugünkü actual → oof_history.parquet
02_fetch_weather   Open-Meteo Forecast API, 14 istasyon (Muğla/Denizli/Aydın) → weather_fc_live.parquet
                   └─ ilk 24h'i weather_history.parquet'e upsert (actual'lar ~D+6 Archive sync ile düzelir)
03_build_features  master + weather_history + weather_fc → Boray DataManager → feature_matrix.parquet
04_predict_48h     4 model recursive (T+0→T+1→T+2): XGB + LGBM + CatBoost(ops.) + Chronos-2(LoRA)
                   └─ Stacking: Rolling Ridge (OOF, 60 gün) → frozen Ridge → basit ortalama (fallback zinciri)
                   └─ Holiday override: hafta içi tatilde CatBoost solo
05_postprocess     Holiday substitution (donmuş alpha) + PV bias correction (donmuş lookup, T1/T2 ayrı)
06_deliver         T+2 günü → Excel teslim + tüm 48h → output/archive/*.parquet
```

Modeller her gün **yeniden eğitilir** (son ~22.000 saat, `MAX_TRAIN_SIZE` concept-drift kapağı) ve
model dosyaları **yerinde üzerine yazılır** (`04:231,237,246`). Post-process artefaktları (PV bias
lookup, holiday alphas) donmuş; `POST_HOLIDAY_MULTIPLIERS` config'de hardcoded.

### 1.2 Mevcut loglama/kayıt

| Mekanizma | İçerik | Granülerlik | Durum |
|---|---|---|---|
| `logs/<date>_run.log` | Adım süreleri, hata traceback | Run başına | Var — ama `logs/` şu an **boş** (⚠ doğrulanmalı) |
| `logs/<date>_summary.json` | Adım sonuç dict'leri | Run başına | Aynı |
| `data/oof_history.parquet` | date, hour, actual, 5 model tahmini | Saatlik | Çalışıyor (01 tetikler) |
| `logs/mape_history.json` | Model bazlı MAPE | Günlük entry | ⚠ Hesap **kümülatif** — aşağıya bak |
| `output/archive/*_full48h.parquet` | Tüm 48h, model bazlı tahminler | Saatlik | Çalışıyor (2 run arşivli) |

**Ne YOK:** WAPE/ME/RMSE yok, günlük bazda MAPE trendi yok, meta-model ağırlıkları hiç kaydedilmiyor
(üretilip atılıyor), düzeltme adımlarının (override/substitution/PV) per-saat katkısı kaydedilmiyor,
kullanılan hava tahmini snapshot'ı ayrıca loglanmıyor, config/model versiyonu hiçbir çıktıya damgalanmıyor.

### 1.3 Config ve versiyonlama — tekrarlanabilirlik durumu

- **Kod git'te DEĞİL.** Çalışma dizini bir git reposu değil; kod OneDrive senkronuyla "versiyonlanıyor". En büyük tekrarlanabilirlik açığı bu.
- Config tek Python dosyası (`config_live.py`) — hash'lenmiyor, değişiklik tarihi izlenmiyor. "3 hafta önceki tahmin hangi config ile üretildi?" sorusu bugün cevaplanamaz.
- Model artefaktları her gün üzerine yazıldığı için geçmiş model durumu geri getirilemez (arşivde sadece *tahminler* var, modeller yok).
- HPO parametreleri JSON dosyalarında (`best_params_*_sagemaker_hpo.json`) — bu iyi; ama hangi HPO run'ından geldikleri kayıtlı değil.
- `requirements.txt` / environment pin yok.

### 1.4 Zayıf noktalar / teknik borç envanteri

1. **Sessiz fallback'ler attribution'ı bozuyor:** Chronos patlarsa `CHRONOS_Pred` kolonuna **XGB kopyası** yazılıyor ([04_predict_48h.py:400](pipeline/04_predict_48h.py:400)); CatBoost import edilemezse sessizce atlanıyor ([04:248](pipeline/04_predict_48h.py:248)). Logda flag yok → OOF history'de Chronos'un "performansı" aslında XGB olabilir. (Faz 0 şemasındaki `chronos_ok`/`cat_present` tam bunu çözüyor.)
2. **`log_daily_mape` yanıltıcı:** [oof_feedback.py:213](src/oof_feedback.py:213) MAPE'yi *tüm OOF geçmişi* üzerinden hesaplıyor, günlük değil — trend izlemek için kullanılamaz. 201-206'daki günlük groupby ölü kod.
3. **Global monkeypatch:** [03_build_features.py:47](pipeline/03_build_features.py:47) `pd.DataFrame.dropna`'yı process-genelinde no-op yapıyor (Boray DataManager'ı değiştirmemek için). Kontrollü ama kırılgan — DataManager'a başka bir dropna eklenirse sessizce farklı davranır.
4. **mtime hack:** [03:213-221](pipeline/03_build_features.py:213) sahte `_tmp_combined.xlsx` + `os.utime` ile DataManager'ın cache mantığı kandırılıyor.
5. **Ölü config:** `ENABLE_WEEKEND_SPLIT_*` flag'leri ve `MODEL_*_WD_SAT/_WE` yolları canlı 04'te hiç kullanılmıyor (import edilip bırakılmış) — config ile gerçek davranış ayrışmış.
6. **Hata yönetimi = maskeleme:** `stack_predictions` ve post-process adımları geniş `except`'lerle fallback'e düşüyor; run "ok" bitiyor ama hangi yolun kullanıldığı sadece stdout print'inde. Pipeline hata ile durursa **alerting yok** (summary.json'a yazılıyor, kimse bakmazsa görülmüyor).
7. **OneDrive bağımlılığı:** Veri girişi OneDrive klasör senkronuna bağlı (`LIVE_DATA_DIR`); parquet/DuckDB gibi dosyaların OneDrive altında kilitlenme/çakışma riski var (bkz. Açık Soru #2).
8. **Test yok:** Hiçbir adımın birim/entegrasyon testi yok; `validate()` (01) tek gerçek veri doğrulama noktası.
9. **UI izleme yapmıyor:** Streamlit paneli 3 sekme (Veri Durumu / Veri Yükleme / Tahmin Üret) — operasyon aracı, izleme/analiz sayfası yok.

**Özet hüküm:** Pipeline'ın kendisi düzgün kurgulanmış (idempotent adımlar, ara parquet'ler, OOF feedback
döngüsü). Eksik olan **gözlemlenebilirlik ve tekrarlanabilirlik katmanı** — tam da Faz 0 tasarımının hedefi.

---

## 2. Hedef Mimari

### 2.1 Katmanlar

```
┌────────────────────────────────────────────────────────────────┐
│  UI: Streamlit multipage (Operasyon / İzleme / Deney)          │  ← Faz 3 (karar verildi: Dash'e geçilmiyor)
├────────────────────────────────────────────────────────────────┤
│  Analiz: daily_scorecard, robust-z, CUSUM, PSI, triyaj,        │  ← Faz 1-2-4
│  perfect-prog rerun, hata madenciliği, shadow karşılaştırma    │
├────────────────────────────────────────────────────────────────┤
│  Log deposu: forecast_log + actuals_log (parquet, partition)   │  ← Faz 0
│  + known_events.csv + monitoring.duckdb (türetilmiş)           │
├────────────────────────────────────────────────────────────────┤
│  Pipeline: run_daily 6 adım (mevcut) + run_id/config_hash      │
│  + snapshot delta'ları + shadow runner (Faz 4)                 │
├────────────────────────────────────────────────────────────────┤
│  Temel hijyen: git repo, requirements pin, model arşivi        │  ← Faz -1 (yeni)
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Logging şeması (özet — tam şema: [stlf_forecast_log_tasarim.md](stlf_forecast_log_tasarim.md))

Kilitlenmiş kararlar (2026-07-04): her düzeltme adımı ayrı delta kolonu; `known_event` başta CSV;
perfect-prog rerun mümkün (weather_history = gerçekleşme, ~D+6); depo = parquet + DuckDB; günlük partition.

| Tablo | Grain | Ne zaman yazılır | Kritik alanlar |
|---|---|---|---|
| `forecast_log` | saat × run | Tahmin anında (05 sonrası) | 4 model ham tahmini, `chronos_ok`/`cat_present`, meta ağırlıklar + method, `y_pred_ens_raw`, `override_delta`/`subst_delta`/`pv_bias_delta`, `y_pred_final`, `wx_*_fcst`, `day_type`, `config_hash`, `model_versions`, `run_id`, `edas_id` |
| `actuals_log` | saat | D+1 (yük) + ~D+6 (hava) iki dalga | `y_actual`, `wx_*_actual`, `data_quality_flag`, `known_event` |
| `daily_scorecard` | gün | Her sabah türetilir (DuckDB) | MAPE/WAPE/RMSE/ME, saat-blok MAPE'leri, model bazlı MAPE, corrector katkısı bps, `temp_fcst_error`/`ghi_fcst_error`, `robust_z`, `verdict_code` |
| `known_events.csv` | olay | Manuel | edas_id, ts_start, ts_end, kategori, not |

Depolama: `logs/forecast_log/edas_id=ADM/target_date=YYYY-MM-DD/run_<issue_date>.parquet` + tek
`monitoring.duckdb`. Hacim: tek EDAŞ × saatlik = günde 48 satır → yıllarca tek makinede sorun yok.
**Varsayım:** Multi-tenant 5+ EDAŞ olsa bile hacim DuckDB sınırlarının çok altında kalır; ağır MLOps
platformu (Kubeflow vb.) bu ölçekte gereksiz — literatürde de küçük-orta hacim için "parquet + DuckDB +
cron + dashboard" deseni yeterli görülüyor (bkz. Evidently model monitoring rehberi, Kaynakça #1).

### 2.3 Deney/versiyon katmanı

- **Git repo** (Faz -1): kod + config versiyonu. `config_hash` = config_live.py içerik hash'i; promotion günü hash değişimi changepoint analizinde otomatik işaret olur.
- **Model artefakt arşivi:** her run'da `models/` → `models/archive/<run_id>/` kopyası (birkaç MB/gün; 90 gün tut, sonra sil). MLflow Model Registry bu ölçekte opsiyonel — dosya-tabanlı arşiv + `model_versions` kolonu aynı işi görür (Açık Soru #4).
- **MLflow'un yeri:** offline deneyler (backtest koşuları) için uygun; **production günlük skorları MLflow'a değil `daily_scorecard`'a** yazılır (metodoloji dokümanı §1 kararıyla uyumlu; MLflow tracking dokümantasyonu da production monitoring'i ayrı tutmayı önerir, Kaynakça #12).

---

## 3. Faz Faz Yol Haritası

Fazlar mevcut plandaki 0-4 sırasını korur; başına "hijyen" fazı (-1), sonuna çoklu-EDAŞ fazı (5) eklendi.
Eforlar kaba tahmindir (tek kişi, kesin taahhüt değil).

### Faz -1 — Temel Hijyen (YENİ) — ~1-2 gün
- **Ne:** `git init` + ilk commit + `.gitignore` (data/, models/, output/, logs/); `requirements.txt` pin; model dosyalarının run sonrası `models/archive/<run_id>/`'a kopyalanması; `logs/` boşluğunun nedeninin bulunması (run.log neden yok?).
- **Neden:** Tekrarlanabilirliğin sıfır noktası. `config_hash` ancak git ile anlamlı olur; forecast_log'a yazılacak `model_versions` ancak arşiv varsa işe yarar. Bugün bir "kötü gün"ün config'ini geri getirmek imkânsız.
- **Bağımlılık:** Yok. Bugün başlanabilir, Faz 0 ile paralel yürür.

### Faz 0 — Günlük Log Altyapısı (tasarım kilitli, sıra kodda) — ~3-5 gün
- **Ne:** `src/forecast_logger.py` (pyarrow şema + `write_forecast_log()` / `update_actuals_log()`); `04`'te override-öncesi ensemble snapshot'ı, `stack_predictions`'ın meta ağırlık/method döndürmesi, `chronos_ok`/`cat_present` flag'leri; `05`'te adım-arası snapshot'lardan delta kolonları; `run_daily`'de `run_id`/`config_hash` üretimi; `01`'de `data_quality_flag`; `output/archive`'dan kısmi backfill.
- **Neden:** "Neden" sorusunun hammaddesi. Tahmin anında yakalanmayan (meta ağırlık, delta, kullanılan hava) geri dönülemez kaybolur. Sessiz fallback'ler (teknik borç #1) ancak flag'lenirse görünür olur.
- **Bağımlılık:** Tasarım onayı ✅ (2026-07-04). Faz -1'den `run_id` disiplinini alır.

### Faz 1 — daily_scorecard + Çoklu Pencere (7/30/365) Analizi — ~4-6 gün
- **Ne:** forecast_log ⋈ actuals_log join'inden günlük scorecard (DuckDB); MAPE/WAPE/ME/RMSE + saat-blok + model bazlı attribution; robust z-score alarmı (`z = (MAPE−median_30d)/(1.4826·MAD_30d)`, z>3 tetik; tatiller ayrı baseline); 7/30/365 pencere raporu (7g: operasyonel sağlık, 30g: sistematik bias/mevsim geçişi, 365g: yapısal drift); perfect-prog rerun scripti (gerçekleşen hava ile tahmini yeniden üret → hata ayrıştırması model-vs-meteoroloji).
- **Neden:** Günlük MAPE gürültülü — tek günden karar çıkmaz; median/MAD tabanlı eşik tatil kirliliğine dayanıklı. Yük literatüründe kötü günlerin büyük payı hava tahmin hatasından gelir; perfect-prog rerun olmadan model haksız yere suçlanır (GEFCom geleneği, Kaynakça #8). WAPE eklenmeli çünkü düşük yük saatlerinde MAPE şişer (Kaynakça #6).
- **Bağımlılık:** Faz 0 (en az ~2 hafta forecast_log birikimi; backfill ile hızlanır). `log_daily_mape` kümülatif hesabı (borç #2) bu fazda emekliye ayrılır.

### Faz 2 — "Kötü Gün" Triyaj Protokolü — ~2-3 gün kod + süreç oturtma
- **Ne:** Metodoloji §3'teki 6 adımlı protokolün (veri kalitesi → dış şok → hava hatası → saat profili → bileşen attribution → verdict) yarı-otomasyonu: z>3 gününde otomatik "triyaj raporu" üret (ilgili tüm log alanları tek sayfada), insan verdict kodunu seçer, `daily_scorecard.verdict_code`'a yazılır. `known_events.csv` süreci başlar. 2-3 geçmiş kötü gün üzerinde prova.
- **Neden:** Verdict dağılımı hem aylık kalite raporunun hem deney backlog'unun hammaddesi. `UNEXPLAINED` oranı >%20 ise log şeması eksik demektir — şemanın kendini test etme mekanizması.
- **Bağımlılık:** Faz 1 (scorecard + rerun).

### Faz 3 — Dashboard MVP (Streamlit'te kalınıyor) — ~4-6 gün
- **Ne:** Mevcut panele multipage yapı: **Operasyon** (bugünkü run durumu, veri sağlığı — mevcut sekmeler), **İzleme** (scorecard trendi, 7/30/365 pencereler, saat-blok ısı haritası, model bazlı MAPE, corrector katkısı, meta ağırlık trendi — Plotly), **Deney** (shadow karşılaştırma, verdict dağılımı, backlog). Veri kaynağı: doğrudan `monitoring.duckdb`.
- **Neden:** Günlük 5-10 dk'lık insan bakışının aracı. Karar verildi: iç araç + az kullanıcı için Streamlit yeterli; Dash'in avantajı (WSGI ölçekleme, fine-grained callback) bu kullanım profilinde gerekmiyor (Kaynakça #15-16). Yeniden değerlendirme tetiği: eşzamanlı kullanıcı >10 veya sayfa-arası state sorunları çıkarsa.
- **Bağımlılık:** Faz 1 (gösterilecek veri). Faz 2 ile paralel yürüyebilir.

### Faz 4 — CUSUM + Drift + Hata Madenciliği + Champion-Challenger — ~1.5-2 hafta
- **Ne:**
  - **CUSUM** günlük ME üzerinde (k=0.5σ, h=4-5σ; σ = son 90 gün, tatil hariç) — sinsi bias kayması alarmı; literatürde forecast-error-üzerinde-CUSUM'un raw-data'dan daha iyi çalıştığı gösterildi (Kaynakça #10).
  - **PSI/KS input drift** raporu aylık (sıcaklık, GHI, yük dağılımı; PSI>0.2 eşiği) + concept drift ayrımı (aynı sıcaklık bin'lerinde residual yıl-yıl karşılaştırması — input aynı + hata farklı = ilişki değişmiş, re-fit yetmez) (Kaynakça #1-3). **Varsayım:** Evidently kütüphanesi hazır PSI/KS verir ama bağımlılık eklemek yerine ~100 satır custom hesap da yeterli olabilir — Faz 4 başında karar.
  - **Hata madenciliği:** residual'ları gün tipi × saat bloğu × sıcaklık rejimi × GHI rejimi hücrelerinde grupla; |ME| anlamlı + kapsam anlamlı hücreler deney adayı; öncelik skoru = (etki × kapsam × güven) / maliyet.
  - **Champion-challenger çerçevesi:** 3 kapı — (1) backtest: sabit fold seti + Diebold-Mariano testi (Harvey küçük-örneklem düzeltmeli) + en-kötü-10-gün guardrail; (2) shadow: challenger her gün aynı inputlarla koşar, `forecast_log`'a `shadow_pred_<id>` kolonu, min 28 gün (4 hafta döngüsü; tatil kritikse kapsayana kadar uzat); (3) kontrollü geçiş: probation 14 gün, z eşiği 2.5, **ters shadow** (eski config shadow'da koşmaya devam — rollback kriteri önceden yazılır). Tek tahminli üründe trafik bölünemediği için gerçek A/B yok; bu disiplin onun ikamesi (Kaynakça #11, #13-14).
- **Neden:** (a) günlük alarm tek kötü günü yakalar, CUSUM sürekli kaymayı; (b) yeni model/feature'ı canlıya güvenle sokmanın tek yolu shadow — "backtest'te iyiydi" tek başına promotion gerekçesi olamaz (M5 dersleri: protokol sabitlenmezse herkes kendi fold'unu seçer, Kaynakça #9).
- **Bağımlılık:** Faz 0-1 (loglar + scorecard). Shadow runner, pipeline'ın 04-05 adımlarını challenger config ile ikinci kez koşturur — **Varsayım:** günlük ek ~1 saat işlem süresi kabul edilebilir (Chronos CPU'da yavaş; challenger GBDT-only ise çok daha hızlı).

### Faz 5 — Çoklu EDAŞ Genişleme Mimarisi — ~2-3 hafta (GDZ somutlaşınca)
- **Ne:** (a) `edas_id` zaten tüm şemalarda (Faz 0 kararı) — veri tarafı hazır; (b) pipeline'ı tenant-parametrik hale getir: `config_live.py` → `configs/<edas_id>.py` (veya YAML) + ortak `config_base`; (c) model katmanına adapter arayüzü: her EDAŞ'ın model seti bir "strategy" (ADM = 4-model stacking; GDZ = kendi metodolojisi) ama hepsi aynı `forecast_log` şemasına yazar — izleme/triyaj/dashboard katmanı hiç değişmez; (d) `run_daily.py --edas GDZ`; (e) scorecard ve dashboard'da tenant seçici.
- **Neden:** Genişleme maliyetini "yeni EDAŞ = yeni monitoring sistemi"nden "yeni EDAŞ = yeni config + model adapter"a indirmek. İzleme katmanının model-agnostik olması bunun ön şartı — Faz 0 şeması bunu şimdiden garanti ediyor.
- **Bağımlılık:** Faz 0-3'ün oturması. **Varsayım:** GDZ verisi/metodolojisi ADM ile aynı saatlik grain'de olacak; değilse şemaya `resolution` alanı eklenir (küçük değişiklik).

### Sürekli (faz değil): Operasyonel ritim
Metodoloji §7'deki günlük (5-10 dk) / haftalık (30-45 dk) / aylık (2-3 saat) / çeyreklik checklist'ler,
Faz 1 tamamlanınca yürürlüğe girer. Alerting kanalı için Açık Soru #3.

---

## 4. İlk 2 Haftada Yapılacaklar (sıralı, somut)

**Hafta 1 — hijyen + log altyapısı:**
1. **Gün 1:** `git init` + ilk commit + `.gitignore`; `pip freeze > requirements.txt`; `logs/` klasörünün neden boş olduğunu araştır (run.log yazılıyor mu, OneDrive mi taşıdı?) — izleme sisteminin kendisi loglara dayanacak, bu belirsizlik kapanmalı.
2. **Gün 1-2:** `config_live.py`'ye `EDAS_ID`, `FORECAST_LOG_DIR`, `ACTUALS_LOG_DIR`, `MONITORING_DB` sabitleri; `run_daily.py`'de `run_id` (`<issue_date>_<config_hash8>`) üretimi; run sonrası model dosyalarını `models/archive/<run_id>/`'a kopyala.
3. **Gün 2-4:** `src/forecast_logger.py` — pyarrow şeması (tasarım dokümanı §2-3 birebir) + `write_forecast_log()` + `update_actuals_log()`. `04`'e snapshot/flag eklemeleri (`y_pred_ens_raw`, meta ağırlıklar, `chronos_ok`, `cat_present`), `05`'e delta kolonları.
4. **Gün 4-5:** `run_daily.py` entegrasyonu → bir sonraki canlı run'da forecast_log dolmaya başlar. `output/archive/*` + `oof_history.parquet`'ten kısmi backfill (y_pred'ler ve actual'lar geriye dönük; meta ağırlık/delta'lar sadece ileriye dönük).

**Hafta 2 — scorecard + ilk sinyaller:**
5. **Gün 6-8:** `build_daily_scorecard()` (DuckDB): MAPE/WAPE/ME/RMSE, saat-blok, model bazlı MAPE, corrector katkısı. Backfill'lenen geçmişle ilk scorecard'ı üret.
6. **Gün 8-9:** Robust z-score hesabı + `known_events.csv` iskeleti. (30 günlük median/MAD penceresi dolana kadar z-score "ısınma modunda" — mutlak eşik, örn. son 60 gün p95, geçici tetik olur. **Varsayım:** backfill ~30+ gün sağlarsa ısınma kısalır.)
7. **Gün 9-10:** Perfect-prog rerun scripti (v1: manuel tetiklenen — hedef gün ver, weather_history'deki actual havayla 03-04-05'i yeniden koştur, iki MAPE'yi karşılaştır).
8. **Gün 10:** Geçmişteki en kötü 1-2 gün üzerinde triyaj provası — şemada eksik alan var mı testi. Bulgulara göre şema küçük revizyon.

Bu iki haftanın sonunda: her run izlenebilir (`run_id`+`config_hash`), her tahminin bileşen dökümü kalıcı,
günlük scorecard otomatik, kötü gün alarmı ve hava-ayrıştırması çalışır durumda.

---

## 5. Açık Sorular / Kararlar (proje sahibinin masasında)

1. ~~Streamlit mi Dash mi?~~ **Karar verildi:** Streamlit + Plotly, multipage. Yeniden değerlendirme tetiği: >10 eşzamanlı kullanıcı veya real-time refresh ihtiyacı.
2. **Log storage nerede?** Mevcut her şey OneDrive-senkronlu dizinde. Parquet append + DuckDB için OneDrive kilitlenme/çakışma riski gerçek. Seçenekler: (a) olduğu gibi bırak (basit, riskli), (b) `logs/` ve `monitoring.duckdb`'yi OneDrive dışı lokal dizine taşı + günlük zip yedeği OneDrive'a (önerim), (c) baştan küçük bir sunucu/NAS. Karar Faz 0 kodundan **önce** gerekli.
3. **Alerting kanalı:** Pipeline hatası ve z>3 alarmı nereye düşsün? E-posta / Teams webhook / sadece dashboard? (Şu an: hiçbiri.)
4. **Model artefakt arşivi vs MLflow Registry:** Önerim dosya-tabanlı arşiv (basit, yeterli); MLflow sadece offline backtest deneylerinde. Ekip büyürse Registry'ye geçiş kolay.
5. **Shadow compute bütçesi:** Challenger'ın günlük tam koşusu (Chronos dahil) ~1 saat ek CPU. Kabul mü, yoksa challenger'lar GBDT-only mi başlasın?
6. **`known_events` sahibi kim?** Kesinti/arıza/etkinlik bilgisini kim, hangi kaynaktan girecek? (Saha ekibiyle temas gerektirir — teknik değil organizasyonel karar.)
7. **Probabilistic forecasting ufku:** Müşteri tek nokta tahmini istiyor; ama pinball loss ile kuantil takibi (Hong & Fan) risk iletişimi için değerli. Faz 4 sonrası "araştırma" maddesi olarak backlog'a alınsın mı?
8. **Chronos fallback politikası:** Chronos patlayınca XGB kopyası yazmak yerine ensemble'ı 3 modelle koşturmak daha dürüst. Faz 0'da `chronos_ok` flag'i görünürlük getirir; davranış değişikliği (kopya yerine drop) ayrıca kararlaştırılmalı.

---

## 6. Kaynakça

**Production ML monitoring / drift:**
1. [Evidently AI — Model monitoring for ML in production: a comprehensive guide](https://www.evidentlyai.com/ml-in-production/model-monitoring) — monitoring metrik seti, batch izleme deseni; PSI/KS otomasyonu ([drift customization docs](https://docs.evidentlyai.com/metrics/customize_data_drift)).
2. [Encord — Model Drift Best Practices](https://encord.com/blog/model-drift-best-practices/) ve [Monitoring and Managing Data Drift](https://encord.com/blog/monitoring-and-managing-data-drift-in-production-ml-systems/) — data vs concept drift ayrımı, root-cause süreci, otomatik alarm önerileri.
3. [Label Your Data — Data Drift Detection Techniques 2026](https://labelyourdata.com/articles/machine-learning/data-drift) — KS/Chi-square/PSI/KL karşılaştırması.
4. [Fiddler — Measuring Data Drift with PSI](https://www.fiddler.ai/blog/measuring-data-drift-population-stability-index) ve [Coralogix — Practical Introduction to PSI](https://coralogix.com/ai-blog/a-practical-introduction-to-population-stability-index/) — PSI eşik pratiği (>0.2 belirgin kayma).

**Forecasting-specific izleme / metrik:**
5. [IBM — What Is Load Forecasting?](https://www.ibm.com/think/topics/load-forecasting) — STLF horizon-amaç eşleşmesi.
6. [MoldStud — Load Forecasting Key Metrics & Best Practices](https://moldstud.com/articles/p-understanding-load-forecasting-key-metrics-best-practices-for-effective-energy-management) — MAPE benchmarkları (National Grid UK ~%2.1, PJM ~%3.3), sık model güncellemenin etkisi; [Procuzy — Forecast Accuracy Metrics Guide](https://procuzy.com/blog/ultimate-guide-to-forecast-accuracy-metrics/) — WAPE'nin MAPE'ye üstün olduğu düşük-hacim durumları.
7. Hong, T. & Fan, S. (2016), ["Probabilistic Electric Load Forecasting: A Tutorial Review", IJF 32:914-938](https://www.sciencedirect.com/science/article/abs/pii/S0169207015001508) — MAPE sınırları, pinball loss / Winkler skoru, olasılıksal değerlendirme.
8. [GEFCom2014 makalesi (Hong et al., IJF)](https://www.sciencedirect.com/science/article/abs/pii/S0169207015001405) — hava senaryolarıyla değerlendirme; perfect-prog rerun fikrinin temeli.
9. Makridakis et al., M4/M5 yarışmaları (IJF) — sabit değerlendirme protokolünün önemi; fold versiyonlama gerekçesi. Tashman (2000), ["Out-of-sample tests of forecasting accuracy" (IJF)](https://www.researchgate.net/publication/223319987_Out-of-sample_tests_of_forecasting_accuracy_An_analysis_and_review) — rolling-origin kanonik kaynak.
10. [Grundy et al. (2026), "Online Detection of Forecast Model Inadequacies Using Forecast Errors", J. Time Series Analysis](https://onlinelibrary.wiley.com/doi/10.1111/jtsa.12843) ([arXiv](https://arxiv.org/html/2502.14173v1)) — forecast error üzerinde sequential changepoint/CUSUM: ME kayması = model bias'landı sinyali; [MetricGate — PELT/BinSeg/CUSUM pratik özeti](https://metricgate.com/blogs/changepoint-detection-methods/).

**Champion-challenger / shadow:**
11. [Wallaroo — A/B Testing and Shadow Deployments](https://wallaroo.ai/ai-production-experiments-the-art-of-a-b-testing-and-shadow-deployments/), [DagsHub — Model Deployment Strategies](https://dagshub.com/blog/model-deployment-types-strategies-and-best-practices/), [Dataiku — Monitoring and Feedback Concept](https://knowledge.dataiku.com/latest/mlops-o16n/model-monitoring/concept-monitoring-feedback.html), [Snowflake — Champion-Challenger Deployment Guide](https://www.snowflake.com/en/developers/guides/ml-champion-challenger-model-deployment/) — shadow = challenger aynı inputla koşar, çıktısı loglanır, karar empirik.
   - DM testi: Diebold & Mariano (1995) + Harvey et al. (1997) düzeltmesi; [Wiley — Evaluating Forecasts at Multiple Horizons (DM uzantısı)](https://onlinelibrary.wiley.com/doi/full/10.1002/for.70150).

**Experiment tracking / logging şeması:**
12. [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/), [Model Registry](https://mlflow.org/docs/latest/ml/model-registry/), [Dataset tracking](https://mlflow.org/docs/latest/ml/dataset/) — ne loglanmalı: params/metrics/artifacts/tags + data lineage (hash, versiyon); registry'nin run-lineage bağı.
13. [Dataiku/MLOps topluluğu — A/B testing in ML](https://mlops.community/blog/the-what-why-and-how-of-a-b-testing-in-ml) — tek-tahminli üründe A/B kısıtları.

**Ensemble/stacking izleme:**
14. [EmergentMind — Stacked Ensemble Model özeti](https://www.emergentmind.com/topics/stacked-ensemble-model) — lineer meta-learner katsayılarının katkı yorumu; ağırlık-doğruluk korelasyonunun kusurlu olması (çeşitlilik etkisi); drift altında periyodik meta-retrain ihtiyacı; [mcpanalytics — OOF stacking ve leakage](https://mcpanalytics.ai/articles/stacking-ensemble-practical-guide-for-data-driven-decisions).

**Dashboard mimarisi:**
15. [Squadbase — Streamlit vs Dash 2025](https://www.squadbase.dev/en/blog/streamlit-vs-dash-in-2025-comparing-data-app-frameworks) ve [Databrain — Streamlit vs Dash 2026](https://www.usedatabrain.com/blog/streamlit-vs-dash) — Streamlit'in per-connection RAM/session-affinity sınırı; Dash'in WSGI ölçeklemesi; "~%80 dashboard'un Dash'e ihtiyacı yok, iç araçta Streamlit default" önerisi.
16. [Plotly — Streamlit Alternatives for Production](https://plotly.com/blog/best-streamlit-alternatives-production-data-apps/) (satıcı perspektifi), [Quansight — Big Four Dashboarding Tools](https://quansight.com/post/dash-voila-panel-streamlit-our-thoughts-on-the-big-four-dashboarding-tools/) — tarafsız dört-araç karşılaştırması.

**Enerji sektörü operasyonel pratik:**
17. [ScienceDirect — "How good are TSO load and renewable generation forecasts" (Applied Energy)](https://www.sciencedirect.com/science/article/abs/pii/S0306261922008753) — ENTSO-E Transparency verisiyle TSO tahmin kalitesi öğrenme eğrileri; operasyonel benchmark bağlamı.
18. [ScienceDirect — "Short-term electricity load forecasting—A systematic approach from system level to secondary substations" (Applied Energy)](https://www.sciencedirect.com/science/article/pii/S0306261922017500) — DSO seviyesinde sistematik STLF yaklaşımı.
19. Özel gün literatürü: [Short-term load forecasting with special days (parametric vs non-parametric)](https://www.researchgate.net/publication/322143085_Short-term_electricity_load_forecasting_with_special_days_an_analysis_on_parametric_and_non-parametric_methods), [Discrete-interval moving seasonalities ile tatil tahmini (Energy)](https://www.sciencedirect.com/science/article/abs/pii/S0360544221012147), [Rule-based triple seasonal methods for anomalous load (arXiv)](https://arxiv.org/pdf/1409.2027) — benzer-gün/substitution yaklaşımlarının literatür karşılığı; tatil profilinin yıldan yıla haftanın gününe göre kaydığı uyarısı.

**Metodolojik temel (mevcut iç dokümanlardan devralınan):** Page (1954) CUSUM; Gama et al. (2014) concept drift taksonomisi; Bifet & Gavaldà (2007) ADWIN; Truong et al. (2020) `ruptures`.

---

*Doküman: 2026-07-04. Kod tabanı keşfi bu tarihli canlı dizin üzerinden; iki canlı run arşivi mevcuttu (2026-07-01, 2026-07-03).*
