# STLF Live-Ops Yol Haritası — ADM + GDZ Canlı Tahmin Ürünü

> **Amaç:** Deneysel/notebook kökenli, canlıya yeni geçmiş STLF sistemini sistematik, izlenebilir,
> aksiyon alınabilir bir live-ops ürününe dönüştürmek. Ana KPI: günlük MAPE'nin düşürülmesi —
> ama asıl hedef, *MAPE neden yüksekti* sorusunun her gün 15 dakikada cevaplanabilir olması.
>
> **Bu doküman iki kaynağı birleştirir:** (1) orijinal kod tabanı keşfi (2026-07-04), (2) 2026-07-07
> itibarıyla ilerleme + GDZ entegrasyonu + demo-hazırlık haftasından çıkan güncellemeler. İki canlı
> EDAŞ artık entegre: **ADM** (`adm live/`) ve **GDZ** (`gdz talep/live/`).

---

## 0. Bu Revizyonun Neleri Değiştirdiği (2026-07-07)

Orijinal doküman (2026-07-04) yazıldıktan sonra bir haftada yol alındı. Özet delta:

| Orijinal iddia (2026-07-04) | Güncel gerçek (2026-07-07) | Aksiyon |
|---|---|---|
| Faz -1, 0, 1 tasarım / sıra kodda | **Faz -1, 0, 1 TAMAMLANDI** (commit `b714150`, `4ff8e40`) | Aşağıda ✅ işaretlendi |
| Tek EDAŞ (ADM); Faz 5 "GDZ somutlaşınca" | **GDZ entegre edildi** — ama *clone + subprocess* mimarisiyle, roadmap'in parametrik tasarımına DEĞİL | Faz 5 "Divergent" olarak yeniden yazıldı |
| `logs/` boş, git yok, requirements yok | **Git repo var** (13 commit), `.gitignore`, `requirements.txt` (586 satır pin), `logs/` LocalAppData'da doluyor | Faz -1 "done" |
| Açık Soru #2 (log storage nerede?) | **Karar verildi:** LocalAppData + günlük zip OneDrive'a | Soru kapatıldı |
| Backtest yok | 4 backtest scripti var (`backtest_30d/7d/tomorrow`, `asof_regen`) — tracked değil | Faz 2b: backtest'i log altyapısına bağla + 1-7 Temmuz doldur |
| Faz 3 (İzleme dashboard) ~4-6 gün | **Hâlâ YAPILMADI** — UI 3 operasyon tabında. Veri altyapısı (monitoring.duckdb, scorecard) hazır ama UI okumuyor | **Faz 3 öncelik yükseldi** — canlı akış 8 Temmuz'da başlıyor, gözle izleyecek yer yok |
| — | **YENİ gereksinim:** her gün 9 Temmuz → 10 Temmuz actuals akışı, sürekli | "Canlı Ops Ritmi" bölümü eklendi |

---

## 1. Mevcut Durum Özeti (Kod Tabanı Envanteri — 2026-07-07)

### 1.1 İki tenant, iki klon

```
çağatay/
├── adm live/              ← ADM tenant (EDAS_ID="ADM")
│   ├── config_live.py     (276 satır)
│   ├── run_daily.py       (157 satır, 6 adım orkestratör)
│   ├── pipeline/{01..06}.py
│   ├── src/{run_context, forecast_logger, scorecard, oof_feedback, holiday_calendar}.py
│   ├── ui/                (Streamlit, 3 tab — Operasyon only)
│   ├── data/ models/ output/ logs/  (logs/ OneDrive DIŞI, %LOCALAPPDATA%\adm_live_logs\)
│   └── backtest_*.py, asof_regen.py, perfect_prog_rerun.py, analyze_models_30d.py
│
└── gdz talep/             ← GDZ tenant (araştırma kökü)
    └── live/              ← GDZ canlı (EDAS_ID="GDZ") — ADM'in klonu
        ├── config_live_gdz.py   (ADM config'i import + GDZ override)
        ├── run_daily.py, pipeline/{01..06}.py
        ├── src/{run_context, forecast_logger, holiday_calendar}.py  (ported)
        └── data/ models/ output/ logs/  (logs/ %LOCALAPPDATA%\gdz_live_logs\)
```

**Mimari kararı (gerçekleşen):** Faz 5'in "tek pipeline + `configs/<edas_id>.py` + `--edas` flag"
tasarımı **uygulanmadı**. Bunun yerine her EDAŞ için tüm `live/` dizini kopyalandı (GDZ_LIVE_PORTING_PLAN.md
buna "Architecture B — directory clone" der). Neden: ADM'in `ui/` modülleri `from run_context import ...`
gibi bare import'lar yapıp `sys.modules`'i cache'liyor — aynı process'te GDZ modüllerini yüklemek
ADM şemalı fonksiyonların GDZ verisine uygulanmasına (sessiz hata) yol açıyor. Çözüm: UI, GDZ
pipeline'ını **subprocess** olarak çağırıyor (`ui/common.py:54-88`, PIPE-deadlock fix'i `:70-78`).

**Sonuç:** İzleme/triyaj/dashboard katmanı model-agnostik kalsın diye forecast_log şemasında `edas_id`
var (Faz 0 kararı sağlam), ama *kod* katmanı duplikatlı. 3. EDAŞ gelinceye kadar kabul edilebilir;
o zaman parametrik refactor gündeme gelir.

### 1.2 Pipeline (her tenant için aynı 6 adım)

`run_daily.py` orkestratörü 6 adımı sırayla çalıştırır (her adım idempotent, ara çıktılar parquet):

```
01_ingest_actual   OneDrive DD.MM klasöründen müşteri CSV'si → master.parquet (upsert)
                   └─ update_oof_history() + update_actuals_log() (D+1 yük dalgası)
02_fetch_weather   Open-Meteo Forecast API, 14 istasyon (Muğla/Denizli/Aydın) → weather_fc_live.parquet
                   └─ weather_history.parquet upsert + update_actuals_log_weather() (~D+6 düzelir)
                   └─ NaN gap auto-repair (fix_weather_history.py, 2026-07-07 eklendi)
03_build_features  master + weather_history + weather_fc → Boray DataManager → feature_matrix.parquet
                   └─ ⚠ global dropna monkeypatch hâlâ var (teknik borç #3, kalmadı)
04_predict_48h     4 model recursive (T+0→T+1→T+2): XGB + LGBM + CatBoost(ops.) + Chronos-2(LoRA)
                   └─ Stacking: Rolling Ridge (OOF, 60 gün) → frozen Ridge → basit ortalama (fallback)
                   └─ Holiday override: hafta içi tatilde CatBoost solo (ADM'de CAT=0'a karantinaya alındı)
05_postprocess     Holiday substitution (donmuş alpha) + PV bias correction (donmuş lookup, T1/T2 ayrı)
                   └─ GDZ'de bias correction WORSEN ettiği için pass-through (GDZ_LIVE_PORTING_PLAN.md:226-232)
06_deliver         T+2 günü → Excel teslim + tüm 48h → output/archive/*.parquet
                   └─ write_forecast_log(ctx) + rebuild_duckdb_views() + backup_logs_zip()
                   └─ build_daily_scorecard() + check_alerts()  (non-blocking)
```

Modeller her gün **yeniden eğitilir** (son ~22.000 saat, `MAX_TRAIN_SIZE` concept-drift kapağı) ve
run sonrası `models/archive/<run_id>/`'a kopyalanır (90 gün tut, sonra `prune_archive()` siler).

### 1.3 Mevcut loglama/kayıt (Faz 0 + Faz 1 sonrası)

| Mekanizma | İçerik | Granülerlik | Durum (2026-07-07) |
|---|---|---|---|
| `%LOCALAPPDATA%/<edas>_live_logs/forecast_log/` | saat × run, full şema (deltas, flags, meta ağırlık) | Saatlik | ✅ Çalışıyor — ADM 4 target_date, GDZ 2 target_date dolu |
| `%LOCALAPPDATA%/<edas>_live_logs/actuals_log/` | y_actual, wx_*_actual, data_quality_flag | Saatlik | ✅ ADM 33 gün (06-05→07-07), GDZ 30 gün |
| `monitoring.duckdb` (her tenant ayrı) | forecast_log_v / actuals_log_v (dedup QUALIFY view) | — | ✅ Çalışıyor |
| `logs/backup/*.zip` | günlük log yedeği → OneDrive | Günlük | ✅ 3 zip (07-05/06/07) |
| `daily_scorecard` (DuckDB'de türetilmiş) | MAPE/WAPE/RMSE/ME + saat-blok + model bazlı + corrector + robust_z | Günlük | ✅ `scorecard.py` çalışıyor ama UI okumuyor |
| `logs/alerts/<date>.json` | z>3 alarmı | Günlük | ⚠ Scaffold var, henüz alarm fir etmedi (ısınma modunda) |
| `logs/<date>_summary.json` | run_id, config_hash, adım süreleri | Run başına | ✅ |
| `data/oof_history.parquet` | date, hour, actual, 5 model tahmini | Saatlik | ✅ Çalışıyor (boyut küçük — sadece dün tahmin ⋈ bugün actual) |
| `output/archive/*_full48h.parquet` | Tüm 48h, model bazlı tahminler | Saatlik | ✅ ADM 6 arşiv (07-01→07-07), GDZ 1 arşiv |
| `models/archive/<run_id>/` | model dosyaları + manifest.json + config snapshot + feature_matrix | Run başına | ✅ ADM 5 run, GDZ 2 run arşivli |
| `verdicts.csv` | insan verdict kodu | Kötü gün başına | ❌ YOK (scorecard `_load_verdicts()` boş DataFrame döndürüyor) |
| `known_events.csv` | kesinti/arıza/etkinlik | Olay başına | ❌ YOK (config path tanımlı ama dosya yok) |

### 1.4 Config ve versiyonlama — tekrarlanabilirlik durumu

- **Kod git'te.** ✅ 13 commit (07-04 → 07-07). `.gitignore` data/output/logs/ ignore ediyor, donmuş
  kalibrasyon artefaktları (`models/*.json`) `!` pattern'le geri include edildi.
- **`config_hash`** = `config_live.py` sha256 ilk 8 karakteri (`run_context.py:66-70`). Her run'a damgalıyor.
- **Model arşivi** `models/archive/<run_id>/` + `manifest.json` (model_versions hash, chronos adapter
  fingerprint, feature_snapshot_ref). Geçmiş model durumu geri getirilebilir.
- **HPO parametreleri** JSON'larda. 2026-07-07'de CAT HPO JSON'larının elden bozulduğu fark edildi,
  `git checkout` ile restore edildi (commit `49f9452`) — git'in değeri kanıtlandı.
- **`requirements.txt`** 586 satır tam pin. ✅
- **GDZ config:** `config_live_gdz.py` ADM'i import + GDZ override (hava noktaları/sütunları `gdz talep/config.py`'den).

### 1.5 Backtest / analiz script envanteri (çoğu untracked — Faz 2b'de track edilmeli)

| Script | İşlev | Tracked? |
|---|---|---|
| `backtest_30d.py` | 30-gün as-of T+2 backtest (idempotent, skip existing) | ✅ commit `ae0443f` |
| `analyze_models_30d.py` | Per-model 30-gün analiz → `model_analysis_report.csv` | ✅ |
| `asof_regen.py` | "Geçmişte olsaydık" replay — master'ı issue date'e kes, 03-06 yeniden koştur | ❌ untracked |
| `backtest_7d.py` | 7-gün as-of backtest (perfect-prog weather) | ❌ untracked |
| `backtest_tomorrow.py` | Gerçek teslim günü backtest (T+1, 2-gün-ahead; `backtest_7d`'nin T+2 bug'ını düzeltir) | ❌ untracked |
| `perfect_prog_rerun.py` | Faz 1: model-vs-meteoroloji ayrıştırması (actual weather rerun) | ❌ untracked |
| `export_hourly_mape_7d.py` | Saatlik MAPE kırılımı (forecast_log_v ⋈ actuals_log_v, read-only) | ❌ untracked |
| `backfill_logs.py` | forecast_log/actuals_log kısmi backfill (y_pred/actual geriye dönük; delta/meta sadece ileri) | ❌ untracked |

**Eksik:** walk-forward backtest koşullarının forecast_log/actuals_log'a **kaydedilmesi** yok — bu
scriptler ayrı parquet çıktılar üretiyor (`oof_30day.parquet`, `model_analysis_report.csv`), log
şemasına yazmıyor. Faz 2b bunu düzeltmeli: backtest koşuları da `forecast_log`'a `run_type='backtest'`
damgasıyla yazılsın ki dashboard canlı + backtest'i tek yerde görsün.

### 1.6 Zayıf noktalar / teknik borç envanteri (güncellenmiş)

1. ~~Sessiz fallback'ler attribution'ı bozuyor~~ → **KISMEN ÇÖZÜLDÜ:** `chronos_ok`/`cat_present`
   flag'leri forecast_log'a yazılıyor (Faz 0). Ama Chronos patlayınca hâlâ `CHRONOS_Pred`'e XGB
   kopyası yazılıyor — davranış değişmedi, sadece artık görünüyor. Açık Soru #8 hâlâ açık.
2. ~~`log_daily_mape` yanıltıcı (kümülatif)~~ → **ÇÖZÜLDÜ:** `run_daily.py:124-125` legacy çağrı
   emekliye ayrıldı, yerini `build_daily_scorecard()` aldı (Faz 1).
3. **Global monkeypatch** (`03_build_features.py:47` `dropna` no-op) — hâlâ duruyor. 2026-07-07'de
   `_suppress_dropna` forecast row koruması eklendi (commit `2dcfa4a`) ama kök neden değil.
4. **mtime hack** (`03:213-221` sahte `_tmp_combined.xlsx` + `os.utime`) — hâlâ duruyor.
5. **Ölü config:** `ENABLE_WEEKEND_SPLIT_*` flag'leri hâlâ kullanılmıyor. CAT `CALIBRATED_ENSEMBLE_WEIGHTS=0`
   yapıldı (karantina) ama config'te yolu duruyor.
6. **Hata yönetimi = maskeleme:** geniş `except`'ler hâlâ var; ama artık `check_alerts()` + summary.json
   + forecast_log var, sessizlik azaldı. **Alerting kanalı hâlâ yok** (Açık Soru #3 açık).
7. ~~OneDrive bağımlılığı parquet için riskli~~ → **ÇÖZÜLDÜ:** logs/ OneDrive dışına taşındı. Veri
   girişi (`LIVE_DATA_DIR`) hâlâ OneDrive'da — bu kabul edilebilir (CSV tek seferlik okuma).
8. **Test yok** — hâlâ. Demo-hazırlık haftası 6+ bug fix'i test olmadan yakalandı; regressyon riski birikiyor.
9. ~~UI izleme yapmıyor~~ → **Hâlâ doğru:** UI 3 operasyon tabında, İzleme sayfası YOK (Faz 3 gap).
10. **YENİ — master bayat hava kolonu bug'ı** (2026-07-07 keşfi, commit `6265b35`): master.parquet'te
    stale weather `_actual` kolonları fresh weather_history merge'inde kazanıyordu → eğitim verisi
    8 gün geride kalıyordu. Düzeltildi ama kök neden (master'a weather yazılması) izlenmeli.
11. **YENİ — clone duplikasyonu:** ADM ve GDZ `live/` dizinleri ~%90 aynı kod. Bug fix bir tarafta
    yapıldığında diğer tarafa propagate edilmesi manuel. 3. EDAŞ'ta bu sürdürülemez.
12. **YENİ — GDZ eksikleri:** oof_feedback (rolling ridge) yok, scorecard yok, ensemble kalibrasyonu
    yok (equal average), Chronos LoRA kullanılmıyor (zero-shot), UI disabled.

**Özet hüküm:** Gözlemlenebilirlik ve tekrarlanabilirlik katmanı (Faz -1/0/1) **oturdu**. Sıradaki
en kritik gap: **Faz 3 İzleme/Analiz dashboard** — veri hazır, gösteren yok. Canlı akış 8 Temmuz'da
başlıyorsa insanların bakacağı yer olmalı. İkinci gap: backtest sonuçlarının log altyapısına
bağlanması (Faz 2b) ki geçmişe dönük analiz tek yerde toplansın.

---

## 2. Hedef Mimari (güncellenmiş)

### 2.1 Katmanlar

```
┌────────────────────────────────────────────────────────────────┐
│  UI: Streamlit multipage                                       │
│   ADM: Operasyon / İzleme&Analiz / Deney   ← Faz 3 (KRİTİK)   │
│   GDZ: Operasyon / İzleme&Analiz (Faz 6)                       │
├────────────────────────────────────────────────────────────────┤
│  Analiz: daily_scorecard ✅, robust-z ✅, window_report ✅,     │  ← Faz 1 DONE
│  perfect-prog ✅, triyaj (partial), CUSUM/PSI/shadow (Faz 4)   │
├────────────────────────────────────────────────────────────────┤
│  Log deposu: forecast_log ✅ + actuals_log ✅ (parquet)         │  ← Faz 0 DONE
│  + known_events.csv (YOK) + monitoring.duckdb ✅ (her tenant)   │
├────────────────────────────────────────────────────────────────┤
│  Pipeline: run_daily 6 adım ✅ + run_id/config_hash ✅          │  ← Faz -1/0 DONE
│  + backtest runner (Faz 2b) + shadow runner (Faz 4)             │
├────────────────────────────────────────────────────────────────┤
│  Tenant: ADM (adm live/) + GDZ (gdz talep/live/) — clone+sub   │  ← Faz 5 DIVERGENT
│  Temel hijyen: git ✅, requirements pin ✅, model arşivi ✅     │  ← Faz -1 DONE
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Logging şeması (Faz 0 kilitli, uygulandı)

| Tablo | Grain | Kritik alanlar | Durum |
|---|---|---|---|
| `forecast_log` | saat × run | 4 model ham tahmini, `chronos_ok`/`cat_present`, meta ağırlıklar + method, `y_pred_ens_raw`, `override_delta`/`subst_delta`/`pv_bias_delta`, `y_pred_final`, `wx_*_fcst`, `day_type`, `config_hash`, `model_versions`, `run_id`, `edas_id` | ✅ Yazılıyor |
| `actuals_log` | saat | `y_actual`, `wx_*_actual`, `data_quality_flag`, `known_event` | ✅ y_actual D+1, wx ~D+6 |
| `daily_scorecard` | gün | MAPE/WAPE/RMSE/ME, saat-blok MAPE'leri, model bazlı MAPE, corrector katkısı bps, `temp_fcst_error`/`ghi_fcst_error`, `robust_z`, `verdict_code` | ✅ Türetiliyor (verdict_code null) |
| `known_events.csv` | olay | edas_id, ts_start, ts_end, kategori, not | ❌ Boş |

Depolama: `%LOCALAPPDATA%/<edas>_live_logs/forecast_log/edas_id=X/target_date=YYYY-MM-DD/run_<issue>.parquet`
+ `monitoring.duckdb` (her tenant ayrı) + günlük zip → `logs/backup/`. Hacim DuckDB sınırlarının
çok altında. **Varsayım (doğrulandı):** tek makinede yıllarca sorun yok.

### 2.3 Deney/versiyon katmanı

- **Git repo** ✅ — `config_hash` her run'a damgalıyor, promotion günü hash değişimi changepoint
  analizi için otomatik işaret.
- **Model artefakt arşivi** ✅ — `models/archive/<run_id>/` + manifest.json + feature snapshot.
  MLflow Model Registry bu ölçekte hâlâ opsiyonel (Açık Soru #4 kapatıldı).
- **MLflow'un yeri:** offline deneyler için uygun; production günlük skorları `daily_scorecard`'a.

---

## 3. Faz Faz Yol Haritası (güncellenmiş)

### Faz -1 — Temel Hijyen — ✅ TAMAMLANDI (2026-07-04, commit `b714150`)
- git repo, .gitignore, requirements.txt, `run_context.py` (run_id/config_hash/archive/prune), `logs/` OneDrive dışında.

### Faz 0 — Günlük Log Altyapısı — ✅ TAMAMLANDI (2026-07-05, commit `4ff8e40`)
- `src/forecast_logger.py` (480 satır, full pyarrow şema + DuckDB views + zip backup).
- `04`'te snapshot/flag'ler, `05`'te delta kolonları, `01`'de actuals_log, `02`'de weather actuals_log.
- LocalAppData'ya yazıldı (Açık Soru #2 kapatıldı: opsiyon (b) seçildi).

### Faz 1 — daily_scorecard + 7/30/365 — ✅ TAMAMLANDI (2026-07-05, commit `4ff8e40`)
- `src/scorecard.py` (295 satır): MAPE/WAPE/ME/RMSE + saat-blok + model bazlı + corrector katkısı
  + robust-z (median/MAD 30g, tatil ayrı baseline, ısınma modu) + `window_report(7/30/365)` + `check_alerts()`.
- `perfect_prog_rerun.py` manual tetiklenen v1 ✅.

### Faz 2 — "Kötü Gün" Triyaj Protokolü — ⚠️ PARTIAL
- **Var:** `check_alerts()` + `logs/alerts/` scaffold + scorecard'da `verdict_code` kolonu (null) +
  `_load_verdicts()` / `_load_known_events()` (boş döner, graceful).
- **Yok:** `verdicts.csv` (insan verdict girişi), `known_events.csv` (saha olayları), otomatik
  triyaj raporu üreticisi, 2-3 geçmiş kötü gün provası.
- **Öner:** 8 Temmuz canlı akış başlayınca ilk kötü günde triyaj protokolünü provo — `known_events.csv`
  iskeletini doldur (sahha ekibiyle temas, Açık Soru #6). Otomatik raporu sonra yaz.

### Faz 2b — Backtest Log Entegrasyonu + 1-7 Temmuz Doldurma — YENİ, ~1-2 gün
- **Ne:** (a) backtest scriptlerini (`backtest_30d/7d/tomorrow`, `asof_regen`) `forecast_log`'a
  `run_type='backtest'` damgasıyla yazacak şekilde güncelle — böylece dashboard canlı + backtest'i
  tek tabloda ayırt edip gösterebilir. (b) **1-7 Temmuz için walk-forward backtest koştur**
  (her gün için sadece o güne kadar olan veriyle — gerçekçi), forecast_log + actuals_log'u doldur.
  (c) Scriptleri git'e track et (çoğu untracked).
- **Neden:** Kullanıcı 8 Temmuz'da canlı başlıyor ama "1 Temmuz'dan başlamışız gibi backtest yapıp
  boş durmasın" diyor. Walk-forward = gerçek canlı akışı simüle eder, MAPE dürüst. Backtest
  sonuçları log'a yazılırsa Faz 3 dashboard ilk açılışta 7 günlük geçmişle karşılaşır (sıfırdan değil).
- **Bağımlılık:** Faz 0 (log şeması) ✅. `asof_regen` zaten as-of replay yapıyor — `write_forecast_log`
  çağrısı eklenebilir. Compute: her gün ~dakikalar (GBDT-only), Chronos dahilse ~saat. **Varsayım:**
  7 gün walk-forward tek makinede bir akşamda biter.
- **Not:** `backtest_tomorrow.py`'nin T+2 vs T+1 bug'ı (yanlış gün ölçümü) düzeltildi — backtest
  metric'leri **teslim edilen günü** (T+1) ölçmeli, T+2'yi değil. Faz 2b'de bu doğrulansın.

### Faz 3 — İzleme & Analiz Dashboard (KRİTİK, öncelik yükseldi) — ❌ YAPILMADI, ~5-7 gün
- **Ne:** Mevcut Streamlit paneline multipage yapı. **3 sayfa:**
  1. **Operasyon** (mevcut 3 tab'ın taşıdığı — Veri Durumu / Veri Yükleme / Tahmin Üret) — GDZ
     subprocess entegrasyonu stabil hale gelsin.
  2. **İzleme & Analiz** (YENİ, Faz 3'ün çekirdeği) — `monitoring.duckdb`'den okur:
     - **Gün seçici** + **model seçici** (kullanıcının ana talebi: "günü istediğim model sonuçlarını
       detaylıca geçmiş günlerle analiz")
     - Tahmin vs actual üst-üste çizgi grafik (Plotly), saatlik MAPE bar, residual plot
     - Scorecard trendi (günlük MAPE/WAPE/RMSE/ME, 7/30/365 pencere)
     - Saat-blok ısı haritası (gece/sabah/PV/akşam × gün)
     - Model bazlı MAPE karşılaştırma (XGB/LGBM/CAT/Chronos/Ensemble)
     - Corrector katkısı (override/subst/PV delta bps) + meta ağırlık trendi
     - Robust-z alarm banner'ı + `verdict_code` (boş olsa da yer hazır)
     - Perfect-prog rerun karşılaştırması (model vs meteoroloji hata payı)
     - **Tenant seçici** (ADM / GDZ) — veri altyapısı `edas_id` ile hazır, UI'da dropdown
     - **Backtest vs canlı ayıracı** (`run_type` kolonu — Faz 2b'ye bağımlı)
  3. **Deney** (sonra, Faz 4'le) — shadow karşılaştırma, verdict dağılımı, backlog. Faz 3'te iskelet.
- **Neden:** 8 Temmuz'da canlı akış başlıyor — her gün 9 Temmuz tahmini → 10 Temmuz actuals. İnsanların
  bakacağı yer yoksa loglar karanlıkta kalır. Bu, kullanıcının "kendi geçmiş verileri/tahmin ile
  karşılaştıran grafik, günü istediğim model sonuçlarını detaylıca geçmiş günlerle analiz" talebinin
  tam karşılığı. Veri altyapısı hazır (Faz 0/1), tek gap UI.
- **Bağımlılık:** Faz 1 ✅ (gösterilecek veri var). Faz 2b ile paralel/ardıl yürüyebilir (backtest
  doldurmadan da açılır, ama 2b önce yapılırsa ilk açılışta 7 gün geçmiş olur). Faz 2 bağımsız.
- **Karar (korumalı):** Streamlit + Plotly, multipage. Yeniden değerlendirme tetiği: >10 eşzamanlı
  kullanıcı veya real-time refresh. (Açık Soru #1 kapatıldı.)

### Faz 4 — CUSUM + Drift + Hata Madenciliği + Champion-Challenger — ❌ YAPILMADI, ~1.5-2 hafta
- **CUSUM** günlük ME üzerinde (k=0.5σ, h=4-5σ; σ = son 90 gün, tatil hariç) — sinsi bias kayması alarmı.
- **PSI/KS input drift** aylık (sıcaklık, GHI, yük; PSI>0.2) + concept drift ayrımı (aynı sıcaklık
  bin'lerinde residual yıl-yıl — input aynı + hata farklı = ilişki değişmiş).
- **Hata madenciliği:** residual'ları gün tipi × saat bloğu × sıcaklık/GHI rejimi hücrelerinde grupla.
- **Champion-challenger:** 3 kapı — (1) backtest: sabit fold + Diebold-Mariano (Harvey düzeltmeli)
  + en-kötü-10-gün guardrail; (2) shadow: challenger her gün aynı inputla koşar, `forecast_log`'a
  `shadow_pred_<id>` kolonu, min 28 gün; (3) kontrollü geçiş: probation 14g, z eşiği 2.5, ters shadow.
- **Bağımlılık:** Faz 0-1 ✅. Shadow runner 04-05'i challenger config ile 2. kez koşturur — **Varsayım:**
  günlük ek ~1 saat CPU kabul (Chronos yavaş; challenger GBDT-only ise çok daha hızlı).

### Faz 5 — Çoklu EDAŞ Genişleme — ⚠️ DIVERGENT (clone ile yapıldı, parametrik değil)
- **Gerçekleşen:** ADM + GDZ iki ayrı `live/` klonu, UI GDZ'yi subprocess ile çağırıyor. forecast_log
  şeması `edas_id` taşıyor (izleme katmanı agnostik) ama *kod* duplikatlı.
- **Neden divergent:** UI modüllerinin bare import'ları `sys.modules` cache'ini kirletiyor — aynı
  process'te iki tenant'ın modüllerini yüklemek sessiz hata riski. Clone+subprocess bu riski sıfırlar.
- **Maliyet:** bug fix'ler manuel propagate. 3. EDAŞ'ta sürdürülemez.
- **Yeniden değerlendirme tetiği (Açık Soru #9, YENİ):** 3. EDAŞ somutlaşınca clone→parametrik
  refactor gündeme gelir. O zamana kadar: (a) her clone kendi `run_context`/`forecast_logger`'ını
  taşır (zaten öyle), (b) Faz 3 dashboard tek process'te iki tenant'ın `monitoring.duckdb`'sini
  dropdown ile açabilir (dosya yolundan, import kirletmez) — bu, izleme tarafında parametrik vizyonu
  *şimdiden* korur. (c) ortak kod (`scorecard.py`, `forecast_logger.py` şema) bir `shared/` pakete
  taşınıp her klon import edebilir — refactor maliyeti düşürür.

### Faz 6 — GDZ Tamamlama — ⚠️ PARTIAL (~1 hafta)
GDZ_LIVE_PORTING_PLAN.md'nin 6 adımından 6.1/6.2 done, gerisi bekliyor:
- ✅ 6.1 `run_context` (archive/prune/manifest)
- ✅ 6.2 `forecast_logger` (parquet + DuckDB + zip)
- ❌ 6.3 `oof_feedback` (rolling ridge) — GDZ'de yok; stacking frozen Ridge kullanıyor
- ❌ 6.4 `scorecard` + `check_alerts` — GDZ'nin `live/src/`inde `scorecard.py` yok
- ❌ 6.5 ensemble kalibrasyonu — GDZ equal average (`t2_df.mean(axis=1)`), `CALIBRATED_ENSEMBLE_WEIGHTS` placeholder
- ❌ 6.6 Chronos LoRA — GDZ zero-shot, kendi LoRA'sı kullanılmıyor (biliberateli kararla)
- ❌ UI for GDZ — disabled, "🚧 Yakında"
- **Öner:** 6.4'ü Faz 3 dashboard'a bindir (GDZ monitoring.duckdb'sini dropdown'da göster, scorecard
  türetme ADM'in `scorecard.py`'si tenant-parametreli hale getirilip her iki duckdb'ye koşabilir).
  6.3/6.5/6.6 GDZ'nin akademik araştırma köküne bağlı — ayrı backlog.

### Sürekli (faz değil): Canlı Ops Ritmi — 8 Temmuz 2026'dan itibaren başlar
Kullanıcının tanımladığı akış:
- **Her gün (8 Temmuz'dan):** `run_daily.py` koş → T+1 (örn. 9 Temmuz) tahminini üret → Excel teslim.
- **Ertesi gün (örn. 10 Temmuz):** dünün actuals'ı OneDrive'a düşer → 01_ingest actuals_log'u günceller
  → scorecard 9 Temmuz için türetilir → Faz 3 dashboard'da 9 Temmuz actual vs tahmin görünür.
- **Günlük 5-10 dk bakış (Faz 3 tamamlanınca):** dashboard'da dünkü MAPE, robust-z, model katkısı.
- **Haftalık 30-45 dk:** 7-gün pencere, saat-blok ısı, model trendi.
- **Aylık 2-3 saat:** 30-gün pencere, verdict dağılımı, deney backlog güncelleme.
- **Alerting kanalı hâlâ açık (Açık Soru #3)** — 8 Temmuz'a kadar minimal: pipeline hatası → e-posta?

---

## 4. Sonraki Adımlar (8 Temmuz canlı start'ına odaklı, sıralı)

**8 Temmuz'a kadar (1 gün):**
1. **Faz 2b — backtest doldurma:** 1-7 Temmuz için walk-forward backtest koştur, forecast_log +
   actuals_log'a `run_type='backtest'` damgasıyla yaz. Böylece Faz 3 dashboard ilk açılışta 7 gün
   geçmişle karşılaşır. Scriptleri track et.
2. **Faz 3 — İzleme sayfası MVP:** gün/model seçici + tahmin-vs-actual grafik + scorecard trend +
   tenant dropdown. Geri kalanını (ısı haritası, perfect-prog, deney) sonra ekle.
3. **Canlı akış provası:** 8 Temmuz'da `run_daily.py` manuel koştur → 9 Temmuz teslimi → 10 Temmuz
   actuals geri beslemesi → dashboard'da görünür mü doğrula.

**8-14 Temmuz (canlı ilk hafta):**
4. Her gün run + teslim + ertesi gün actuals. Dashboard'da günlük bakış.
5. İlk kötü günde (z>3 fir ederse) Faz 2 triyaj provası — `known_events.csv`'i doldurmaya başla.
6. Faz 3 dashboard'ı geri kalan widget'larla tamamla (ısı haritası, model MAPE, corrector, perfect-prog).

**15-31 Temmuz:**
7. Faz 2 triyaj otomasyonu (rapor üreticisi) + verdicts.csv süreci otur.
8. Faz 4 CUSUM (önceki 14 günlük ME dizisi ısınmaya başlar) + PSI aylık rapor iskeleti.
9. GDZ Faz 6.4 (scorecard tenant-parametrik) + 6.3/6.5 backlog'a.

Bu planın sonunda (Temmuz sonu): canlı akış oturmuş, her gün dashboard'da analiz, kötü gün triyajı
çalışır, CUSUM ısınıyor, iki tenant'ın da izlemesi tek dashboard'da.

---

## 5. Açık Sorular / Kararlar (güncellenmiş)

1. ~~Streamlit mi Dash mi?~~ **Kapandı:** Streamlit + Plotly, multipage. Tetik: >10 eşzamanlı kullanıcı.
2. ~~Log storage nerede?~~ **Kapandı:** LocalAppData + günlük zip OneDrive'a (opsiyon (b)).
3. **Alerting kanalı (HÂLÂ AÇIK, acil):** 8 Temmuz canlı start'ında pipeline hatası nereye düşsün?
   E-posta / Teams webhook / sadece dashboard? Minimum: `run_daily.py` hataında e-posta.
4. ~~Model artefakt arşivi vs MLflow?~~ **Kapandı:** dosya-tabanlı arşiv (yeterli).
5. **Shadow compute bütçesi (açık):** Challenger günlük tam koşu (Chronos dahil) ~1 saat. Kabul mü,
   GBDT-only mi başlasın? Faz 4 başında karar.
6. **`known_events` sahibi (açık, organizasyonel):** Kesinti/arıza bilgisini kim girecek? Sahha ekibi
   teması gerek. 8 Temmuz'da ilk kötü günde provo.
7. **Probabilistic forecasting (backlog):** Faz 4 sonrası araştırma maddesi.
8. **Chronos fallback politikası (HÂLÂ AÇIK):** `chronos_ok` flag'i var ama davranış değişmedi
   (Chronos patlayınca XGB kopyası). Drop yerine kopya mı? 8 Temmuz canlıda riskli — öncesinde karar.
9. **Clone→parametrik refactor zamanı (YENİ):** 3. EDAŞ somutlaşınca. O zamana kadar `shared/`
   paket (scorecard/forecast_logger şema) extracts maliyeti düşürür. Şimdilik acil değil.
10. **Backtest metric doğru gün (YENİ, doğrulanmalı):** `backtest_tomorrow.py` T+1 (teslim günü)
    ölçüyor, `backtest_7d` T+2 (bug'lı). Faz 2b'de tüm backtest'ler teslim gününü ölçmeli.
11. **GDZ LoRA (YENİ):** GDZ kendi Chronos LoRA'sını biliberateli kullanmıyor (zero-shot). Doğrulama
    backlog'ta — performans yetersizse LoRA eğitimi gündeme gelir.

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

*Doküman: ilk 2026-07-04, güncellendi 2026-07-07. Kod tabanı keşfi 2026-07-07 tarihli canlı dizin
+ `gdz talep/live/` üzerinden; 13 commit, 5 ADM arşivli run, 2 GDZ arşivli run, 33 günlük actuals_log.*
