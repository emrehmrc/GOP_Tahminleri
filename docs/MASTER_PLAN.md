# STLF Live — Backend "Ürünleştirme" Master Planı (rev2, 2026-07-13)

## Faz Durumu

| Faz | Konu | Durum |
|-----|------|-------|
| 0 | Güvence altına alma (commit + baseline + yedek) | ⬜ |
| 1 | Güvenilirlik + veri kalitesi | ⬜ |
| 2 | Doğruluk paketi (Pazar problemi + tenant feature + learning ensemble) | ⬜ |
| 3 | Multi-tenant çekirdek | ⬜ |
| 4 | Deliverable'lar (Excel + Diagnostic + LLM-export + Mail) | ⬜ |
| 5 | Hijyen + dokümantasyon | ⬜ |
| 6 | Otomasyon | ⬜ |

## Context

ADM + GDZ için günlük D+1/D+2 talep tahmini üreten sistem canlıda ama güvenilir değil ve doğruluk
sorunları var. **Kapsam kararı (2026-07-13): UI/dashboard TAMAMEN kapsam dışı — Emre Bey geliştiriyor.
Biz sadece backend:** pipeline, modeller, ensemble, loglama/izleme verisi, raporlar (Excel/HTML/mail),
codebase düzeni, otomasyon.

**12 Temmuz Pazar olayı (planın yeni odağı):** ADM %4.9, GDZ %2.8 MAPE — en kötü gün. Pazar günü yük
bariz düştü, modeller yakalayamadı. Veri teşhisi (13 Tem):
- GDZ 07-12 T+2 final %2.80 (doğrulandı). Desen: gece 00-05 %4-5 ALTINDA (Pazar gece düşüşünü fazla
  öngörmüş), akşam 21-23 %5-6 ALTINDA (akşam yük yüksek kalmış — memory'deki "4 model de akşam pikini
  düşük tahmin ediyor" bulgusuyla tutarlı), öğlen %2-3 üstünde.
- ADM 07-12 actual'ı master'a HENÜZ tam girmemiş / saat-tekrarı şüphesi var (analiz sırasında aynı gün
  önce 24 saat sonra tek saat okundu → ingest veri-kalite kapısı ihtiyacının kanıtı). Kullanıcı ölçümü:
  %4.9, gün geneli AŞIRI tahmin (düşüşü kaçırma).
- Domain bilgisi: ADM ağırlıklı ticarethane/turizm → sıcaklık duyarlılığı DÜŞÜK, haftalık profil
  (lag168) daha belirleyici olmalı. GDZ sıcaklığa daha duyarlı. Tenant'a özel feature profili gerekli.
- ADM'de T+2 tahmini (07-10 çıkışlı) T+1'den (07-11 çıkışlı) DAHA İYİydi — daha taze bilgi daha kötü
  sonuç vermiş; recursive T+1 zinciri / güncel-gün özellikleri şüpheli, incelenecek.
- CAT her yerde kötü (ADM T+2 %6.8, ME -106) — ağırlığı zaten 0.05, ama OOF/ensemble'a gürültü katıyor.

Diğer teşhisler (rev1'den geçerli): loglama iki yolda da sessizce yutulabiliyor; Rolling Ridge 168 temiz
OOF örneği bekliyor ve hiç devreye girmiyor (fiilen statik ağırlık); `stacking_strategies.py` canlıya
bağlı değil; scorecard per-model kırılımı sadece günlük agrege; 07 os.system ile 08'i ikinci kez
çağırıyor; iki EDAŞ iki klon codebase; output/ karışık; AGENTS.md bayat; 21+ dosya commit edilmemiş.

Kullanıcı kararları: tek multi-tenant codebase'e kademeli geçiş; Excel tam tarihçe sağa büyüyen;
otomasyon son faz; LLM analizi = chat'e yapıştırılacak export (API değil); mail İLK AŞAMADA SADECE MRC.

---

## Yürütme modeli

> **İlk adım (onay sonrası):** bu plan `docs/MASTER_PLAN.md` olarak commit edilir — o kadar.
> Fazlar ayrı oturumlarda, "Faz X'i uygula" komutuyla koşulur.

- **`ui/` klasörüne DOKUNULMAZ** — Emre Bey'in alanı. UI↔backend kontratı şunlardır ve korunur:
  `monitoring.duckdb` view'leri (`forecast_log_v`, `actuals_log_v`, `daily_scorecard`),
  `logs/<date>_summary.json`, `data/run_context.json`, `output/` dosya düzeni, pipeline modüllerinin
  `run()` imzaları. Bunlarda breaking change gerekiyorsa önce Emre'ye haber + geçiş süresi.
- Her faz ayrı oturum; faz bitişi = `pytest` yeşil + 1 gerçek/dry-run koşu + anlamlı commit.
- Canlı sistem kuralları: `04_predict_48h.py` asla standalone koşulmaz; `master.parquet` /
  `weather_history.parquet` asla in-place mutate edilmez; LightGBM save Unicode-path workaround'u
  korunur; subprocess stdout=PIPE yerine dosyaya yönlendirilir; riskli işlem öncesi parquet yedeği
  `data/backups/`e.

---

## FAZ 0 — Güvence altına alma (yarım gün)

1. Uncommitted working set mantıklı parçalara bölünerek commit edilir (monitoring/ + shim'ler; tests/;
   diagnostic_core; stacking/optimize; UI dosyaları OLDUĞU GİBİ tek commit — içeriğine dokunmadan;
   doc'lar). `git tag baseline-2026-07-13`.
2. `pytest tests/` baseline sonucu kaydedilir.
3. `data/backups/` açılır; `master.parquet`, `weather_history.parquet`, `oof_history.parquet` kopyalanır.
4. Kullanılmıyorsa `.claude/worktrees/beautiful-shirley-bcbf48` kaldırılır.

## FAZ 1 — Güvenilirlik + veri kalitesi: "hiçbir tahmin sessizce kaybolmaz, hiçbir bozuk veri sessizce girmez" (1-2 gün)

Dosyalar: `run_daily.py`, `monitoring/forecast_logger.py`, `src/run_context.py`,
`pipeline/01_ingest_actual.py`, `pipeline/07_report_excel.py`.

1. **Ortak `finalize_run(ctx)`:** step-06 sonrası kuyruk (forecast_log → duckdb views → backup →
   reconcile → scorecard → alerts → 07 → 08) tek fonksiyona. run_daily bunu çağırır; UI'daki kopya
   blok Emre'yle koordineli olarak aynı fonksiyona yönlendirilir (ui/ dosyasına tek satırlık çağrı
   değişikliği — kontrat gereği haber verilerek; olmazsa run_daily tarafı yeter, UI mevcut haliyle kalır).
2. **Loglama hard-fail + verify-after-write:** `write_forecast_log` hatası koşuyu
   `status="delivered_NOT_LOGGED"` yapar; yazım sonrası parquet geri okunup satır sayısı doğrulanır;
   `no_postproc`/`no_datetime_col` sessiz dönüşleri exception olur; summary.json'a
   `forecast_logged: true/false` alanı.
3. **Ingest veri-kalite kapısı (07-12 ADM bulgusunun cevabı):** `01_ingest_actual` sonrası kontrol —
   duplicate timestamp, eksik saat (gün 24 saat tam mı), negatif/sıfır/outlier değer (son 30 gün aynı
   saat medyanına robust-z), monotonik index. İhlal → koşu uyarıyla işaretlenir, summary'ye
   `data_quality` bloğu, `logs/alerts/`e kayıt. Master'a bozuk gün yazılmışsa raporlanır (otomatik
   silme YOK — karar insanda).
4. **DuckDB view gölgeleme görünürlüğü:** view'e `run_count` kolonu; aynı target_date için birden çok
   canlı run varsa scorecard/summary'de belirtilir.
5. **07/08 düzeni:** 07 içindeki `os.system(...08...)` kaldırılır (çifte diagnostic); bare `except:`ler
   daraltılır; GDZ path'i config'e alınır.
6. **Testler:** forecast_logger write/verify/heal, scorecard join, reconcile, view dedup, veri-kalite
   kapısı, finalize_run happy-path + hata senaryosu. Golden mini-parquet fixture'ları `tests/fixtures/`.
7. **output/ düzeni (tek seferlik migrasyon):** `output/daily/DD.MM.YYYY/` (teslimat seti),
   `output/archive/` (aynen — immutable doğruluk kaynağı), `output/backtest/` (REGEN'ler),
   `output/analysis/`. Tüketici script glob'ları aynı PR'da güncellenir. Çöp temizliği: `Model/`,
   `07_report_excel.py.bak`, `models/live_*_test*`, `output/*Kopya*`; data yedekleri → `data/backups/`.
   (Emre'ye output düzeni değişikliği önceden bildirilir — UI dosya okuyorsa kırılmasın.)

## FAZ 2 — Doğruluk paketi: Pazar problemi + tenant feature profilleri + learning ensemble (3-4 gün)

Planın kalbi. Dosyalar: `src/oof_feedback.py`, `pipeline/04_predict_48h.py`, `pipeline/03_build_features.py`,
`monitoring/scorecard.py`, `backtest_walkforward.py`, `optimize_ensemble_offline.py`, `analyze_models_30d.py`.

### 2a. Ölçüm altyapısı (önce görünürlük)
1. **Naive benchmark scorecard'a:** her gün `mape_naive_lag168` (geçen haftanın aynı günü = tahmin)
   hesaplanır. "Model lag168'den iyi mi?" sorusu kalıcı otomatik cevap bulur — 07-12'de ADM muhtemelen
   naive'in ALTINDA kaldı; bu bir daha görünmez kalmayacak.
2. **Per-model × saat-blok × gün-tipi scorecard:** `mape_{model}` kırılımı HOUR_BLOCKS × daytype
   (hafta içi/Cmt/Paz/özel gün) bazında. `analyze_models_30d.py` REGEN dosyaları yerine
   forecast_log + actuals_log'dan beslenir (canlı veriyle her gün koşabilir hale gelir).
3. **Günlük post-mortem artefaktı (backend; "Günün Karnesi"nin veri hali):** actual gelince
   `output/daily/<gün>/postmortem_<edas>.{md,json}` — dünün tahmini vs actual saatlik, per-model MAPE
   + en kötü 3 saat, naive benchmark farkı, hava tahmini hatası payı (perfect-prog ayrıştırması),
   gün-tipi bağlamı. Emre UI'da gösterebilir; kullanıcı LLM chat'ine yapıştırabilir.

### 2b. Pazar/hafta sonu problemi (12 Tem post-mortem'i + kalıcı çözüm)
1. **Tam post-mortem:** ADM 07-12 actual'ı tam ingest edilince 2a-3 araçlarıyla ADM+GDZ raporu üretilir;
   ADM'de "T+2 > T+1 doğruluk" tersliğinin kökü incelenir (recursive T+1 zinciri, güncel gün feature'ları,
   hava tahmini farkı).
2. **ADM: haftalık-profil vurgusu (lag168 hipotezi):** deneyler walkforward A/B ile:
   - Pazar (ve genel hafta sonu) tahminini lag168-tabanlı baseline'a çeken blend
     (`pred_final = α·pred + (1-α)·lag168_profil`, α gün-tipine göre; sadece backtest kazandırırsa canlıya).
   - Feature önem analizi: ADM modellerinde lag168/336/504 vs sıcaklık önem dengesi; gerekirse ADM
     feature setinde haftalık profil feature'ları güçlendirilir (Pazar-özel etkileşimler).
   - Mevcut hafta sonu split (XGB/LGBM) Pazar'ı Cmt ile aynı torbaya koyuyor — Pazar'a özel split/ağırlık
     denenir (LGBM Sunday boost 2.5 var ama yetmemiş).
3. **GDZ: akşam piki + gece profili:** 21-23 sistematik alçak tahmin (07-12'de yine görüldü) için
   saat-bazlı rezidüel bias analizi; GDZ'de sıcaklık/GHI feature vurgusu korunur. GDZ ensemble hâlâ
   eşit-ağırlık — segment ağırlıklandırmadan (2c) en çok GDZ kazanacak.
4. **Tenant feature profili altyapısı:** feature seti/vurguları TenantConfig'ten yönetilebilir hale
   gelir (ADM=haftalık profil ağırlıklı, GDZ=hava ağırlıklı) — Faz 3'teki multi-tenant işinin öncüsü.

### 2c. Learning ensemble (son 30 günden öğrenen)
1. **OOF beslemesini onar:** `update_oof_history` finalize_run'a bağlanır (hard-fail disiplini);
   walkforward ile son 30 gün OOF backfill (source="backfill" ayrımı korunur); Chronos-fallback
   karantinası gevşetilir (fallback günde diğer 3 modelin OOF'u kullanılır, chronos NaN).
2. **Segment-bazlı adaptif ağırlık:** `hour_block × daytype` segmentlerinde rolling-30g inverse-MAPE
   veya EWA (exponentially weighted average — expert aggregation literatürü). Config anahtarı:
   `ENSEMBLE_STRATEGY = "segment_ewa" | "rolling_ridge" | "inverse_mape" | "static"`; mevcut cascade
   korunur. Dormant `stacking_strategies.py` sınıflarından uygunlar bu arayüze bağlanır.
3. **Governance:** hiçbir ağırlık/strateji değişikliği walkforward A/B raporu
   (`output/analysis/ensemble_ab_<date>.md`) olmadan canlıya girmez; canlıya alma = config + commit.
   CAT'in ensemble'dan tamamen çıkarılması da aynı süreçle değerlendirilir (0.05 ağırlık + kötü OOF).

## FAZ 3 — Multi-tenant çekirdek: "yeni EDAŞ = 1 config" (3-5 gün, kademeli)

1. **TenantConfig genişletme:** istasyonlar, hedef kolon, ağırlık/bias sabitleri, path'ler, HPO
   paramları, frozen artefact'lar, (2b-4) feature profili.
2. **Model registry:** `MODELS = {"xgb": ..., "lgbm": ..., "cat": ..., "chronos": ...}`; `y_pred_{key}`
   kolon adları registry'den türer; şu an 6-8 yerdeki hardcoded literal tek kaynağa iner
   (04_predict, forecast_logger, oof_feedback, scorecard, schema). Yeni model = manager + kayıt + config.
3. **Kademeli ortaklaştırma:** 08 (ortak) → 07 → 06/05 → 03 → 04 → 01/02; her adımda iki tenant'ta
   birer koşu + çıktı diff'i sıfır olmadan ilerlenmez; sonunda `gdz talep/live` = config + data + models;
   subprocess izolasyonu korunur.
4. **"Yeni EDAŞ ekleme tarifi"** `docs/RUNBOOK.md`'ye.

## FAZ 4 — Patron deliverable'ları: Excel + Diagnostic + LLM-export + Mail (2-3 gün)

### 4a. STLF_LIVE_RAPOR.xlsx (`pipeline/07_report_excel.py`)
- Tam tarihçe, sağa büyüyen (kaynak: `output/archive/` + forecast_log; flat xlsx bağımlılığı kalkar).
- 5 sabit tablo bloğu (T1 Realized / T2 D+1 / T3 D+1 sapma% / T4 D+2 / T5 D+2 sapma%), MAPE%/ME
  satırları, renk kuralları (yeşil<3, sarı<6, kırmızı≥6), freeze panes; ADM/GDZ ayrı sheet.
- Terminoloji sözlüğü: 06_deliver "T+2" vs rapor "D+1/D+2" tek tanıma bağlanır, docs'a yazılır.

### 4b. Diagnostic HTML güçlendirme (`src/diagnostic_core.py` — iki EDAŞ birden alır)
Mevcut 7 sekmeye eklenecekler:
1. **Ramp/saat-geçiş kontrolü:** tahmin Δ'ları son 60 günün aynı-saat-geçiş dağılımı P5-P95 bandına karşı.
2. **Gün-geçişi kontrolü:** dün→hedef gün toplam & pik değişimi vs ΔT'den beklenen (P95 aralıklı).
3. **Pik analizi:** pik saati/büyüklüğü vs 4 referans gün; pik kayması uyarısı.
4. **Empirik P10-P90 bandı** ana tahmin eğrisine (son 30 gün saatlik hata dağılımından).
5. **Benzer-gün bulucu:** sıcaklık profili + gün tipi benzerliğiyle en yakın 3 gün overlay.
6. **Gün-tipi bağlam paneli (Pazar dersi):** hedef gün Pazar/özel günse geçmiş 4 aynı-gün-tipi günün
   düşüş oranları tablo + tahminin ima ettiği düşüş oranı yan yana ("geçen 4 Pazar %-4.2 ort. düştü,
   tahminin +%1.5 ima ediyor" tipi uyarı) + özel gün literal overlay + hava tahmini güvenilirlik kutusu.
7. Öneriler sekmesi yeni kontrolleri kart olarak üretir.

### 4c. LLM-ready export
- `output/daily/<gün>/diagnostic_<edas>_<date>_LLM.md`: 48h tahmin, referans gün eğrileri, segment
  duyarlılıkları (MW/°C), ramp anomalileri, P95 aralıkları, per-model son-7g karne (2a'dan), gün-tipi
  bağlamı, hava tahmini + geçmişi — kompakt markdown tablolar + başa gömülü hazır analiz prompt'u.
  Kullanıcı chat'e yapıştırır, saat bazlı düzeltme önerisi alır.

### 4d. Mail (`pipeline/09_email_report.py`)
- Alıcı listesi config/env'e. **İlk aşamada SADECE MRC:** emre.hangul@mrc-tr.com,
  cagatay.bayrak@mrc-tr.com, dataanalyticsteam@mrc-tr.com. Müşteri adresi (talep.tahmin@aydemenerji.com.tr)
  iç doğrulama bitene kadar EKLENMEZ (config'de hazır, kapalı).
- İnsan-onay akışı (awaiting_approval) ve EMAIL_SENT marker aynen korunur.

## FAZ 5 — Codebase hijyeni + dokümantasyon + agent memory (1 gün)

1. `docs/` klasörü: `ARCHITECTURE.md`, `RUNBOOK.md` (operasyon + arıza + yeni EDAŞ tarifi),
   `MASTER_PLAN.md` (bu plan, faz durumlu), `docs/archive/` (eski roadmap/plan/tasarım md'leri).
   `STLF_LIVE_OPS_OZET.html` silinir.
2. AGENTS.md yeniden yazılır: monitoring/ dahil, shim'ler, var-olmayan `src/ensemble_weights.py`
   referansı düzeltilir, "asla yapma" kuralları + **"ui/ Emre'nin alanı, dokunma"** kuralı en üstte;
   detay docs/'a delege.
3. Kökteki `best_params_*_hpo.json` → `models/hpo/` (config path güncellenir).

## FAZ 6 — Otomasyon (creds hazır olunca, 1-2 gün)

1. **Windows Task Scheduler:** sabah otomatik `run_daily.py` (awaiting_approval'da durur, mail atmaz).
2. **Teams Incoming Webhook:** koşu bitti kartı (durum, dünün karne MAPE'si, naive-benchmark farkı,
   sanity uyarıları); hata kartı.
3. **FTP ingest (`pipeline/00_fetch_ftp.py`):** creds gelince; o zamana dek dosya-bekleme modu.

---

## Doğrulama (faz kapanış kriterleri)

- `pytest` yeşil; gerçek koşuda summary.json `forecast_logged: true` + `data_quality` bloğu dolu;
  İzleme verisi (duckdb) günün satırını içeriyor; `output/daily/<gün>/` tam set.
- Faz 2: (i) scorecard'da naive_lag168 kolonu dolu; (ii) walkforward A/B raporları — segment ensemble
  ve Pazar-blend son 30 günde mevcut düzeni MAPE'de geçiyor mu; geçmeyen değişiklik canlıya GİRMEZ;
  (iii) 07-12 post-mortem raporu üretilmiş ve kök neden yazılı.
- Faz 3: her ortaklaştırma adımında iki tenant koşusu, çıktı diff = 0.
- Faz 4: Excel'de 5 tablo/tam tarihçe/freeze pane; diagnostic yeni sekmeler tarayıcıda; LLM .md
  chat testinden geçiyor (kullanıcı denemesi).
- UI kontratı hiçbir fazda kırılmadı (duckdb şema + summary.json alanları geriye uyumlu).

## Riskler / dikkat

- `ui/` dosyalarına dokunulmaz; kontrat değişiklikleri Emre'ye önceden bildirilir.
- OneDrive senkron + LightGBM Unicode-path workaround'ları korunur.
- Faz 3 en riskli: adım adım, diff-sıfır doğrulamasız ilerleme yok.
- `output/` migrasyonu glob kırar — tüketiciler tek PR'da güncellenir, Emre bilgilendirilir.
- Model değişikliklerinde tek hakem walkforward A/B'dir; "hissiyatla" ağırlık değişikliği yasak.
