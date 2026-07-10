# Demo Hazırlık Analizi — ADM & GDZ Canlı STLF Pipeline
**Tarih:** 2026-07-07 (demo yarın) · **Kapsam:** uçtan uca sağlık, 4-model, feature/leakage, UI, MLOps, domain
**Yöntem:** salt-okunur inceleme (kod + loglar + parquet artefaktları). Hiçbir dosya değiştirilmedi, pipeline çalıştırılmadı.

---

## 0. TL;DR — Demo Hazırlık Durumu

| Sistem | Uçtan uca | 4 model | Feature/Leakage | UI | Bugün canlı koştu mu? | Verdict |
|--------|-----------|---------|-----------------|-----|-----------------------|---------|
| **GDZ** | ✅ Tam koştu (15:08) | ✅ 4/4 (LGBM+CAT+XGB+Chronos) | ✅ Temiz (T2_SAFE ayrımı) | ❌ UI'da kapalı (CLI'dan çalışıyor) | ✅ 07-08 teslim üretildi | 🟢 **Demoya hazır (CLI ile)** |
| **ADM** | ⚠️ Bugün tamamlanamadı | ⚠️ 4 eğitiliyor ama ensemble ~2.5 | ✅ Temiz | ✅ UI'da bağlı | ❌ 6 deneme başarısız, son run 04'te kaldı | 🔴 **2 kritik sorun var** |

**Net cevap:** "İkisi de çalışıyor, her gün bam güm koşar" — **şu an DEĞİL.** GDZ bugün temiz koştu ve gerçekten iyi durumda. **ADM'de iki gerçek (silent) sorun var** — bunlar demo öncesi çözülmezse ADM ya hiç tahmin üretmez ya da **son 8 günü görmemiş bayat bir modelle** tahmin üretir.

---

## 1. 🔴 EN KRİTİK BULGU — ADM her gün eklenen dünkü veriyi modele SOKMUYOR (silent)

Bu, senin en çok önemsediğin şeyin — "her gün dünün verisini ekliyorum, ertesi günü tahmin ediyorum" — **ADM tarafında sessizce kırık olduğu** anlamına geliyor.

### Kanıt
- `data/master.parquet` içine 07-06 actual'ı **doğru ingest edildi** (74616 satır, target NaN=0). Buraya kadar sorun yok.
- **AMA** `data/weather_cache/feature_matrix.parquet` (modelin eğitildiği matris) içinde **eğitim verisi 2026-06-28'de bitiyor.** 06-29 → 07-06 arası **8 gün (bu demo haftasının TAMAMI) eğitimden düşmüş.**

```
feature_matrix'te mevcut son eğitim günleri:
  ... 06-26(24) 06-27(24) 06-28(24) 06-29(1)   ← burada bitiyor
06-30, 07-01 ... 07-06 → HİÇ YOK
```

### Kök neden (zincir)
1. Bu `master.parquet` "raw-only 3 kolon" olması gerekirken (kod yorumları öyle diyor) **90 kolon** taşıyor — içinde eski hava (`_actual`) kolonları var. Muhtemelen **07-06 master korupsiyon/restore olayının** kalıntısı (bkz. memory: *master-parquet-korupsiyon-olayi*).
2. Bu master'da **AYDIN hava kolonları 06-29'dan itibaren tamamen NaN** (günde 192 hücre).
3. `pipeline/03_build_features.py:89` — master ile `weather_history` birleştirilirken `suffixes=(None, "_dup")` kullanılıyor → **master kazanıyor.** Doğrudan test ettim:
   - merge sonrası AYDIN NaN: **192/192**
   - `weather_history`'de aynı pencere: **0/192** (yani temiz veri VAR ama atılıyor)
4. `src/data_manager.py:392` sonundaki `df.dropna()` — `03`'ün `_suppress_dropna` yaması sadece **forecast** satırlarını koruyor; **NaN'lı eğitim satırlarını sessizce siliyor.**
5. Sonuç: 06-29→07-06 eğitim satırları uçuyor → `03`'ün NaN guard'ı **"OK (0 NaN)" diyor** (çünkü satırlar dolduruldu değil, silindi) → pipeline "başarılı" raporluyor.

### Neden ciddi
- Modelin gördüğü en yeni gün **06-28** — yani her gün koşsan bile model **9 gün bayat.**
- `RECENCY_HALFLIFE_DAYS=60` recency ağırlıklandırması **en değerli (en yeni) günleri hiç görmüyor** — özellikle yaz soğutma rampasında (config'in kendi gerekçesi) tam ters etki.
- **Tamamen sessiz.** Log "TAMAM" diyor, UI yeşil yanıyor. Kimse fark etmez.
- Demo backtest MAPE'leri (aşağıda ~%4-5) **as-of** ölçüldüğü için iyi görünüyor; canlı model bu bug yüzünden **backtest'ten daha kötü** tahmin üretebilir.

### Çözüm seçenekleri (öncelik sırası)
1. **En temiz:** `master.parquet`'i gerçekten raw'a indir (Tarih, Saat, target) — hava hep `weather_history`'den taze gelsin. Kod zaten bunu varsayıyor.
2. **Hızlı yama:** `03`'te merge'i hava kolonları için `weather_history` lehine çevir (master'ın `_actual` kolonlarını merge'den önce düş), VEYA `combine_first` sırasını history öncelikli yap.
3. **Minimum güvenlik ağı:** `03`'ün NaN guard'ına "eğitim satırı silindi mi / son actual günü feature_matrix'te var mı" kontrolü ekle — silent drop bir daha yeşil raporlanmasın.

---

## 2. 🔴 İKİNCİ KRİTİK — ADM'nin bugünkü canlı run'ı tamamlanmadı

`logs/2026-07-07_run.log` bugün **6 ayrı deneme** gösteriyor:

| Saat | Sonuç |
|------|-------|
| 13:39 | `03_FEATURES` HATA — 2312 NaN (AYDIN) |
| 13:41, 13:46, 13:51 (UI'dan) | `03` HATA — 14456 NaN (AYDIN) |
| 14:36 | `02_fetch_weather` HATA — `numpy.datetime64 has no .date()` (git'te fix var ama o an patladı) |
| 14:37 | `03` HATA — 14456 NaN (AYDIN) |
| 14:44 | `--skip-weather` ile → `03` TAMAM → `04_PREDICT BAŞLIYOR` … **ve log burada bitiyor** |

- `04` başladıktan sonra modeller **14:46'da yazıldı** ama `raw_predictions.parquet` hâlâ **13:06'dan** (o da canlı değil, 30-günlük backtest REGEN çıktısı — 06-28/06-29 tahmin ediyor).
- **Şu an çalışan Python process YOK** (`tasklist` = 0). Yani `04` bitmeden öldü/durduruldu.
- **Sonuç: Şu an ADM için 07-08/07-09'a ait TAZE bir canlı tahmin dosyası YOK.** `output/` içindeki her şey ya REGEN backtest ya da dünkü run.

Bu, #1'deki AYDIN sorununun operasyonel yüzü: hava çekimi kırılgan, `03` sürekli patlıyor, sonunda `--skip-weather` ile zorlandı ve `04` yarıda kaldı.

---

## 3. ADM — Ayrıntılı Durum

### 3.1 "4 model optimal mı?" → Hayır, efektif ~2.5 model
`config_live.py:112`:
```python
CALIBRATED_ENSEMBLE_WEIGHTS = {"XGB_Pred": 0.40, "LGBM_Pred": 0.10, "CHRONOS_Pred": 0.50}
```
- **CatBoost: ensemble ağırlığı 0** (eğitiliyor, loglanıyor, ama kullanılmıyor). Gerekçe config'te dürüstçe yazılı: CAT solo ~%6.9, param düzeltmesi (l2_leaf_reg vb.) kurtaramadı, kök neden bulunamadı.
- **LGBM: sadece %10** — "kullanıcı 4 modelin birlikte çalışmasını istediği için bedelsiz zorlanan pay". Yani tahmine gerçek katkısı marjinal.
- Ağırlığın asıl taşıyıcıları: **Chronos %50 + XGB %40.**
- Rolling Ridge (OOF öğrenen stacker) `ROLLING_RIDGE_MIN_SAMPLES=168` (7g temiz OOF) dolana kadar **devrede değil** — statik ağırlık köprüsü kullanılıyor. Demo haftası için bilinçli/konservatif bir karar.

> **Demo mesajı önerisi:** ADM'yi "4 model eğitiliyor, kalibre edilmiş ağırlıklı ensemble XGB+Chronos ağırlıklı, CAT şu an karantinada" diye dürüst anlat. "4 model eşit optimal çalışıyor" dersen bu doğru değil.

### 3.2 Chronos silent-fallback koruması — ✅ iyi
`04_predict_48h.py:537-546`: Chronos çökerse `CHRONOS_Pred = XGB kopyası` olur ve XGB çift sayılırdı (07-01/07-03'te gerçekten olmuş). Artık `chronos_ok=False` işaretleniyor, Rolling Ridge atlanıyor ve ağırlıklar renormalize ediliyor (`:426`). Bu geçmiş bir hatanın düzgün kapatılmış hali.

### 3.3 Leakage — ✅ temiz
- Tüm target-türevli feature'lar `target_clean.shift(k)`, **k≥24h** (`data_manager.py:258-356`). Sub-24h target lag yok.
- Hava feature'ları forecast (inference'ta biliniyor) → sızıntı değil.
- Çok-adımlı recursive lag güncellemesi (`04:_recompute_lags_for_t2`) T+1 tahminini T+2 lag'ine besliyor, `shift(48)` ile geçmişi doğru ayırıyor — leak yok, standart recursive yaklaşım.
- Holiday-aware lag cleaning + post-holiday recovery lag var (domain açısından olgun).

### 3.4 Doğruluk (as-of backtest, `output/model_analysis_report.csv`, 06-06→06-27)
- T+2 **Ensemble MAPE ort. ~%4-5**, iyi günlerde ~%1.6, kötü hafta içi günlerinde ~%8-10 (06-08/09, 06-23).
- Final (post-process sonrası) çoğu gün Ensemble'a yakın/biraz daha iyi.
- **Uyarı:** Bu sayılar as-of (her gün o güne kadar eğitilmiş) — **canlı modeldeki #1 bayatlık bug'ı bu backtest'te YOK.** Canlı doğruluk bundan kötü olabilir.

### 3.5 Post-process — çalışıyor ama demo-haftasına aşırı fit riski
`05_postprocess.py` + `config_live.py:118-138`: bias +10/+15 MWh, hafta sonu/Pazar için scale (0.20–0.60), LGBM Pazar weight boost. Hepsi son ~30 günün ME analizine elle kalibre. İşe yarıyor ama **birkaç günlük pencereye overfit** riski taşıyor; genel bir prensip değil, nokta-atışı düzeltme.

---

## 4. GDZ — Ayrıntılı Durum (🟢 asıl iyi haber)

### 4.1 Bugün tam koştu
`live/logs/2026-07-07_summary.json` → **status: ok**, 14:59→15:08, 6 adım da tamam:
- 01 ingest 07-06 (gediz CSV, doğru routing), 02 hava (07-07→07-09), 03 features (115 feature, 0 NaN), 04 predict (chronos_ok=true), 05, 06 → **`2026-07-08_gdz_forecast.xlsx` teslim edildi.**
- Teslim sağlığı: 24 saat, ort. 2557 MWh, aralık 1837–3137 (mantıklı günlük profil).

### 4.2 4 model — hepsi koştu, ama eşit ortalama (kalibre değil)
`04_predict_48h.py`: LGBM + CatBoost + XGB + **Chronos (zero-shot)** dördü de çalıştı. T2 tahmin ortalamaları birbirine çok yakın (sağlıklı uyum):
```
LGBM 2549.7 · CAT 2567.7 · XGB 2535.4 · CHRONOS 2577.9
```
- Ensemble = **basit ortalama** (`t2_df.mean(axis=1)`), config'teki `CALIBRATED_ENSEMBLE_WEIGHTS` (0.34/0.33/0.33) **kullanılmıyor — placeholder.**
- Bayram saatlerinde → sadece-GBDT routing (validated bulgu).
- Chronos = **zero-shot** (GDZ'nin kendi LoRA'sı var ama "izole pilotla doğrulanmadı" diye bilinçli kullanılmıyor).

### 4.3 Eğitim TAZE — ✅ (ADM'nin tersine)
GDZ eğitim verisi **07-06'ya kadar dolu** (07-01…07-06 hepsi 24 satır, 0 NaN). GDZ master raw, hava ayrı `GDZ_WEATHER.parquet`'ten geliyor → **ADM'deki bayat-master sorunu GDZ'de YOK.** Günlük güncelleme GDZ'de gerçekten çalışıyor.

### 4.4 Leakage — ✅ açıkça ve doğru ele alınmış
`src/gbdt_features.py`: feature matrisinde Lag1h–Lag12h var **ama T2 (yarın) modeli bunları KULLANMIYOR.** Ayrı bir `T2_SAFE_LOAD_COLS` seti sadece **≥48h lag'leri** (Lag168h/336h/504h, Mean_Last_3_Days_Lag48h, Trend_48_vs_168) alıyor. Docstring sızıntıyı açıkça tartışıyor. Ayrıca `test_gbdt_leakage.py` mevcut. Bu, olgun bir leakage disiplini.

### 4.5 GDZ'nin eksikleri (demo-blocker DEĞİL ama bilinmeli)
- **Bugün doğmuş bir port** (tüm `live/` dosyaları 07-07 tarihli) — tek bir başarılı koşu var, gün-tekrarı stres-test edilmedi.
- Ensemble ağırlıkları **kalibre/backtest edilmedi** (eşit ortalama placeholder).
- **OOF / scorecard / forecast_log feedback yok** (Faz 6 bekliyor) — ADM'deki izleme altyapısı henüz yok.
- **UI'da kapalı** → demoda GDZ'yi **CLI'dan** (`python run_daily.py`) göstermelisin.
- Post-process no-op (bias correction kapalı, delta=0).

---

## 5. Multi-model Gerçeği (özet)

| | ADM | GDZ |
|--|-----|-----|
| Eğitilen model | XGB, LGBM, CatBoost, Chronos | LGBM, CatBoost, XGB, Chronos (zero-shot) |
| Ensemble | Kalibre statik ağırlık (Chronos .50/XGB .40/LGBM .10, **CAT=0**) | **Basit eşit ortalama** (4 model) |
| Ağırlık kalibrasyonu | Var (as-of backtest + LOO) | **Yok (placeholder)** |
| Rolling Ridge | Hazır, 7g temiz OOF bekliyor | Hazır, Faz 6'da |
| Chronos | Fine-tuned LoRA yolu var | Zero-shot (LoRA kullanılmıyor) |

Her ikisinde de "4 model gerçekten optimize edilmiş şekilde birlikte" iddiası **abartılı**: ADM'de CAT dışarıda + LGBM token, GDZ'de ağırlık kalibre değil.

---

## 6. UI Sağlamlığı

- `ui/tab_tahmin_uret.py`: ADM için 6 adım `st.status` içinde **senkron** koşuyor; her adım try/except, hata → göster + `st.stop()` (temiz). Loglama/scorecard 06 sonrası, hatası teslimi düşürmüyor.
- **`04` yavaş** (Chronos CPU, `CHRONOS_FORCE_CPU=True`, ~dakikalar) ve senkron → UI birkaç dakika "donmuş" görünür. Demoda "takıldı mı?" paniği olmasın diye önceden söyle.
- **GDZ butonu UI'da DISABLED** ("🚧 Yakında"). Demoda GDZ'yi UI'dan gösteremezsin — CLI kullan.
- Bugünkü UI koşuları da aynı AYDIN NaN'a patlamış (log satır 43-46) — yani UI'dan "Başlat" dersen **şu anki master durumuyla ADM yine `03`/`04`'te sıkıntı çıkarabilir.**

---

## 7. MLOps / Güvenilirlik

**İyi olanlar:** run_context + config_hash + model arşivleme (90g retention), forecast_log/actuals_log (OneDrive dışı, %LOCALAPPDATA%, kilitlenme riskine karşı), günlük zip yedeği, DuckDB view'ları, scorecard + robust-z alarm, teslim öncesi sanity guard (band/flat/negatif — engellemez, flag'ler).

**Riskler:**
1. **Hava çekimi kırılgan (ADM #1 operasyonel risk).** AYDIN istasyonları için forecast API bazen boş dönüyor, `weather_history` NaN kalıyor, `fix_weather_history` Archive API backfill'ine bağımlı; Archive API bugünü kapsamıyor + gecikmeli. Bugün 6 denemede zor geçti. "Her sabah otomatik bam güm" için bu tek başına bir engel.
2. **master ↔ weather_history merge önceliği** (#1 bug'ın kaynağı) — mimari borç.
3. **Silent success:** eğitim satırı düşmesi yeşil raporlanıyor. Guard yalnızca NaN'a bakıyor, "en yeni actual eğitimde mi" ye bakmıyor.
4. GDZ izleme altyapısı (OOF/scorecard) henüz yok — GDZ kötü gün üretirse otomatik yakalanmaz.

---

## 8. Domain / Literatür Tutarlılığı

Genel olarak **sağlam** — modern kısa-dönem yük tahmini (STLF) pratiğiyle uyumlu:
- ✅ HDD/CDD (16°C/24°C eşikleri), Temp², ekstrem sıcaklık kareleri — bina termal yükü literatürüyle uyumlu.
- ✅ Takvim: bayram substitution, post-holiday recovery, köprü günleri, Ramazan/sahur, sömestr & yaz tatili, seçim, milli bayram — TR yük profili için kapsamlı.
- ✅ Hafta içi/hafta sonu model ayrımı + Pazar under-prediction'a özel weight (yük profili literatürüyle uyumlu; Cmt≠Paz≠hafta içi).
- ✅ PV/solar shaving proxy (GHI × açık-gökyüzü) — çatı-üstü GES penetrasyonunun net-yükü aşağı çekmesini modelliyor; dağıtım-seviyesi net yük tahmininde giderek standart.
- ✅ GBDT + foundation model (Chronos) hibriti + recursive multi-step — güncel yaklaşım.
- ✅ Recency-weighted training (concept drift'e karşı) — literatürde makul.
- ⚠️ Küçük not: T+2'de ağırlıklı GHI ve yük için tek `WEATHER_GHI_WEIGHTS` (il ağırlıkları) elle set; ADM için MUGLA .25/DENIZLI .40/AYDIN .35 — kurulu GES/tüketim payıyla doğrulanmış mı, teyit edilebilir.

---

## 9. Demo Öncesi Öncelikli Aksiyon Listesi

**🔴 Bloklayıcı (ADM):**
1. **master bayatlık bug'ını çöz** (§1): master'ı raw'a indir VEYA merge'i weather_history lehine çevir. Sonra `03`'ün eğitiminin **07-06'ya kadar dolu** olduğunu doğrula (feature_matrix son eğitim günü = 07-06 olmalı).
2. **ADM'yi baştan temiz koştur** ve `04→05→06`'nın tamamlanıp **07-08 teslim dosyası ürettiğini** gör. Şu an taze ADM tahmini yok.
3. **Hava çekimini stabilize et** (§7.1): AYDIN NaN senaryosunda `03`'ün sert durması yerine, en azından demo için `weather_history`'nin dolu olduğunu garantile.

**🟡 Demo kalitesi:**
4. GDZ'yi CLI'dan göstermeye hazırlan (UI'da kapalı). İstersen basit bir "GDZ da CLI'dan aynı 6 adımı koşuyor" demosu.
5. Anlatıyı dürüstleştir: ADM 4 model eğitir, ensemble XGB+Chronos ağırlıklı (CAT karantinada); GDZ 4 model eşit ortalama (kalibrasyon Faz-4'te).
6. `04`'ün yavaşlığını (Chronos CPU) demoda önceden belirt.

**🟢 Sonraki iterasyon (demo-blocker değil):**
7. Silent-drop'a karşı guard (en yeni actual eğitimde mi kontrolü).
8. GDZ ensemble ağırlık kalibrasyonu + OOF/scorecard portu.
9. Post-process bias'ının demo-haftasına overfit'ini daha geniş pencerede doğrula.

---

### Ek Not — Doğrulama izi
- ADM eğitim bitişi 06-28: `feature_matrix.parquet` doğrudan okundu.
- Merge önceliği: master AYDIN 192/192 NaN vs weather_history 0/192 — izole merge testiyle doğrulandı.
- ADM canlı run yarıda: `logs/2026-07-07_run.log` (04 BAŞLIYOR'da bitiyor) + çalışan Python process yok + `raw_predictions.parquet` 13:06 (backtest).
- GDZ başarı: `live/logs/2026-07-07_summary.json` status=ok + `2026-07-08_gdz_forecast.xlsx` + raw_predictions model spread.
- GDZ leakage: `src/gbdt_features.py` `T2_SAFE_LOAD_COLS` (≥48h).
