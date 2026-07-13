# STLF Monitoring Altyapısı — Kökten Düzeltme Planı (2026-07-10)

> **Tetikleyici:** 10 Temmuz'da İzleme & Analiz sekmesi bir günde 3 kez farklı şekilde bozuldu
> (ADM'de T+2 boşluğu, GDZ'de karışık ufuk etiketleri, ADM'de StreamlitAPIException crash). Her
> seferinde kaynağa inip düzelttik ama kullanıcı haklı bir tespit yaptı: **bunlar birbirinden
> bağımsız semptomlar değil, tek bir zayıf altyapının farklı yerlerden sızması.** Bu doküman o
> altyapıyı kökten düzeltme planı — Faz 0 (acil triyaj) tamamlandı, Faz 1-4 henüz UYGULANMADI,
> sadece tasarlandı.

---

## 0. Neden sürekli bozuluyor — kök nedenler

Semptom değil, kök neden listesi (hepsi 10 Temmuz oturumunda canlı olarak doğrulandı):

| # | Kök neden | Somut kanıt |
|---|---|---|
| A | **Tek doğruluk kaynağı yok, 4 tane var**, senkron değiller | 10 Temmuz'un tahmini `output/2026-07-10_ADM_forecast.xlsx`'te sapasağlamdı ama `forecast_log`'da yoktu — panel "kayıp" gösterdi |
| B | **forecast_log kayıplı, kendini onaramıyor** — artımlı yazım, idempotent projeksiyon değil | 8 Temmuz'daki run 48 satır yazdığını logladı (`summary.json`) ama diskte sadece 24'ü vardı |
| C | **Ufuk (horizon) etiketi 3 farklı kodlamayla var**: `"T+1"/"T+2"` (ADM), `"T1_bugun"/"T2_yarin"` (GDZ eski), string `"T+2"` ile issue+1 gün karışıklığı (`06_deliver.py` içindeki "yarın=T+2" yanlış etiketi) | GDZ'de 07-07/07-08 satırları `horizon_day='T1_bugun'` diye kayıtlıydı, `["T+1","T+2"]` filtresi bunları hiç görmüyordu |
| D | **Zaman damgası konvansiyonu tutarsız** (hour-ending vs hour-beginning), dosyadan dosyaya değişiyor | GDZ arşivlerinin çoğu ilk saati "01:00" ile başlıyordu (hour-ending), biri "00:00" ile (hour-beginning) — otomatik tespit etmek zorunda kaldık |
| E | **Backfill/regen scriptleri canlı dosyaların üstüne yazıp "sonra geri yüklüyordu"** | `asof_regen.py`'nin eski tasarımı — 6 Temmuz master.parquet korupsiyonu ve muhtemelen 8 Temmuz'daki forecast_log kaybı buradan |
| F | **İki tenant, iki kod kopyası** (`forecast_logger.py`, `scorecard.py`, `run_daily.py`) | GDZ'nin bugüne dek hiç `daily_scorecard`'ı yoktu — `run_daily.py`'ye scorecard adımı hiç eklenmemiş, kimse fark etmemiş |
| G | **Monitoring adımları sessizce çöküyor** (`try/except` + sadece log warning) | `run_daily.py`: forecast_log/scorecard hataları sadece `log.warning()`, hiçbir yerde görünmüyor |
| H | **UI, veri katmanının işini yapıyor** — gap-detection, reindex, tarih widget'ı için session_state juggling `tab_izleme.py` içinde | Bugünkü ADM crash'i (`06-28 < min_value 06-29`) tam bunun sonucu: güvenilmez veri katmanını sunum katmanında telafi etmeye çalışmak kırılgan |

---

## 1. Hedef mimari — ilkeler

1. **Tek, değişmez, yeniden kurulabilir kaynak.** 48h arşiv parquet'i (`output/archive/*_full48h.parquet`) her run'ın **atomik + doğrulanmış** (48 satır, doğru şema, doğru saat gridi) tek doğruluk kaynağı olsun. `forecast_log` = tüm arşivlerin **saf, deterministik projeksiyonu**. Rebuild = tara + dedup + normalize + yaz. Arşiv varsa boşluk otomatik dolar; elle backfill script'i diye bir kategori kalmaz.

2. **Semantiği bir kez normalize et.** String ufuk zoo'sunu öldür. Her forecast satırında `issue_date` + `target_date` sakla, `horizon_days = (target_date - issue_date).days` **integer** olarak türet. UI/scorecard filtresi `horizon_days == 2` gibi çalışsın. Yanlış etiketleme yapısal olarak imkânsız hale gelir. `target_ts` her zaman hour-beginning, tz-naive, tek bir yazma fonksiyonundan geçer (elle saat hesaplayan ikinci bir kod yolu olmaz).

3. **İki tenant, tek kod.** `forecast_logger`/`scorecard`/şema tek bir paylaşımlı `monitoring/` paketi olsun, `TenantConfig` dataclass'ıyla parametrize (edas_id, kolon adları, log yolları, teslim lead'i). Fix bir kere yazılır, ikisine de otomatik gider.

4. **Kendini iyileştiren günlük reconcile.** Her run sonunda: arşivden `forecast_log`'u tam kur → actuals'ı tam kur → view+scorecard kur → **tamlık kontrolü** (beklenen aralıkta her gün ilgili satır sayısı tam mı, actual gelmesi gerekince gelmiş mi). Boşluk + arşiv mevcut → otomatik dolar. Boşluk + arşiv yok → **gerçek kayıp**, sessiz warning değil, yüksek sesle flag. İdempotent olduğu için bir günkü geçici hata ertesi gün kendiliğinden düzelir — elle müdahale istisna olur, rutin olmaz.

5. **UI aptal okuyucu olsun.** Garantili-tam bir view'dan okur; gap-detection/reindex/tarih-clamp mantığı olmaz çünkü altındaki veri zaten tam. Sadece "veri var mı yok mu" gösterir, "neden yok"u tahmin etmeye çalışmaz.

---

## 2. Faz 0 — Acil triyaj (TAMAMLANDI, 2026-07-10)

- [x] `ui/tab_izleme.py`: `izleme_start`/`izleme_end` session_state key'leri tenant'a göre ayrıldı + widget kurulmadan önce defansif `[min,max]` clamp eklendi → ADM'i çökerten `StreamlitAPIException` bitti.
- [x] Baslangıç/Bitiş varsayılanları artık seçili ufkun **kendi** min/max'ına göre (`_forecast_edge_date_for_horizon`) — genel fc_min/fc_max karışımı kullanılmıyor → sahte "bu gün için veri yok" uyarıları bitti.
- [x] ADM 10 Temmuz T+2 boşluğu, donmuş modellerle as-of regen ile dolduruldu.
- [x] GDZ 06-28→07-09 arası forecast_log, arşivden temiz saat-gridli backfill ile dolduruldu.
- [x] GDZ'ye `scorecard.py` portlandı + `run_daily.py`'ye günlük scorecard adımı eklendi (önceden hiç yoktu).
- [x] `asof_regen.py` tamamen sandbox izolasyonuna alındı — canlı dosyalara artık hiç dokunmuyor.
- [x] `ui/common.py` ↔ `src/common.py` isim çakışması (rastgele ImportError kaynağı) → `ui/dashboard_common.py` diye yeniden adlandırıldı.

**Doğrulanmış durum:** ADM ve GDZ, T+2 ufkunda 2026-07-01→07-11 arası sıfır boşluk, crash yok.

**Bu faz neyi düzeltmedi:** kök nedenler A-H hâlâ duruyor. Bugünkü gibi bir olay (arşiv/log senkron kayması, yeni bir string-etiket uyuşmazlığı, sessiz bir adım hatası) her an tekrar olabilir — sadece o anki semptomu yamaladık.

---

## 3. Faz 1 — Kanonik şema + idempotent projeksiyon (TAMAMLANDI, 2026-07-10)

**Amaç:** "forecast_log'da satır yok" diye bir durum, arşiv varken bir daha asla olmasın.

**Gerçekleşen (planla küçük bir fark var — daha düşük riskli, ek katman şeklinde
uygulandı):** `FORECAST_LOG_SCHEMA`'ya `issue_date`/`horizon_days`(int8)/`source`
eklendi — ama eski `horizon_day` string alanı SİLİNMEDİ, geriye-uyumluluk için
kaldı (UI hâlâ onu okuyor). "Tam yeniden projeksiyon" yerine daha güvenli bir
**gap-fill** yaklaşımı seçildi: `heal_forecast_log_gaps()` her run sonunda
arşivi tarar, SADECE eksik/kısmi (24'ten az satır) hücreleri doldurur, zaten
dolu olana dokunmaz. Yol boyunca bulunup düzeltilen 3 gerçek bug: (1)
`hive_partitioning=1` + `union_by_name` birlikte DuckDB internal error
veriyordu, (2) arşiv dosya adından issue_date çıkarırken ilk grup yerine
ikinci grup-1 kullanılması gerekiyordu (toplu regen dosyalarında ilk grup
issue_date değil), (3) `existing_counts` kontrolü `horizon_days IS NOT NULL`
ile yapılınca Faz 1 öncesi 'live' satırlar hep 'eksik' sanılıp gereksiz yere
tekrar tekrar healed ediliyordu — `horizon_day` string ile düzeltildi.
Doğrulandı: ADM+GDZ, T+2'de 06-29→07-12 arası sıfır boşluk, actuals join
%100.

### 3.1 Şema (`monitoring/schema.py`, yeni)

```python
FORECAST_LOG_SCHEMA_V2 = pa.schema([
    ("edas_id", pa.string()),
    ("issue_date", pa.date32()),        # YENİ: horizon_days'in türetildiği çapa
    ("target_ts", pa.timestamp("us")),  # HER ZAMAN hour-beginning, tz-naive
    ("target_date", pa.string()),
    ("horizon_days", pa.int8()),        # YENİ: (target_date - issue_date).days — string YOK
    ("lead_time_h", pa.int32()),
    ("run_id", pa.string()),
    ("source", pa.string()),            # "live" | "backfill_archive" | "archregen" — ne zaman neyle dolduruldugu izlenebilir
    # ... (mevcut y_pred_*, meta_*, override_* alanlari birebir korunur)
])
```

- `horizon_days` mevcut `"T+1"/"T+2"` string'inin yerini alır. UI/scorecard `horizon_days.isin([1,2])` filtreler — GDZ'nin "T1=bugün/T2=yarın" ile ADM'nin "T+1=yarın/T+2=öbürgün" farkı artık **sayı olarak açık**, karışma ihtimali yok. `HEADLINE_HORIZON` gibi tenant-bazlı yorum gerektiren sabitler tamamen kalkar; her tenant `TenantConfig.delivery_horizon_days` alanıyla kendi teslim gününü tanımlar (bkz. §4).
- `source` alanı, bir satırın canlı run'dan mı yoksa backfill/archregen'den mi geldiğini UI'de görünür kılar (bugün elle takip ettiğimiz "archregen" run_id konvansiyonu artık ilk sınıf alan olur).

### 3.2 `rebuild_forecast_log()` — artımlı değil, TAM projeksiyon

Şu an: her run kendi 48 satırını yazar, eksik/bozuk run sessizce eksik kalır.
Hedef: `output/archive/*_full48h.parquet` içindeki **her** dosya taranır, şema normalize edilir (saat gridi, horizon_days), `(edas_id, target_ts, horizon_days)` bazında dedup edilir (en güncel `issue_date` kazanır, `source='archregen'` en düşük öncelik), ve **tüm `forecast_log` bu taramadan yeniden üretilir** — mevcut parquet dosyaları silinip yerine konur, artımlı append yok.

- Saat gridi normalizasyonu **tek yerde**: `_normalize_to_hour_beginning(df)` — dosyanın ilk satırının saatine bakıp (0 ise dokunma, 1 ise -1h kaydır, başka bir şeyse hata fırlat, sessizce geçme). Bugün elle yaptığımız auto-detect kalıcı, tek fonksiyon olur.
- Bu fonksiyon **idempotent**: hangi gün çalıştırılırsa çalıştırılsın, aynı arşiv seti aynı `forecast_log`'u üretir. Bir günün verisi "kayıp" olamaz, sadece "arşivi de yoksa gerçekten kayıp" olabilir (bkz. Faz 3).

### 3.3 Migration (gerçekleşen)

- Migration YOK — eski dosyalara dokunulmadı. `union_by_name=1` sayesinde eski
  (Faz 1 öncesi) ve yeni şemalı dosyalar aynı view'da sorunsuz bir arada
  okunuyor, eksik kolonlar NULL olarak gelir.
- `daily_scorecard`/`actuals_log` şeması değişmedi (`horizon_day` string join
  hâlâ kullanılıyor — `horizon_days` int şu an sadece heal'in kendi iç
  mantığında, dedup önceliğinde ve gelecekteki bir UI geçişi için hazır
  duruyor, henüz scorecard/UI tarafı ona geçmedi).

**Çıktı (planlanandan farklı):** `ui/tab_izleme.py`'deki `_missing_forecast_dates`/
`_reindex_hourly_gaps`/`_forecast_edge_date_for_horizon` **silinmedi** —
kasıtlı. `heal_forecast_log_gaps()` sadece ARŞİVİ OLAN boşlukları otomatik
doldurur; arşivi de kaybolmuş gerçek bir veri kaybı (ör. ADM'nin orijinal
10 Temmuz T+2'si) için hâlâ görünür bir uyarı gerekiyor — UI'nin bu telafi
kodu artık "her gün tetiklenen bir bug'ın belirtisi" değil, "gerçekten
kurtarılamayan bir boşluğun dürüst göstergesi" rolünde kalıyor.

---

## 4. Faz 2 — Ortak kütüphane, iki tenant birleşir (TAMAMLANDI, 2026-07-10)

**Amaç:** Bir fix'i iki kere yazıp birinin unutulması (GDZ'nin scorecard'ı gibi) yapısal olarak imkânsız olsun.

**Gerçekleşen (planla küçük bir fark var — daha düşük riskli):**

- `monitoring/` paketi `adm live/` altında (git-tracked orada) — GDZ tarafı
  `config_live_gdz.py`'nin başında `sys.path.insert(0, ".../adm live")` ile
  buraya erişir. `schema.py`, `tenant_config.py`, `forecast_logger.py`,
  `scorecard.py`. `reconcile.py` henüz yok (Faz 3'e bırakıldı).
- `TenantConfig` dataclass — plandaki alanlara ek olarak `horizon_day_label_offset`
  (ADM=0, GDZ=1 — GDZ'nin "T1=bugün" kaymasını tek yerde kapatır) ve
  `logger_name` eklendi.
- **`write_forecast_log`/`update_actuals_log`/`update_actuals_log_weather`
  BİLEREK paylaşıma taşınmadı** — GDZ'nin T1/T2 ayrı-kolon (coalesce
  gerektiren) şeması ile ADM'nin doğrudan-kolon şeması arasında gerçek
  yapısal fark var, zorla tek fonksiyona sıkıştırmak yapay soyutlama
  olurdu. Bunlar hâlâ her tenant'ın kendi `src/forecast_logger.py`'sinde
  tenant-spesifik — ama `rebuild_duckdb_views`, `heal_forecast_log_gaps`,
  `compute_calendar_fields`, upsert yardımcıları, `backup_logs_zip` ve
  **tüm `scorecard.py`** (2026-07-10'a kadar GDZ'de hiç yoktu) artık tek
  kopya.
- Eski `adm live/src/forecast_logger.py`/`scorecard.py` ve GDZ karşılıkları
  **silinmedi**, ince shim'e çevrildi: aynı fonksiyon imzalarını koruyup
  içeriden `monitoring/`'e delege ediyorlar — `run_daily.py`,
  `ui/tab_tahmin_uret.py`, `asof_regen.py`, `backfill_logs.py`,
  `perfect_prog_rerun.py` gibi HİÇBİR çağıran dosya değişmedi.

**Doğrulandı:** her iki tenant taze subprocess'te (`python run_daily.py --help`)
ve doğrudan fonksiyon çağrılarıyla (`heal_forecast_log_gaps`,
`rebuild_duckdb_views`, `build_daily_scorecard`, `check_alerts`) hatasız
çalışıyor; T+2 06-29→07-12 (ADM) / 06-29→07-11 (GDZ) sıfır boşluk korundu.
Bu süreçte paylaşıma taşınırken bulunan bir bug (existing_counts'un
`horizon_days IS NOT NULL` ile yanlış sayması) TEK yerde düzeltilip iki
tenant'ı da otomatik düzeltti — paketin var olma sebebinin canlı kanıtı.

---

## 5. Faz 3 — Günlük reconcile + tamlık kontrolü (TAMAMLANDI, 2026-07-11)

**Amaç:** Bir monitoring adımı sessizce çökemez; bir gün eksik kalırsa biri (log değil, görünür bir yer) bunu söyler.

**Gerçekleşen (planla küçük bir fark var — Faz 1'in gerçek tasarımına uyarlandı):**
Faz 1 "tam projeksiyon" yerine gap-fill (`heal_forecast_log_gaps`) olarak
uygulandığı için, `rebuild_forecast_log()` diye bir fonksiyon hiç yok —
`monitoring/reconcile.py`'nin 1. adımı onun yerine mevcut `heal_forecast_log_gaps()`'i
çağırıyor. Actuals tarafı (madde 2) zaten idempotent (`upsert_by_date`) olduğu
için dokunulmadı — D+1/D+6 gecikmesi normal, "kayıp" değil, bu yüzden actuals
tamlık kontrolünün KAPSAMI DIŞINDA bırakıldı (yanlış pozitif üretirdi).

- `monitoring/reconcile.py` → `reconcile(config)`: tek çağrıda
  1. `heal_forecast_log_gaps(config)` (Faz 1) — arşivi olan boşlukları doldurur.
  2. **Tamlık kontrolü:** son 14 gün x (`T+1`,`T+2`) için `forecast_log_v`'de
     24 satır var mı bak. Eksik + arşiv de yok → gerçek kayıp.
  3. Gerçek kayıp varsa `logs/gaps/<run_date>.json`'a yazar (ADM: proje-lokal
     `logs/gaps/`; GDZ: `LOCALAPPDATA/gdz_live_logs/gaps/` — `alerts_dir` ile
     aynı mevcut asimetri, `TenantConfig.gaps_dir` property'si `log_root`'tan türetiliyor).
- **Bulunan ve düzeltilen bug (canlı veride yakalandı):** ilk saf implementasyon
  ADM'de 3 sahte "gerçek kayıp" üretti (`2026-06-28` T+1/T+2, `2026-06-29` T+2)
  — bunlar hata değil, monitoring'in gerçekten başladığı günün (`06-29`)
  YAPISAL OLARAK imkânsız ufukları (T+2 için issue_date=06-27 gerekir, sistem
  henüz yoktu). `_system_start_issue_date()` eklendi: en eski `target_date`'te
  mevcut olan EN KISA ufuktan (`horizon_day` string'den türetilir, `issue_date`
  kolonu Faz 1 öncesi çoğu satırda NULL) sistemin gerçek başlangıç `issue_date`'i
  çıkarılır; ondan önceki hücreler gerçek kayıp sayılmaz. Enjekte edilmiş
  sentetik bir kayıp (dolu bir günün 24 satırını silme) ile pozitif tespit,
  canlı veriyle negatif tespit (0 sahte pozitif) doğrulandı.
- `run_daily.py` (ADM+GDZ) ve `ui/tab_tahmin_uret.py` (ADM'nin in-process UI
  tetikleyicisi — GDZ zaten subprocess olarak kendi `run_daily.py`'sini
  çağırdığı için otomatik kapsanıyor) artık `heal_forecast_log_gaps()` yerine
  `reconcile()` çağırıyor; gerçek kayıp varsa `log.warning`/`st.warning` ile
  görünür oluyor. Tenant shim'lerine (`src/forecast_logger.py`, ikisinde de)
  ince `reconcile()` wrapper'ı eklendi — imza `heal_forecast_log_gaps()` ile
  aynı desende, mevcut çağıranlar bozulmadı.

**Doğrulandı:** ADM+GDZ, taze subprocess'te (`python run_daily.py --help`) ve
doğrudan `reconcile()` çağrısıyla hatasız; her ikisi de `status: ok, n_gaps: 0`
(mevcut veri gerçekten tam). `ui/tab_tahmin_uret.py` `py_compile` ile
doğrulandı (Streamlit UI görsel olarak bu oturumda test edilemedi — tarayıcı
önizleme aracı koptu).

**Bu faz neyi düzeltmedi:** gerçek kayıp bulunduğunda otomatik regen tetiklenmiyor
(elle `asof_regen.py` gerekiyor) ve Teams/mail bildirimi yok — ikisi de Faz 4/5'in işi.

---

## 6. Faz 4 — Gerçek-kayıp otomatik regen + tam izolasyon

**Amaç:** Bugün elle yaptığımız "donmuş modelle as-of regen" adımı otomatik olsun; hiçbir script canlı dosyaya bir daha dokunmasın.

- Faz 3'ün tamlık kontrolü gerçek bir kayıp bulursa (arşiv de yok), `asof_regen.py`'nin (zaten sandbox'lı) mantığı otomatik tetiklenir, sonucu `source='archregen'` etiketiyle `forecast_log`'a yazılır. **HENÜZ YAPILMADI.**
- `backtest_walkforward_gdz.py`, `perfect_prog_rerun.py` gibi kalan script'ler `asof_regen.py`'deki `_enter_sandbox()`/`_exit_sandbox()` mekanizmasını yeniden kullanacak şekilde güncellenir. **HENÜZ YAPILMADI** — GDZ'nin kendi `asof_regen.py` eşdeğeri hiç yok; `backtest_walkforward_gdz.py` hâlâ canlı `GDZ_MASTER.parquet`'i yerinde kesip restore eden eski (riskli) desende. `perfect_prog_rerun.py` incelendi: hiçbir canlı dosyaya yazmıyor (salt-okunur, arşivlenmiş donmuş modelleri yükler) — sandbox'a ihtiyacı yok, plandaki madde gereksizmiş.

### 6.1 Kısmi ilerleme (2026-07-11): `backtest_walkforward.py` düzeltildi

Kapsam kullanıcı onayıyla daraltıldı — sadece ADM'nin `backtest_walkforward.py`'si (GDZ portu ve otomatik-tetikleme ayrı, henüz onaylanmamış işler).

**Bulunan durum:** `asof_regen.py`'nin 07-10 sandbox yeniden yazımından beri `backtest_walkforward.py` artık var olmayan `AR._backup()`/`AR._restore()`'u çağırıyordu — script kırıktı (AttributeError). Ama çağrıları silmek TEK BAŞINA yeterli değildi: `write_forecast_log(ctx)` modül-seviyesinde sabitlenmiş GERÇEK `POSTPROC_PATH`/`RAW_PREDICTIONS_META_PATH`'i okuyor; `regen_one()` artık tamamen sandbox'ta çalıştığından bu satır sessizce bir önceki GERÇEK run'ın bayat tahmin/ağırlıklarını `forecast_log`'a yazardı (kök neden A'nın yeni bir versiyonu).

**Düzeltme:**
1. `asof_regen.py:regen_one()` artık `raw_predictions_meta.json`'ı da (`postprocessed_predictions.parquet` gibi) sandbox'tan `output/<tarih>_meta_REGEN.json`'a kopyalıyor; dönen dict'e `models_path`/`meta_path` eklendi.
2. `src/forecast_logger.py:write_forecast_log()` opsiyonel `postproc_path`/`meta_path`/`source` parametreleri aldı (varsayılan `None`/`"live"` → mevcut çağıranlar `run_daily.py`/`ui/tab_tahmin_uret.py` DEĞİŞMEDİ, davranışları birebir aynı).
3. `backtest_walkforward.py`: `AR._backup()`/`AR._restore()` kaldırıldı (sandbox zaten canlıya dokunmuyor); `write_forecast_log()` artık `regen_one()`'ın döndürdüğü REGEN path'leriyle + `source="backfill"` çağrılıyor.
4. `monitoring/schema.py`: `source` alanı yorum satırına `"backfill"` eklendi.

**Doğrulandı:** `python asof_regen.py 2026-07-02` gerçek pipeline'ı (03→06) uçtan uca çalıştırdı, `output/2026-07-02_meta_REGEN.json` doğru ensemble ağırlıklarıyla (`meta_w_xgb=0.4` vb.) üretildi. (Aynı komutta script sonunda sandbox klasörünü silerken OneDrive reparse-point kilidiyle karşılaştı — pipeline'la/bu düzeltmeyle ilgisiz, ortama özgü bilinen bir OneDrive senkron sorunu, manuel temizlendi.)

**Kapsam dışı bırakılan (henüz yapılmadı):** GDZ sandbox portu, otomatik `archregen` tetikleme.

---

## 7. Öncelik / sıralama önerisi

```
Faz 0 (bitti)
   │
Faz 1 — kanonik şema + idempotent projeksiyon   (asıl "adam etme", en yüksek etki)
   │
Faz 2 — ortak kütüphane, iki tenant birleşir     (Faz 1 ile birlikte/hemen sonra, aynı oturumda)
   │
Faz 3 — günlük reconcile + tamlık kontrolü       (Faz 1+2 üzerine ince, düşük risk)
   │
Faz 4 — otomatik regen + tam izolasyon           (Faz 3 üzerine, düşük-orta risk)
   │
Faz 5 — zamanlanmış tetikleme + Teams/mail       (ayrı iş, bkz. OTOMASYON_YOL_HARITASI_2026-07-10.md — Faz 1-4 sağlamlaşınca anlamlı)
```

Faz 1+2 birlikte, sürekli bozulmaların büyük kısmını kökten bitirir (tek kaynak + otomatik dolum + tek kod = kök neden A, B, C, D, F kapanır). Faz 3-4 bunun üstüne görünürlük ve otomasyon ekler (kök neden E, G, H kapanır).
