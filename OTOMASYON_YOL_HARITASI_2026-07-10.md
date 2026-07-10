# STLF Tam Otomasyon Yol Haritası — Taslak (2026-07-10)

> **Durum:** Sadece tasarım/plan dokümanı. Hiçbir kod değişikliği içermez, uygulama başlamadı.
> Kullanıcı talebi: UI'ı "tek giriş noktası" haline getirmek şimdilik tamamlandı (bkz. `ui/tab_tahmin_uret.py`
> — 9 adımın tamamı artık UI'dan tetikleniyor). Bu doküman, konuşmada tarif edilen **nihai otomasyon
> vizyonunu** ve oraya varmak için fazlı bir planı yakalıyor. Faz'lara başlamadan önce kullanıcı onayı gerekir.

---

## 1. Nihai Hedef (kullanıcının tarif ettiği akış)

```
[FTP Server]                          (yeni gün verisi her sabah düşer)
     │  otomatik bağlan + indir
     ▼
[01_ingest]  →  master.parquet güncellenir (otomatik, manuel kopyala-yapıştır yok)
     │
     ▼
[02_weather + 03_features]  →  otomatik feature matrisi
     │
     ▼
[04_predict → 05_postprocess → 06_deliver]  →  tahmin üretilir
     │
     ▼
[07_report_excel + 08_diagnostic_html]  →  rapor + diagnostic hazırlanır
     │
     ▼
[Teams'e otomatik gönderim]  →  diagnostic dashboard'un özeti/linki Teams'e düşer
     │
     ▼
[Kullanıcı Teams'te/UI'da inceler → ONAY butonuna basar]
     │
     ▼
[09_email_report tetiklenir]  →  müşteriye (emre.hangul, cagatay.bayrak + müşteri kontağı) gönderilir
```

**Kritik tasarım kararı (kullanıcı ima etti, netleştirilmeli):** Email adımı **insan onayı olmadan
otomatik ateşlenmeyecek** — mevcut `09_email_report.py` şu an "üretilirse gönder" mantığında, bunun
"üretilir → bildirim gönder → onay bekle → sonra gönder" mantığına dönmesi gerekiyor. Bu, pipeline'a
yeni bir **ara durum** (`awaiting_approval`) ekler.

---

## 2. Şu Anki Durum (2026-07-10 itibarıyla, bu konuşmadan doğrulandı)

| Bileşen | Durum |
|---|---|
| Veri girişi | **Manuel** — kullanıcı her sabah OneDrive'daki CSV'yi UI'dan "veri yükleme" sekmesiyle ingest ediyor |
| Tahmin üretimi | UI'dan tek tuşla (`▶ Tahmini Başlat`) — **bugün itibarıyla 9 adımın tamamı** UI'dan çalışıyor |
| Analiz/reshape | Kullanıcı elle: geçmiş veriyle mukayese, sıcaklık/feature analizi, ufak reshape/deneme — **tamamen manuel, kod değişikliği gerektirmiyor ama zaman alıyor** |
| Teams bildirimi | **Yok** |
| Onay mekanizması | **Yok** — email adımı (09) SMTP ayarlıysa otomatik gönderiyor, onay adımı hiç yok |
| FTP entegrasyonu | **Yok** — veri OneDrive'a müşteri tarafından/manuel konuyor |

---

## 3. Fazlı Plan

### Faz A — FTP Otomatik Veri Çekme
**Amaç:** `01_ingest_actual.py`'nin bugün okuduğu OneDrive DD.MM klasör deseni yerine (ya da ona ek
olarak) bir FTP/SFTP sunucudan otomatik indirme.

- **Açık sorular (kullanıcıdan netleştirilmeli):**
  - FTP sunucu bilgisi kimden geliyor — müşteri mi (Aydem/Gediz) sağlıyor, yoksa MRC'nin kendi
    ara-katmanı mı olacak?
  - Kimlik bilgileri nasıl saklanacak? (öneri: ortam değişkeni, `STLF_SMTP_*` deseninin aynısı —
    `STLF_FTP_HOST/USER/PASS`, asla config dosyasına hardcode edilmemeli)
  - Dosya formatı/isimlendirme deseni FTP'de de OneDrive'daki `DD.MM` klasör deseniyle aynı mı?
  - İndirilen dosya OneDrive'daki mevcut klasör yapısına mı düşecek (mevcut `data_scanner.py` mantığı
    değişmeden çalışır), yoksa yeni bir path mi olacak?
- **Teknik yaklaşım:** `pipeline/01_ingest_actual.py`'den ÖNCE çalışan yeni bir adım
  (`00_fetch_ftp.py`) — `ftplib`/`paramiko` (SFTP ise) ile bağlanıp indirir, sonra mevcut
  `01_ingest_actual.py` değişmeden aynı OneDrive yoluna bakmaya devam eder. Bu, mevcut ingest
  mantığını bozmadan en az riskli entegrasyon noktası.
- **Risk:** FTP sunucusu geç/eksik veri koyarsa (müşteri gecikmesi) pipeline'ın ne zaman tetikleneceği
  belirsizleşir → Faz B'deki zamanlamayla birlikte "veri hazır mı" kontrolü şart.

### Faz B — Zamanlanmış Otomatik Tetikleme
**Amaç:** Sabah kullanıcı UI'dan tıklamadan, pipeline kendiliğinden çalışsın.

- **Teknik seçenekler:**
  1. **Windows Task Scheduler** ile `run_daily.py`'yi her sabah belirli saatte çalıştırmak (en basit,
     mevcut CLI'yi hiç değiştirmeden kullanır — zaten 9 adımın hepsini yapıyor).
  2. Bu ortamda mevcut olan `scheduled-tasks` / `schedule` mekanizması ile bir cron görevi kurmak
     (Claude Code tarafında zaten `mcp__scheduled-tasks__*` araçları mevcut — ama bunlar bu konuşma
     ortamına özel, üretim sunucusunda kullanıcının kendi görev zamanlayıcısı gerekir).
  - **Önerilen:** Windows Task Scheduler + `run_daily.py --target ...` — UI'daki her şeyi
    tekrar yazmaya gerek yok, CLI zaten UI ile fonksiyonel olarak eşit (bu konuşmada UI'yı CLI'ye
    eşitledik).
- **Bağımlılık:** Faz A bitmeden bu fazın anlamı sınırlı (veri elle konuyorsa otomatik tetikleme
  boşa döner ya da eski veriyle çalışır).
- **Gerekli guard:** Veri "hazır" değilse (Faz A henüz indirmediyse / master.parquet güncellenmediyse)
  pipeline sessizce eski veriyle çalışıp yanlış tahmin üretmemeli — "freshness guard" zaten
  `03_build_features.py`'de kısmen var ([[adm-bayat-master-egitim-bug-2026-07-07]] hafıza kaydına
  bakınız), ama tetikleme zamanlamasına da bir "veri bugünün mü" ön-kontrolü eklenmeli.

### Faz C — Analiz/Reshape Otomasyonu (kullanıcının "ufak değişiklikler yapıyorum" dediği kısım)
**Amaç:** Kullanıcının şu an elle yaptığı "geçmiş veriyle mukayese + sıcaklık analizi + reshape"
işinin ne kadarı zaten otomatik (bkz. önceki mesajdaki `08_diagnostic_html.py` — D+2 Karşılaştırma,
Sensitivity, Cross Check sekmeleri bunu byük ölçüde YAPIYOR).

- **Netleştirilmesi gereken:** Kullanıcı diagnostic HTML'in ürettiği analizin ötesinde spesifik
  olarak ne yapıyor? ("ufak değişiklikler, reshape falan" — bu instruction'da net değil.) Eğer bu
  gerçekten modelin/feature'ların manuel ayarlanması ise (örn. bir ensemble ağırlığını elle
  değiştirmek), bu otomasyona alınamaz — insan kararı olarak kalmalı, sadece **daha hızlı
  görünür kılınabilir** (diagnostic HTML zaten bunu yapıyor).
- **Aksiyon:** Bu fazın kapsamı, kullanıcıyla oturup "diagnostic HTML'de hangi bilgi eksik, hangi
  kararı hâlâ elle veriyorsun" sorusunu netleştirdikten sonra netleşmeli. Şimdilik varsayım:
  diagnostic HTML'in kapsamı yeterli, sadece görünürlüğü (Faz D) eksik.

### Faz D — Teams Bildirimi
**Amaç:** `08_diagnostic_html.py` çıktısı (veya bir özeti) her sabah otomatik olarak Teams'e düşsün.

- **Teknik seçenekler (basitten karmaşığa):**
  1. **Teams Incoming Webhook** (Teams kanalına "connector" eklenir, `requests.post()` ile mesaj
     atılır) — en hızlı kurulum, ama Microsoft 365 Aralık 2024 itibarıyla klasik Office 365
     Connector'ları kademeli olarak kapatıyor; **Workflows (Power Automate) tabanlı webhook**'a
     geçilmesi gerekebilir — bu netleştirilmeli (tenant'ın hangi seçeneği desteklediği IT'ye sorulmalı).
  2. **Power Automate flow**: HTTP request tetikleyicisi ile bir Adaptive Card postalar, kart
     içinde "Diagnostic'i Aç" linki + (Faz E ile) "Onayla" / "Reddet" butonları olabilir.
  3. **Teams Bot (Azure Bot Service)**: Tam kontrol ama kurulum/onay süreci en ağır seçenek
     (Azure AD app registration, bot channel registration) — muhtemelen bu iş için gereğinden fazla.
- **Önerilen başlangıç:** Seçenek 2 (Power Automate + Adaptive Card) — hem bildirim hem onay
  butonunu (Faz E) tek altyapıda çözer, kod tarafında sadece bir HTTP POST eklemek yeterli.
- **İçerik:** HTML'in tamamını Teams kartına gömmek pratik değil — özet kart (bugünün ortalama/pik
  tahmini, son 7 gün MAPE, P95 uyarısı varsa flag) + "Detaylı Dashboard'u Aç" linki (OneDrive'daki
  HTML dosyasına ya da yeni bir basit web sunucusuna link).
  - **Açık soru:** HTML dosyası şu an sadece yerel `output/` klasöründe — Teams'ten tıklanabilir
    olması için ya OneDrive-senkronize bir yolda olmalı (zaten öyle olabilir, kontrol edilmeli) ya
    da küçük bir statik dosya sunucusu/paylaşım linki gerekir.

### Faz E — Onay Mekanizması + Otomatik Müşteri Teslimi
**Amaç:** Kullanıcı diagnostic'i inceleyip onayladıktan SONRA email müşteriye gitsin.

- **Teknik yaklaşım (Power Automate ile devam edilirse):**
  1. `06_deliver` sonrası pipeline **email'i otomatik ATMAZ**, bunun yerine bir "onay bekliyor"
     durumuna geçer (örn. `output/<tarih>_PENDING_APPROVAL` marker dosyası ya da `summary.json`'da
     `awaiting_approval: true`).
  2. Teams kartındaki "Onayla" butonu bir Power Automate action'ı tetikler → bu action, UI'ın
     (ya da küçük bir FastAPI/Flask endpoint'inin) bir "approve" endpoint'ine HTTP çağrısı yapar.
  3. Approve endpoint'i `09_email_report.py`'yi tetikler (mevcut kod zaten hazır, sadece tetikleme
     noktası değişiyor: "pipeline sonu" yerine "onay callback'i").
  4. **Alternatif (Teams entegrasyonu karmaşıksa, daha basit ilk versiyon):** Onay adımını Teams'te
     değil, **UI'da** yapmak — diagnostic üretildikten sonra UI'da "Müşteriye Gönder" butonu
     görünür, Teams sadece "diagnostic hazır, UI'ya bak" diye bildirim atar (webhook, karmaşık
     Adaptive Card gerekmez). Bu, Faz D'yi Seçenek 1 (basit webhook) ile sınırlı tutup Faz E'yi
     tamamen UI'da çözer — **önerilen ilk iterasyon**, çünkü Teams'ten interaktif onay almak
     (Power Automate/Bot) başlı başına ayrı bir entegrasyon projesi.
- **Güvenlik notu:** Eğer bir HTTP approve endpoint'i açılırsa (3. madde), bu endpoint'in kimlik
  doğrulaması olmalı — Teams webhook'undan gelen her isteğin gerçekten yetkili kullanıcıdan
  geldiği doğrulanmalı (paylaşılan secret token ya da Azure AD).

---

## 4. Önerilen Sıralama (bağımlılıklara göre)

```
1. Faz D (basit versiyon) — Teams'e "diagnostic hazır" bildirimi + UI linki
   (bağımsız, hemen başlanabilir, risk düşük — sadece bir webhook POST)
        │
2. Faz E (UI-tabanlı onay) — UI'da "Müşteriye Gönder" butonu, 09_email_report'u
   pipeline sonundan çıkarıp bu butona bağlamak
        │
3. Faz A — FTP entegrasyonu (FTP sunucu bilgisi netleşince başlanabilir,
   paralel de yürüyebilir — Faz D/E'den bağımsız)
        │
4. Faz B — Zamanlanmış tetikleme (Faz A bitmeden anlamsız)
        │
5. Faz C — netleştirme sonrası, muhtemelen ayrı iş kalemi olmaz (Faz D'nin
   diagnostic içeriği yeterliyse kapanır)
        │
6. (İleri seviye, opsiyonel) Faz D/E'yi Power Automate ile tam interaktif
   Teams onayına yükseltmek — ilk versiyon stabil çalıştıktan sonra
```

**Gerekçe:** Faz D+E (UI-tabanlı onay + basit Teams bildirimi) en düşük riskli, en hızlı değer
üreten adım — mevcut kodun büyük kısmını (07/08/09) hiç değiştirmeden sadece tetikleme noktasını
ve bir bildirim POST'unu ekliyor. FTP/zamanlama (A+B) dış sistem bağımlılığı olduğu için süre
tahmini kullanıcı/IT'den gelecek bilgiye bağlı.

---

## 5. Netleştirilmesi Gereken Sorular (bir sonraki konuşmada sorulmalı)

1. FTP sunucu bilgisi (host/port/protokol SFTP mi FTP mi/kimlik bilgisi kaynağı) kimden gelecek?
2. Teams tarafında hangi entegrasyon seçeneği mevcut/izinli — Incoming Webhook mı, Power Automate
   mı, yoksa kurumsal bot politikası bunları da mı kısıtlıyor? (IT'ye sorulması gerekebilir.)
3. Onay adımı gerçekten Teams içinden mi olmalı, yoksa "Teams'te bildirim gör → UI'da onayla" akışı
   yeterli mi? (Yukarıdaki plan ikinciyi öneriyor — daha az entegrasyon riski.)
4. Müşteriye giden email şu an sadece iç ekibe (`emre.hangul`, `cagatay.bayrak`) gidiyor
   (`pipeline/09_email_report.py:26`) — müşterinin gerçek email adresi ne zaman eklenecek, kim
   onaylayacak bu genişlemeyi?
5. Faz C'deki "ufak reshape" işleminin somut içeriği ne — otomasyona alınabilir bir örnek var mı?

---

## 6. Bu Dokümanın Kapsamadığı Şey

Bu planın hiçbir maddesi henüz uygulanmadı. `ui/tab_tahmin_uret.py`'ye 07/08/09 adımlarının
eklenmesi (bu konuşmada tamamlanan tek kod değişikliği) bu planın **önkoşulu** değildi ama
UI'ı "tek giriş noktası" yapma isteğiyle doğrudan uyumlu olduğu için önce o yapıldı.
