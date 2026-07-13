"""
monitoring/ — ADM + GDZ ortak izleme katmanı (Faz 2, 2026-07-10)
====================================================================
STLF_MONITORING_REFACTOR_PLAN.md §4'te tasarlanan "iki tenant, tek kod"
paketi. adm live/ (git repo) altında yaşıyor; GDZ tarafı sys.path ile
buraya erişir (gdz talep/live/src/forecast_logger.py ve scorecard.py'deki
ince shim'lere bakın).

Neden burada, ADM'nin şemasına/mantığına GENELLİKLE dokunulmadan:
  - forecast_log/actuals_log şeması ADM ile GDZ arasında zaten birebir aynı
    (GDZ_LIVE_PORTING_PLAN.md Faz 6.2'de kasıtlı olarak böyle tasarlandı).
  - scorecard.py mantığı 2026-07-10'a kadar iki ayrı, birebir aynı kopya
    olarak yaşıyordu (GDZ'ninki hiç yazılmamıştı bile — bu paketin var
    olma sebebi tam olarak bu sınıf hatayı bir daha imkânsız kılmak).
  - write_forecast_log/update_actuals_log/update_actuals_log_weather
    BİLEREK burada DEĞİL — GDZ'nin T1/T2 ayrı kolon şeması (coalesce
    gerektiren) ile ADM'nin doğrudan kolon şeması arasında gerçek,
    yapısal bir fark var; bunu zorla tek fonksiyona sıkıştırmak
    yapay bir soyutlama olurdu. Bu üçü her tenant'ın kendi
    src/forecast_logger.py'sinde tenant-spesifik kalır.
"""
