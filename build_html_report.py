# -*- coding: utf-8 -*-
import base64, os

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, "output")
IMAGES = [f for f in os.listdir(OUTPUT) if f.endswith(".png")]
b64 = {}
for img in sorted(IMAGES):
    with open(os.path.join(OUTPUT, img), "rb") as f:
        b64[img] = base64.b64encode(f.read()).decode()

# Read klima profili CSVs
import pandas as pd
adm_csv = pd.read_csv(os.path.join(OUTPUT, "klima_profili_ADM.csv"))
gdz_csv = pd.read_csv(os.path.join(OUTPUT, "klima_profili_GDZ.csv"))
adm_hours = list(adm_csv["hour"])
adm_cl = [round(v, 1) for v in adm_csv["cooling_load_mwh"]]
gdz_hours = list(gdz_csv["hour"])
gdz_cl = [round(v, 1) for v in gdz_csv["cooling_load_mwh"]]

# Chart.js script for bar chart
bar_script = f"""
new Chart(document.getElementById('coolingBar'), {{
    type: 'bar',
    data: {{
        labels: {adm_hours},
        datasets: [
            {{ label: 'ADM (MWh)', data: {adm_cl}, backgroundColor: 'rgba(44,123,182,0.75)', borderColor: '#2c7bb6', borderWidth: 1 }},
            {{ label: 'GDZ (MWh)', data: {gdz_cl}, backgroundColor: 'rgba(215,25,28,0.75)', borderColor: '#d7191c', borderWidth: 1 }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ position: 'top' }} }},
        scales: {{ y: {{ title: {{ display: true, text: 'Klima Yukü (MWh)' }} }}, x: {{ title: {{ display: true, text: 'Saat' }} }} }}
    }}
}});
"""

html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Klima Etkisi Analizi — ADM &amp; GDZ</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;line-height:1.7}}
.hero{{background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);padding:60px 20px;text-align:center;border-bottom:3px solid #38bdf8}}
.hero h1{{font-size:2.4em;font-weight:800;background:linear-gradient(90deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.hero p{{color:#94a3b8;margin-top:10px;font-size:1.1em}}
.container{{max-width:1100px;margin:0 auto;padding:20px 30px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin:30px 0}}
.card{{background:#1e293b;border-radius:14px;padding:24px;border:1px solid #334155;transition:transform .2s}}
.card:hover{{transform:translateY(-3px)}}
.card .value{{font-size:2em;font-weight:800;color:#38bdf8}}
.card .label{{color:#94a3b8;font-size:.9em;margin-top:4px}}
.card .sub{{font-size:.8em;color:#64748b;margin-top:4px}}
.card.orange .value{{color:#fb923c}}
.card.green .value{{color:#4ade80}}
.card.red .value{{color:#f87171}}
.section{{background:#1e293b;border-radius:16px;padding:32px;margin:30px 0;border:1px solid #334155}}
.section h2{{font-size:1.6em;color:#38bdf8;margin-bottom:6px;display:flex;align-items:center;gap:10px}}
.section h2 .emoji{{font-size:1.4em}}
.section .meta{{color:#64748b;font-size:.85em;margin-bottom:20px}}
.section h3{{color:#e2e8f0;margin:24px 0 10px;font-size:1.2em}}
.section p,.section li{{color:#cbd5e1;margin-bottom:8px}}
.section ul,.section ol{{padding-left:24px;margin-bottom:12px}}
.tldr{{background:#0f3460;border-left:4px solid #38bdf8;padding:14px 20px;border-radius:8px;margin:16px 0;font-weight:600}}
.tldr span{{color:#38bdf8}}
.chart-box{{background:#0f172a;border-radius:12px;padding:20px;margin:20px 0;text-align:center;overflow:hidden}}
.chart-box img{{max-width:100%;height:auto;border-radius:8px}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
@media(max-width:800px){{.chart-grid{{grid-template-columns:1fr}}}}
table{{width:100%;border-collapse:collapse;margin:16px 0;font-size:.95em}}
th{{background:#0f172a;color:#38bdf8;padding:12px 16px;text-align:left;font-weight:700}}
td{{padding:12px 16px;border-bottom:1px solid #334155}}
tr:nth-child(even){{background:rgba(15,23,42,.4)}}
.highlight{{background:rgba(56,189,248,.1);border-radius:8px;padding:16px 20px;margin:12px 0}}
.canvas-box{{background:#0f172a;border-radius:12px;padding:20px;margin:20px 0}}
kbd{{background:#334155;padding:2px 8px;border-radius:4px;font-size:.85em;color:#38bdf8}}
footer{{text-align:center;padding:40px;color:#475569;font-size:.85em}}
</style>
</head>
<body>

<div class="hero">
<h1>🌡️ Klima Etkisi STLF Analizi</h1>
<p>ADM &amp; GDZ Bölgeleri Karşılaştırmalı — 2018–2026 8 Yıllık Veri</p>
</div>

<div class="container">

<!-- ═══ KEY METRICS ═══ -->
<div class="cards">
<div class="card">
<div class="value">22.7°C</div>
<div class="label">ADM Klima Tetiklenme</div>
<div class="sub">Bu sıcaklıktan sonra tüketim hızla artıyor</div>
</div>
<div class="card orange">
<div class="value">26.8°C</div>
<div class="label">GDZ Klima Tetiklenme</div>
<div class="sub">GDZ daha yüksek sıcaklıkta klimaları açıyor</div>
</div>
<div class="card">
<div class="value">76.3 MWh/°C</div>
<div class="label">ADM Soğutma Hassasiyeti</div>
<div class="sub">Her 1°C sıcaklık artışında ek tüketim</div>
</div>
<div class="card orange">
<div class="value">122.2 MWh/°C</div>
<div class="label">GDZ Soğutma Hassasiyeti</div>
<div class="sub">GDZ her dereceye daha sert tepki veriyor</div>
</div>
<div class="card green">
<div class="value">20°C</div>
<div class="label">ADM En İyi CDD Eşiği</div>
<div class="sub">r = 0.747 — en yüksek korelasyon</div>
</div>
<div class="card green">
<div class="value">24°C</div>
<div class="label">GDZ En İyi CDD Eşiği</div>
<div class="sub">r = 0.718 — ADM'den yüksek eşik</div>
</div>
<div class="card red">
<div class="value">15:00</div>
<div class="label">Klima Tepe Saati (ortak)</div>
<div class="sub">Her iki bölgede de öğleden sonra zirve</div>
</div>
<div class="card">
<div class="value">1.63x</div>
<div class="label">GDZ / ADM Tüketim Oranı</div>
<div class="sub">GDZ, ADM'den %63 daha fazla tüketiyor</div>
</div>
</div>

<!-- ═══ SECTION 1 ═══ -->
<div class="section">
<h2><span class="emoji">📈</span> 1. Sıcaklık — Tüketim İlişkisi</h2>
<div class="meta">Scatter plot + LOWESS eğrisi + Kırılım noktası tespiti</div>

<div class="tldr">
<span>🧠 Mala Anlat:</span> Sıcaklık arttıkça elektrik tüketimi de artıyor. Ama bu artış düz bir çizgi değil — belli bir sıcaklığa kadar yavaş, o noktadan sonra hızlı. İşte o nokta "klima devreye girme sıcaklığı". ADM'de 22.7°C, GDZ'de 26.8°C. Yani ADM bölgesindeki insanlar daha düşük sıcaklıkta klima açmaya başlıyor (turistik bölge etkisi), GDZ (İzmir-Manisa sanayi) daha yüksek sıcaklığa dayanıyor.
</div>

<p><strong>Bu ne demek?</strong> Grafikteki her nokta bir günü gösteriyor. Sıcak günlerde tüketim belirgin şekilde yüksek. Kırmızı çizgi (LOWESS) tüketimin sıcaklıkla nasıl değiştiğinin "yumuşatılmış" hali. Yeşil kesik çizgi ise klimanın devreye girdiği kırılım noktası.</p>

<div class="chart-grid">
<div class="chart-box"><img src="data:image/png;base64,{b64['sec1_scatter_ADM.png']}" alt="ADM Scatter"></div>
<div class="chart-box"><img src="data:image/png;base64,{b64['sec1_scatter_GDZ.png']}" alt="GDZ Scatter"></div>
</div>

<h3>Sıcaklık Dilimi Bazında Tüketim</h3>
<p>Aşağıdaki grafikler sıcaklığı 2'şer derecelik dilimlere bölüp her dilimdeki ortalama tüketimi gösteriyor. Hata çubukları (±1 standart sapma) o dilimde tüketimin ne kadar değişken olduğunu anlatıyor.</p>
<div class="chart-grid">
<div class="chart-box"><img src="data:image/png;base64,{b64['sec1_temp_bin_ADM.png']}" alt="ADM Temp Bin"></div>
<div class="chart-box"><img src="data:image/png;base64,{b64['sec1_temp_bin_GDZ.png']}" alt="GDZ Temp Bin"></div>
</div>
</div>

<!-- ═══ SECTION 2 ═══ -->
<div class="section">
<h2><span class="emoji">🕒</span> 2. Saatlik Klima Profili</h2>
<div class="meta">Sıcak günler (>24°C) vs Serin günler (<20°C) — Aradaki fark = klima yükü</div>

<div class="tldr">
<span>🧠 Mala Anlat:</span> Klimalar günün hangi saatinde en çok çalışıyor? Bunu bulmak için sıcak günlerin saatlik tüketiminden serin günlerin tüketimini çıkardık. Sonuç: <strong>klima en çok 15:00'te çalışıyor</strong>. Sabah 6'dan sonra yavaş yavaş açılıyor, öğlen hızlanıyor, 15:00'te zirve yapıyor, sonra azalarak gece 22:00 civarı kapanıyor. Bu profil neredeyse güneşin hareketiyle birebir aynı — GHI (güneş radyasyonu) ile paralel.
</div>

<div class="chart-grid">
<div class="chart-box"><img src="data:image/png;base64,{b64['sec2_hourly_profile_ADM.png']}" alt="ADM Hourly"></div>
<div class="chart-box"><img src="data:image/png;base64,{b64['sec2_hourly_profile_GDZ.png']}" alt="GDZ Hourly"></div>
</div>

<h3>📊 Klima Yükü Vektörü (24 Saat, MWh)</h3>
<div class="canvas-box"><canvas id="coolingBar" height="300"></canvas></div>

<div class="highlight">
<strong>ADM:</strong> Tepe 15:00 → 620 MWh | Toplam günlük klima yükü ~9,836 MWh<br>
<strong>GDZ:</strong> Tepe 15:00 → 687 MWh | Toplam günlük klima yükü ~10,008 MWh
</div>
</div>

<!-- ═══ SECTION 3 ═══ -->
<div class="section">
<h2><span class="emoji">🗺️</span> 3. Bölgesel Sıcaklık Etkisi</h2>
<div class="meta">Hangi il / istasyon tüketimi en iyi açıklıyor? GHI (güneş) bağımsız etkiye sahip mi?</div>

<div class="tldr">
<span>🧠 Mala Anlat:</span> ADM bölgesinde 14 hava istasyonu var. Acaba hangi istasyonun sıcaklığı tüketimle en ilişkili? <strong>Kazanan: Muğla Dalaman!</strong> (r=0.43). Dalaman ovası, Yatağan ve Milas sanayi bölgeleri en yüksek korelasyona sahip. Bu mantıklı: buralar hem nüfus yoğun hem de yazın çok sıcak. Denizli istasyonları orta sırada, Aydın Büyük Menderes yine yüksek.
</div>
<p><strong>GDZ'de</strong> İzmir (r=0.30) ile Manisa (r=0.30) neredeyse eşit etkiye sahip — sanayi yükü her iki bölgeye de yayılmış durumda.</p>
<p><strong>GHI (güneş radyasyonu)</strong> sıcaklıktan bağımsız olarak tüketimi neredeyse etkilemiyor (kısmi korelasyon r=-0.06). Yani güneşin etkisi zaten sıcaklık üzerinden yansıyor, ayrı bir "güneşli ama serin" klima etkisi yok.</p>

<div class="chart-grid">
<div class="chart-box"><img src="data:image/png;base64,{b64['sec3_regional_adm.png']}" alt="Regional ADM"></div>
<div class="chart-box"><img src="data:image/png;base64,{b64['sec3_ghi_effect_adm.png']}" alt="GHI Effect"></div>
</div>
</div>

<!-- ═══ SECTION 4 ═══ -->
<div class="section">
<h2><span class="emoji">📅</span> 4. Hafta İçi / Hafta Sonu Klima Farkı</h2>
<div class="meta">Sıcak günlerde hafta içi ve hafta sonu klima kullanımı aynı mı?</div>

<div class="tldr">
<span>🧠 Mala Anlat:</span> Hafta sonu evdeyiz, hafta içi işteyiz. Peki sıcak günlerde klima kullanımı değişiyor mu? <strong>Evet, hem de çok!</strong> ADM'de hafta sonu tüketim hafta içine göre <strong>98 MWh daha düşük</strong>. GDZ'de bu fark <strong>274 MWh</strong>. Yani sanayi ağırlıklı GDZ bölgesinde hafta sonu fabrikalar durduğu için klima yükü dramatik düşüyor. ADM'de turistik tesisler ve konutlar hafta sonu da çalıştığı için düşüş daha az.
</div>

<div class="chart-grid">
<div class="chart-box"><img src="data:image/png;base64,{b64['sec4_weekday_weekend_ADM.png']}" alt="Weekday ADM"></div>
<div class="chart-box"><img src="data:image/png;base64,{b64['sec4_weekday_weekend_GDZ.png']}" alt="Weekday GDZ"></div>
</div>

<table>
<tr><th>Metrik</th><th>ADM</th><th>GDZ</th></tr>
<tr><td>Hafta içi ort. (sıcak gün)</td><td>1,512 MWh</td><td>2,251 MWh</td></tr>
<tr><td>Hafta sonu ort. (sıcak gün)</td><td>1,414 MWh</td><td>1,977 MWh</td></tr>
<tr><td>Fark</td><td><strong>-98 MWh</strong> (%6.5)</td><td><strong>-274 MWh</strong> (%12.2)</td></tr>
</table>
</div>

<!-- ═══ SECTION 5 ═══ -->
<div class="section">
<h2><span class="emoji">🌡️</span> 5. CDD (Cooling Degree Days) Analizi</h2>
<div class="meta">Farklı eşik değerleri için soğutma derece-gün hesabı — hangisi tüketimi en iyi açıklıyor?</div>

<div class="tldr">
<span>🧠 Mala Anlat:</span> CDD = "bugün hava kaç derece soğutma ihtiyacı yarattı" ölçüsü. Formül basit: CDD = max(0, Sıcaklık − Eşik). 4 farklı eşik denedik (18, 20, 22, 24°C) ve tüketimle korelasyonuna baktık. <strong>ADM için en iyi eşik 20°C</strong> (r=0.747), <strong>GDZ için 24°C</strong> (r=0.718). Bu da mantıklı: ADM daha serin bir iklimde (Muğla kıyı), insanlar daha düşük sıcaklıkta soğutma ihtiyacı duyuyor. GDZ (İzmir iç kesim) daha yüksek sıcaklığa alışık.
</div>

<div class="chart-grid">
<div class="chart-box"><img src="data:image/png;base64,{b64['sec5_cdd_ADM.png']}" alt="CDD ADM"></div>
<div class="chart-box"><img src="data:image/png;base64,{b64['sec5_cdd_GDZ.png']}" alt="CDD GDZ"></div>
</div>

<h3>Aylık CDD ve Tüketim Trendi</h3>
<div class="chart-grid">
<div class="chart-box"><img src="data:image/png;base64,{b64['sec5_monthly_cdd_ADM.png']}" alt="Monthly ADM"></div>
<div class="chart-box"><img src="data:image/png;base64,{b64['sec5_monthly_cdd_GDZ.png']}" alt="Monthly GDZ"></div>
</div>

<table>
<tr><th>Eşik</th><th>ADM r</th><th>GDZ r</th></tr>
<tr><td>18°C</td><td>0.734</td><td>0.626</td></tr>
<tr><td><strong>20°C</strong></td><td><strong>0.747 ✅</strong></td><td>0.668</td></tr>
<tr><td>22°C</td><td>0.741</td><td>0.701</td></tr>
<tr><td><strong>24°C</strong></td><td>0.702</td><td><strong>0.718 ✅</strong></td></tr>
</table>
</div>

<!-- ═══ SECTION 6 ═══ -->
<div class="section">
<h2><span class="emoji">⚔️</span> 6. ADM vs GDZ Karşılaştırması</h2>
<div class="meta">İki bölge arasındaki farklar — kim daha çok tüketiyor, kim sıcağa daha hassas?</div>

<div class="tldr">
<span>🧠 Mala Anlat:</span> GDZ bölgesi ADM'den <strong>1.63 kat daha fazla</strong> elektrik tüketiyor (1905 vs 1167 MWh/gün). Bunun sebebi GDZ'nin İzmir gibi büyük bir sanayi metropolünü kapsaması, ADM'nin ise Muğla-Aydın-Denizli gibi daha turistik ve tarımsal bölgeleri kapsaması. <strong>Ama asıl ilginç olan:</strong> GDZ'nin soğutma hassasiyeti (122 MWh/°C) ADM'den (76 MWh/°C) çok daha yüksek. Yani GDZ'de hava 1°C ısındığında tüketimdeki sıçrama ADM'ye göre %60 daha fazla. Bu da sanayi soğutmasının konut soğutmasından daha enerji-yoğun olduğunu gösteriyor.
</div>

<div class="chart-box"><img src="data:image/png;base64,{b64['sec6_adm_vs_gdz.png']}" alt="ADM vs GDZ"></div>

<table>
<tr><th>Karşılaştırma</th><th>ADM</th><th>GDZ</th></tr>
<tr><td>Günlük ortalama tüketim</td><td>1,167 MWh</td><td>1,905 MWh</td></tr>
<tr><td>Klima tetiklenme sıcaklığı</td><td>22.7°C</td><td>26.8°C</td></tr>
<tr><td>Soğutma hassasiyeti (MWh/°C)</td><td>76.3</td><td>122.2</td></tr>
<tr><td>En iyi CDD eşiği</td><td>20°C</td><td>24°C</td></tr>
<tr><td>Hafta sonu düşüşü</td><td>-98 MWh (%6.5)</td><td>-274 MWh (%12.2)</td></tr>
<tr><td>Bölge tipi</td><td>Turistik / Tarım</td><td>Sanayi / Metropol</td></tr>
</table>
</div>

<!-- ═══ SECTION 7 ═══ -->
<div class="section">
<h2><span class="emoji">💡</span> 7. STLF Modeli İçin Yeni Feature Önerileri</h2>
<div class="meta">Bu analizden çıkan sonuçlarla tahmin modeline eklenebilecek özellikler</div>

<ol>
<li><strong>CDD eşiği optimizasyonu:</strong> ADM için CDD_Cooling_Stress eşiği <kbd>20°C</kbd>'ye, GDZ için <kbd>24°C</kbd>'ye güncellenmeli. Mevcut eşik değeri bu değerlerle karşılaştırılıp optimize edilmeli.</li>
<li><strong>GHI × Sıcaklık etkileşimi:</strong> GHI'nin bağımsız etkisi ihmal edilebilir düzeyde (r=-0.06). Mevcut GHI feature'ları yeterli, ek etkileşim feature'ına gerek yok.</li>
<li><strong>Rolling 3-gün sıcaklık ortalaması:</strong> Sıcak dalgası etkisini yakalamak için — üst üste sıcak günlerde klima kullanımı artıyor (ısı birikimi). Modele eklenmeli.</li>
<li><strong>Sıcaklık rampa hızı (ΔT/Δt):</strong> Düne göre kaç derece değişti? Ani sıcaklık artışlarında klima kullanımı daha agresif olabilir.</li>
<li><strong>Saatlik klima profili regresörü:</strong> 15:00 tepe saatini yakalayan bir <kbd>Cooling_Peak_Hour</kbd> dummy veya saat bazlı ağırlıklandırma.</li>
<li><strong>Hafta sonu cooling scale faktörü:</strong> GDZ'de hafta sonu klima etkisi hafta içine göre ~%12 daha düşük. Hafta sonu günlerinde CDD/Cooling feature'larına scale faktörü uygulanabilir.</li>
<li><strong>Bölgesel sıcaklık ağırlıkları:</strong> Dalaman, Yatağan, Milas istasyonlarına daha yüksek ağırlık verilebilir. Mevcut eşit ağırlıklı ortalama yerine korelasyon-bazlı ağırlıklandırma denenebilir.</li>
</ol>
</div>

</div>

<footer>
STLF Klima Etkisi Analizi · 2018–2026 · ADM &amp; GDZ Karşılaştırmalı · Otomatik oluşturuldu
</footer>

<script>
{bar_script}
</script>
</body>
</html>"""

out_path = os.path.join(BASE, "output", "klima_analizi_raporu.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Rapor kaydedildi: {out_path}")
