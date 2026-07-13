"""
diagnostic_core.py — ADM + GDZ ortak interaktif diagnostic motoru
==================================================================
TEK KAYNAK: hem veri hesabı hem HTML/JS render burada. ADM ve GDZ
scriptleri sadece kendi kolonlarını "kanonik" isimlere map edip
compute() + render() çağırır. Böylece iki EDAŞ hiçbir zaman ayrışmaz
(eski "ADM template'ini string-splice et" yaklaşımının GDZ'yi bozması
bu modülle kökten çözülür).

Kanonik `merged` kolonları (wrapper doldurur):
    dt      : normalize edilmiş gün (datetime)
    h       : saat 0-23 (int)
    load    : hedef tüketim (MWh)
    temp    : hissedilen sıcaklık (°C)
    ghi     : global ışınım (W/m²)   [opsiyonel]
    cloud   : bulutluluk (%)          [opsiyonel]
    wind    : rüzgar (m/s veya km/s)  [opsiyonel]
    special : özel gün adı (str) ya da "Değil"/None  [opsiyonel]

fc      : D+2 tahmini, 24 değerli liste
fc_wx   : {"temp":[24], "ghi":[24]}  D+2 tahmin havası [opsiyonel]
"""
import json
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

# ── Sabitler ──────────────────────────────────────────────────────────
SEASONS = [("Kis", [12, 1, 2]), ("Ilkbahar", [3, 4, 5]),
           ("Yaz", [6, 7, 8]), ("Sonbahar", [9, 10, 11])]
TEMP_BINS = [(-10, 5), (5, 10), (10, 15), (15, 20),
             (20, 25), (25, 30), (30, 35), (35, 50)]
HOUR_GROUPS = [("Gece (00-06)", range(0, 7)), ("Sabah (07-10)", range(7, 11)),
               ("Ogle (11-16)", range(11, 17)), ("Aksam (17-23)", range(17, 24))]
CMP_PAIRS = [(7, "1 hafta once"), (14, "2 hafta once"),
             (364, "1 yil once"), (371, "1 yil + 1 hafta once")]


def _f(x):
    """numpy/nan güvenli float→JSON."""
    try:
        v = float(x)
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 3)
    except (TypeError, ValueError):
        return None


def _slope(x, y):
    """Basit lineer eğim (MW/°C). Yetersiz/degenerate veride None."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 15 or np.std(x) < 0.8:
        return None
    b = np.polyfit(x, y, 1)[0]
    return _f(b)


# ══════════════════════════════════════════════════════════════════════
#  COMPUTE
# ══════════════════════════════════════════════════════════════════════
def compute(merged, fc, fc_wx, fc_date, edas):
    merged = merged.copy()
    merged['dow'] = merged['dt'].dt.dayofweek
    merged['month'] = merged['dt'].dt.month
    merged['doy'] = merged['dt'].dt.dayofyear
    TODAY = date.fromisoformat(fc_date)
    has_ghi = bool('ghi' in merged and merged['ghi'].notna().sum() > 100)
    has_cloud = bool('cloud' in merged and merged['cloud'].notna().sum() > 100)
    has_wind = bool('wind' in merged and merged['wind'].notna().sum() > 100)
    has_special = 'special' in merged

    # özel gün adı → tarih haritası
    special_map = {}
    if has_special:
        sp = merged.dropna(subset=['special'])
        sp = sp[~sp['special'].astype(str).isin(['Değil', 'Degil', 'nan', ''])]
        for d, name in sp.groupby(sp['dt'].dt.date)['special'].first().items():
            special_map[str(d)] = str(name)

    def gs(ds):
        d = pd.Timestamp(ds).date()
        day = merged[merged['dt'].dt.date == d]
        if len(day) < 20:
            return None
        day = day.set_index('h').sort_index().reindex(range(24))
        r = {"load": [_f(v) for v in day['load'].values]}
        if day['load'].notna().sum() < 18:
            return None
        r["temp"] = [_f(v) for v in day['temp'].values] if 'temp' in day else None
        if has_ghi:   r["ghi"] = [_f(v) for v in day['ghi'].values]
        if has_cloud: r["cloud"] = [_f(v) for v in day['cloud'].values]
        if has_wind:  r["wind"] = [_f(v) for v in day['wind'].values]
        r["special"] = special_map.get(str(d))
        return r

    # ── CP: karşılaştırma günleri ─────────────────────────────────────
    cp = {}
    if fc:
        for off, lbl in CMP_PAIRS:
            s = gs(str(TODAY - timedelta(days=off)))
            if s:
                cp[lbl] = s

    # ── P95: saatlik hafta-üstü hata bandı ────────────────────────────
    p95 = {}
    for h in range(24):
        seg = merged[merged['h'] == h].dropna(subset=['load'])
        if len(seg) < 40:
            continue
        err = seg['load'].values - seg['load'].shift(168).values
        err = err[~np.isnan(err)]
        if len(err) > 50:
            ml = float(np.mean(seg['load'].values))
            p95[h] = {"p5_ape": _f(np.percentile(np.abs(err), 5) / ml * 100),
                      "p95_ape": _f(np.percentile(np.abs(err), 95) / ml * 100),
                      "p50_ape": _f(np.median(np.abs(err)) / ml * 100),
                      "p5_err": _f(np.percentile(err, 5)),
                      "p95_err": _f(np.percentile(err, 95))}

    # ── SN: sezon / saat-grubu / gün-tipi global eğim (MW/°C) ─────────
    sn = {}
    seg = merged.dropna(subset=['temp', 'load'])
    if len(seg) > 100:
        for nm, ms in SEASONS:
            s = _slope(seg[seg['month'].isin(ms)]['temp'], seg[seg['month'].isin(ms)]['load'])
            if s is not None: sn[nm] = s
        for hg, hr in HOUR_GROUPS:
            s = _slope(seg[seg['h'].isin(hr)]['temp'], seg[seg['h'].isin(hr)]['load'])
            if s is not None: sn[hg] = s
        for lbl, msk in [("Haftaici", seg['dow'] < 5), ("Cumartesi", seg['dow'] == 5),
                         ("Pazar", seg['dow'] == 6)]:
            s = _slope(seg[msk]['temp'], seg[msk]['load'])
            if s is not None: sn[lbl] = s

    # ── SNB: SICAKLIK ARALIĞI (bin) bazında yerel duyarlılık ──────────
    # "20-25°C aralığında her +1°C → +X MW" — sezon kırılımlı.
    snb = {"bins": [], "overall": [], "season": {}}
    if len(seg) > 200:
        for lo, hi in TEMP_BINS:
            snb["bins"].append(f"{lo}-{hi}")
            bs = seg[(seg['temp'] >= lo) & (seg['temp'] < hi)]
            snb["overall"].append({
                "slope": _slope(bs['temp'], bs['load']),
                "n": int(len(bs)),
                "mean_load": _f(bs['load'].mean()) if len(bs) else None,
                "mean_temp": _f(bs['temp'].mean()) if len(bs) else None,
            })
        for nm, ms in SEASONS:
            ss = seg[seg['month'].isin(ms)]
            row = []
            for lo, hi in TEMP_BINS:
                bs = ss[(ss['temp'] >= lo) & (ss['temp'] < hi)]
                row.append({"slope": _slope(bs['temp'], bs['load']), "n": int(len(bs))})
            snb["season"][nm] = row

    # ── HTE: saatlik sıcaklık etkisi (son 7 gün) ──────────────────────
    hte = {}
    for h in range(24):
        sh = merged[merged['h'] == h].dropna(subset=['temp', 'load']).tail(21)
        if len(sh) >= 10 and np.std(sh['temp'].values) > 0.5:
            sl = np.polyfit(sh['temp'].values, sh['load'].values, 1)[0]
            r = np.corrcoef(sh['temp'].values, sh['load'].values)[0, 1]
            hte[h] = {"slope": _f(sl), "r2": _f(r * r), "n": int(len(sh))}

    # ── REC: HAVA-KOŞULLU saatlik öneri motoru + P95 aralık ───────────
    # Her saat için: geçmişte (aynı saat, ±ay penceresi) sıcaklık/ışınıma
    # göre beklenen yük + P95 aralığı; model tahminini kıyasla.
    rec = []
    ref = cp.get("1 hafta once")
    doy0 = TODAY.timetuple().tm_yday
    tgt_weekend = TODAY.weekday() >= 5
    # aynı mevsim (±40 gün) + aynı gün-tipi (haftaici/haftasonu) + son 3 yıl:
    # yıl-üstü yük büyümesi ve haftaici/haftasonu karışımı P95'i şişirmesin.
    dd = np.minimum(np.abs(merged['doy'] - doy0), 365 - np.abs(merged['doy'] - doy0))
    recent = merged['dt'] >= (pd.Timestamp(TODAY) - pd.Timedelta(days=1100))
    daytype = (merged['dow'] >= 5) if tgt_weekend else (merged['dow'] < 5)
    base_mask = (dd <= 40) & recent & daytype
    t0 = pd.Timestamp(TODAY)
    tgt_t = 0.0  # bugün referans; geçmiş günler negatif (yıl-üstü büyüme trendi)
    for h in range(24):
        temp_fc = fc_wx.get("temp", [None] * 24)[h] if fc_wx else None
        ghi_fc = fc_wx.get("ghi", [None] * 24)[h] if fc_wx else None
        sh = merged[base_mask & (merged['h'] == h)].dropna(subset=['temp', 'load'])
        if len(sh) < 25:  # gün-tipi filtresi çok daralttıysa mevsim+son 3 yıla düş
            sh = merged[(dd <= 40) & recent & (merged['h'] == h)].dropna(subset=['temp', 'load'])
        entry = {"h": h, "fc": _f(fc[h]) if fc else None, "temp_fc": _f(temp_fc),
                 "n": int(len(sh)), "exp": None, "lo": None, "hi": None,
                 "lastweek": _f(ref["load"][h]) if ref and ref.get("load") else None}
        if len(sh) >= 25 and temp_fc is not None:
            X = sh['temp'].values.astype(float)
            Y = sh['load'].values.astype(float)
            # zaman trendi (yıl gün): yıl-üstü yük büyümesini soğurur, seviyeyi bugüne getirir
            tt = (sh['dt'] - t0).dt.days.values.astype(float) / 365.0
            cols = [np.ones_like(X), X, tt]
            xq = [1.0, float(temp_fc), tgt_t]
            if has_ghi and ghi_fc is not None and sh['ghi'].notna().sum() > 20:
                g = sh['ghi'].fillna(sh['ghi'].median()).values.astype(float)
                if np.std(g) > 5:
                    cols.append(g); xq.append(float(ghi_fc))
            A = np.column_stack(cols)
            try:
                beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
                pred = float(np.dot(xq, beta))
                resid = Y - A.dot(beta)
                entry["exp"] = _f(pred)
                entry["lo"] = _f(pred + np.percentile(resid, 2.5))
                entry["hi"] = _f(pred + np.percentile(resid, 97.5))
                entry["slope_temp"] = _f(beta[1])
            except np.linalg.LinAlgError:
                pass
        rec.append(entry)

    # ── DRIFT: D→D+2 sıcaklık & tüketim kayması ───────────────────────
    drift = {"temp_fc": (fc_wx.get("temp") if fc_wx else None)}
    if ref and ref.get("temp"):
        drift["temp_lastweek"] = ref["temp"]
    if cp.get("2 hafta once", {}).get("temp"):
        drift["temp_2week"] = cp["2 hafta once"]["temp"]
    # son 14 gün günlük ortalama sıcaklık trendi
    tail = merged[merged['dt'] >= (pd.Timestamp(TODAY) - pd.Timedelta(days=16))]
    dmt = tail.dropna(subset=['temp']).groupby(tail['dt'].dt.date)['temp'].mean()
    drift["daily_temp"] = [[str(d), _f(v)] for d, v in dmt.items()]
    # tüketim benzerliği: FC vs her CP günü (MAPE + korelasyon)
    sim = []
    if fc:
        fca = np.array(fc, float)
        for lbl, s in cp.items():
            if not s.get("load"):
                continue
            la = np.array([x if x is not None else np.nan for x in s["load"]], float)
            mk = ~np.isnan(la) & (la > 0)
            if mk.sum() > 12:
                mape = float(np.mean(np.abs((fca[mk] - la[mk]) / la[mk])) * 100)
                corr = float(np.corrcoef(fca[mk], la[mk])[0, 1])
                sim.append({"lbl": lbl, "mape": _f(mape), "corr": _f(corr)})
    drift["sim"] = sim
    # son 7 gün: saatler-arası Δsıcaklık → Δyük (MW/°C)
    d7 = merged[merged['dt'] >= (pd.Timestamp(TODAY) - pd.Timedelta(days=8))].sort_values(['dt', 'h'])
    dl = d7['load'].diff().values; dtp = d7['temp'].diff().values
    mk = ~(np.isnan(dl) | np.isnan(dtp))
    if mk.sum() > 20 and np.std(dtp[mk]) > 0.3:
        drift["hourly_dtemp_slope"] = _f(np.polyfit(dtp[mk], dl[mk], 1)[0])

    # ── SPECIAL: özel gün etkileri ────────────────────────────────────
    special = {}
    fri = merged[(merged['dow'] == 4) & (merged['h'].isin([12, 13]))]
    base = merged[(merged['dow'].isin([1, 2, 3])) & (merged['h'].isin([12, 13]))]
    if len(fri) > 50 and len(base) > 100:
        fm, wm = fri['load'].mean(), base['load'].mean()
        special["friday"] = {"mw": _f(fm - wm), "pct": _f((fm - wm) / wm * 100)}
        # sıcaklık aralığına göre cuma etkisi
        by_temp = []
        for lo, hi in [(-10, 15), (15, 22), (22, 28), (28, 50)]:
            fb = fri[(fri['temp'] >= lo) & (fri['temp'] < hi)]
            bb = base[(base['temp'] >= lo) & (base['temp'] < hi)]
            if len(fb) > 10 and len(bb) > 20:
                by_temp.append({"range": f"{lo}-{hi}C", "mw": _f(fb['load'].mean() - bb['load'].mean()),
                                "n": int(len(fb))})
        special["friday_by_temp"] = by_temp
    # hafta sonu etkisi (saatlik, haftaiçi ortalamaya göre %)
    we, wd = {}, {}
    for h in range(24):
        a = merged[(merged['dow'] == 6) & (merged['h'] == h)]['load'].mean()
        b = merged[(merged['dow'] < 5) & (merged['h'] == h)]['load'].mean()
        if not np.isnan(a) and not np.isnan(b) and b > 0:
            we[h] = _f((a - b) / b * 100)
    special["sunday_pct"] = we
    # D+2 özel mi?
    special["target_is_special"] = special_map.get(fc_date)

    # ── L7: son 14 günün MAPE'si (fc dosyaları vs actual) — wrapper doldurur ──
    return {
        "D": fc_date, "EDAS": edas, "FC": [_f(v) for v in fc] if fc else None,
        "WX": fc_wx if fc_wx else None, "CP": cp, "SN": sn, "SNB": snb,
        "P95": p95, "P95K": [int(k) for k in p95.keys()], "HTE": hte,
        "REC": rec, "DRIFT": drift, "SPECIAL": special,
        "HAS": {"ghi": has_ghi, "cloud": has_cloud, "wind": has_wind},
    }


# ══════════════════════════════════════════════════════════════════════
#  RENDER
# ══════════════════════════════════════════════════════════════════════
def render(data, title, last7=None):
    data = dict(data)
    data["L7"] = last7 or []
    data_json = json.dumps(data, ensure_ascii=False)
    html = _TEMPLATE.replace("__TITLE__", title).replace(
        "__EDAS__", data.get("EDAS", "")).replace(
        "__FCD__", data.get("D", "")).replace(
        "/*__DATA__*/", "const DATA=" + data_json + ";")
    return html


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b0e14;color:#dfe6f3;font-family:'Segoe UI',sans-serif;font-size:13.5px}
.app{display:flex;min-height:100vh}
nav{width:190px;background:linear-gradient(180deg,#0d111a,#0b0e14);border-right:1px solid #26304a;padding:12px 0;position:sticky;top:0;height:100vh;overflow:auto;flex-shrink:0}
nav h1{font-size:13px;color:#5ad1a0;margin:4px 12px 2px;font-weight:700}
nav .sub{font-size:10px;color:#8b97b3;margin:0 12px 10px}
nav button{display:block;width:100%;text-align:left;background:none;border:0;color:#8b97b3;padding:8px 12px;cursor:pointer;font-size:12px;border-left:3px solid transparent}
nav button:hover{color:#dfe6f3;background:#11151f}
nav button.on{color:#dfe6f3;background:#11151f;border-left-color:#5ad1a0;font-weight:600}
main{flex:1;padding:18px 22px 60px;max-width:1180px}
.tab{display:none;animation:fade .2s}.tab.on{display:block}
@keyframes fade{from{opacity:0}to{opacity:1}}
h2{font-size:18px;margin:0 0 3px;font-weight:700}
.lead{color:#8b97b3;margin:0 0 14px;font-size:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:14px}
.card{background:#131823;border:1px solid #26304a;border-radius:9px;padding:10px 12px}
.card .lab{font-size:10px;color:#8b97b3;text-transform:uppercase}
.card .val{font-size:20px;font-weight:700;margin-top:2px}
.chartbox{position:relative;height:260px;margin-bottom:8px}
.chartbox.tall{height:340px}
.panel{background:#131823;border:1px solid #26304a;border-radius:10px;padding:12px 14px;margin-bottom:12px}
.panel h3{margin:0 0 2px;font-size:13px;color:#c7d0e3}
.panel .note{color:#8b97b3;font-size:11px;margin:0 0 8px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
table.dt{width:100%;font-size:11px;color:#c7d0e3;border-collapse:collapse}
table.dt th{color:#8b97b3;font-weight:600;padding:3px 5px;text-align:center;border-bottom:1px solid #26304a}
table.dt td{padding:3px 5px;text-align:center;border-bottom:1px solid #1c2334}
.rm{padding:11px 14px;margin-bottom:9px;background:#131823;border:1px solid #26304a;border-left:4px solid #5ad1a0;border-radius:8px}
.rm.warn{border-left-color:#ffce6a}.rm.danger{border-left-color:#ff6b6b}.rm.ok{border-left-color:#5ad1a0}
.rm h4{margin:0 0 3px;font-size:13px}.rm p{margin:3px 0;font-size:12px;color:#c7d0e3;line-height:1.55}
.rm .meta{display:flex;gap:12px;flex-wrap:wrap;font-size:10px;color:#8b97b3;margin:4px 0}
.pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;font-weight:600}
input[type=range]{width:100%}
.slider-val{display:inline-block;background:#26304a;padding:2px 9px;border-radius:4px;font-weight:700}
.badge{font-size:10px;padding:1px 6px;border-radius:4px;background:#2a2140;color:#c9a7ff;margin-left:5px}
@media(max-width:820px){.grid2,.grid3{grid-template-columns:1fr}}
</style></head><body><div class="app">
<nav><h1>__EDAS__ STLF</h1><div class="sub">D+2 &middot; __FCD__</div><div id="nv"></div></nav>
<main id="mn"></main></div>
<script>
/*__DATA__*/
const D=DATA.D, FC=DATA.FC, WX=DATA.WX, CP=DATA.CP, SN=DATA.SN, SNB=DATA.SNB,
      L7=DATA.L7, P95=DATA.P95, P95K=DATA.P95K, HTE=DATA.HTE, REC=DATA.REC,
      DRIFT=DATA.DRIFT, SPECIAL=DATA.SPECIAL, HAS=DATA.HAS;
const hrs=Array.from({length:24},(_,i)=>i);
const pal=['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1'];
const GRID={color:'#1c2334'}, TICK={color:'#8b97b3'};
const AX=(t)=>({title:{display:!!t,text:t,color:'#8b97b3'},ticks:TICK,grid:GRID});
const LEG={labels:{color:'#8b97b3',boxWidth:10,font:{size:9}}};
const charts={};
function ch(id,cfg){const x=document.getElementById(id);if(!x)return;if(charts[id])charts[id].destroy();
  cfg.options=Object.assign({responsive:true,maintainAspectRatio:false},cfg.options||{});charts[id]=new Chart(x,cfg);}
function fx(v,d){return v==null?'&mdash;':(+v).toFixed(d==null?0:d);}
function g(id){document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===id));
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.id==='t_'+id));
  if(g._i[id])g._i[id]();location.hash=id;}
g._i={};const T={};
const avg=a=>a.reduce((x,y)=>x+(y||0),0)/a.length;

// ═══ TAB 1: OZET ═══
T.ozet=()=>{let h='<h2>Ozet</h2><p class="lead">'+D+' icin D+2 tahmini &mdash; '+DATA.EDAS+'</p><div class="cards">';
 if(FC){h+='<div class="card"><div class="lab">Ortalama</div><div class="val" style="color:#5ad1a0">'+fx(avg(FC))+'</div><div class="lab">MWh</div></div>';
 const mx=Math.max(...FC);h+='<div class="card"><div class="lab">Pik</div><div class="val" style="color:#ffce6a">'+fx(mx)+'</div><div class="lab">s.'+FC.indexOf(mx)+':00</div></div>';
 h+='<div class="card"><div class="lab">Min</div><div class="val" style="color:#7fb2ff">'+fx(Math.min(...FC))+'</div><div class="lab">MWh</div></div>';}
 if(P95K.length){const m=avg(P95K.map(k=>P95[k].p95_ape));h+='<div class="card"><div class="lab">P95 bant</div><div class="val" style="color:#ff8a8a;font-size:17px">&plusmn;'+fx(m,1)+'%</div></div>';}
 if(SPECIAL.target_is_special)h+='<div class="card" style="border-color:#8E24AA"><div class="lab">Ozel Gun</div><div class="val" style="color:#c9a7ff;font-size:14px">'+SPECIAL.target_is_special+'</div></div>';
 h+='</div>';
 if(L7.length)h+='<div class="panel"><h3>Son gunler MAPE (%)</h3><p class="note">Teslim edilen tahmin vs gerceklesme</p><div class="chartbox"><canvas id="c7"></canvas></div></div>';
 if(WX&&WX.temp)h+='<div class="panel"><h3>D+2 tahmin havasi</h3><div class="chartbox tall"><canvas id="cw"></canvas></div></div>';
 return h;};
g._i.ozet=()=>{
 if(L7.length)ch('c7',{type:'bar',data:{labels:L7.map(x=>x[0]),datasets:[{label:'MAPE%',data:L7.map(x=>x[1]),backgroundColor:L7.map(x=>x[1]<3?'rgba(90,209,160,.75)':x[1]<6?'rgba(255,206,106,.75)':'rgba(255,107,107,.75)')}]},options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true,...AX('%')},x:AX()}}});
 if(WX&&WX.temp)ch('cw',{type:'line',data:{labels:hrs,datasets:[{label:'Sicaklik C',data:WX.temp,borderColor:'#ff9f5a',borderWidth:2,pointRadius:2,yAxisID:'y'},WX.ghi?{label:'GHI W/m2',data:WX.ghi,borderColor:'#ffce6a',borderWidth:1.5,pointRadius:1,yAxisID:'y2'}:null].filter(Boolean)},options:{plugins:{legend:LEG},scales:{x:AX(),y:{position:'left',...AX('C')},y2:{position:'right',title:{display:true,text:'GHI',color:'#8b97b3'},ticks:TICK,grid:{drawOnChartArea:false}}}}});};

// ═══ TAB 2: KARSILASTIRMA ═══
T.karsi=()=>{const k=Object.keys(CP);if(!k.length||!FC)return '<h2>Karsilastirma</h2><p class="lead">Veri yok</p>';
 let badges=k.map(l=>CP[l].special?l+' <span class="badge">'+CP[l].special+'</span>':l).join(' &middot; ');
 let h='<h2>D+2 Karsilastirma</h2><p class="lead">Referans gunler: '+badges+'</p>';
 h+='<div class="panel"><h3>Yuk profili + P95 bandi</h3><div class="chartbox tall"><canvas id="c1"></canvas></div></div>';
 h+='<div class="grid2"><div class="panel"><h3>Normalize profil (yuk/ortalama)</h3><p class="note">Sekil karsilastirmasi</p><div class="chartbox"><canvas id="c2"></canvas></div></div>';
 h+='<div class="panel"><h3>Sicaklik karsilastirmasi</h3><div class="chartbox"><canvas id="c3"></canvas></div></div></div>';return h;};
g._i.karsi=()=>{const k=Object.keys(CP);if(!k.length||!FC)return;
 let d1=[{label:'TAHMIN '+D,data:FC,borderColor:'#E53935',borderWidth:3,pointRadius:3}];
 if(P95K.length){d1.push({label:'P95 ust',data:hrs.map(h=>P95[h]?FC[h]+P95[h].p95_err:null),borderColor:'transparent',backgroundColor:'rgba(229,57,53,.10)',pointRadius:0,fill:'+1'},
  {label:'P95 alt',data:hrs.map(h=>P95[h]?FC[h]+P95[h].p5_err:null),borderColor:'transparent',pointRadius:0,fill:false});}
 k.forEach((l,i)=>d1.push({label:l,data:CP[l].load,borderColor:pal[(i+1)%pal.length],borderWidth:1.5,pointRadius:0,borderDash:[6,3]}));
 ch('c1',{type:'line',data:{labels:hrs,datasets:d1},options:{plugins:{legend:LEG},scales:{x:AX(),y:AX('MWh')}}});
 let fm=avg(FC),d2=[{label:'TAHMIN',data:FC.map(v=>v/fm),borderColor:'#E53935',borderWidth:3,pointRadius:2}];
 k.forEach((l,i)=>{let m=avg(CP[l].load);d2.push({label:l,data:CP[l].load.map(v=>v/m),borderColor:pal[(i+1)%pal.length],borderWidth:1.5,pointRadius:0,borderDash:[6,3]});});
 ch('c2',{type:'line',data:{labels:hrs,datasets:d2},options:{plugins:{legend:LEG},scales:{x:AX(),y:AX('yuk/ort')}}});
 let d3=[];if(WX&&WX.temp)d3.push({label:'TAHMIN',data:WX.temp,borderColor:'#E53935',borderWidth:2,pointRadius:2});
 k.forEach((l,i)=>{if(CP[l].temp)d3.push({label:l,data:CP[l].temp,borderColor:pal[(i+1)%pal.length],borderWidth:1.5,pointRadius:0,borderDash:[6,3]});});
 if(d3.length)ch('c3',{type:'line',data:{labels:hrs,datasets:d3},options:{plugins:{legend:LEG},scales:{x:AX(),y:AX('C')}}});};

// ═══ TAB 3: SICAKLIK & DUYARLILIK ═══
T.sens=()=>{let h='<h2>Sicaklik &amp; Duyarlilik</h2><p class="lead">Sicaklik aralik/sezon/saat kirilimlarinda +1C -> kac MW yuk</p>';
 if(SNB&&SNB.bins&&SNB.bins.length){h+='<div class="panel"><h3>Sicaklik araligi bazinda duyarlilik (MW/C)</h3><p class="note">Yerel egim; her bar o sicaklik kovasindaki +1C etkisi</p><div class="chartbox"><canvas id="cb1"></canvas></div>';
  h+='<table class="dt"><tr><th>Sezon \\ Aralik</th>'+SNB.bins.map(b=>'<th>'+b+'C</th>').join('')+'</tr>';
  Object.keys(SNB.season).forEach(sz=>{h+='<tr><td style="color:#c7d0e3;font-weight:600">'+sz+'</td>'+SNB.season[sz].map(c=>{let v=c.slope;return '<td style="color:'+(v==null?'#4a5570':v>0?'#ff9f5a':'#5ad1a0')+'">'+(v==null?'&mdash;':(v>0?'+':'')+v)+'</td>';}).join('')+'</tr>';});
  h+='</table></div>';}
 if(SN&&Object.keys(SN).length)h+='<div class="panel"><h3>Sezon / saat-grubu / gun-tipi duyarlilik (MW/C)</h3><div class="chartbox"><canvas id="cs2"></canvas></div></div>';
 if(Object.keys(HTE).length)h+='<div class="panel"><h3>Saatlik sicaklik etkisi (son 7 gun, MW/C)</h3><div class="chartbox"><canvas id="cs1"></canvas></div></div>';
 h+='<div class="panel"><h3>Senaryo motoru</h3><p class="note">Sicaklik sapmasina gore beklenen yuk degisimi (aktif sezon egimi ile)</p>';
 h+='<div style="margin:10px 0"><label style="color:#8b97b3;font-size:12px">Sicaklik sapmasi: <span id="sval" class="slider-val">0 C</span></label>';
 h+='<input type="range" id="srange" min="-6" max="6" value="0" step="1"></div><div id="scenOut" style="font-size:12px;color:#c7d0e3"></div></div>';return h;};
g._i.sens=()=>{
 if(SNB&&SNB.bins&&SNB.bins.length)ch('cb1',{type:'bar',data:{labels:SNB.bins.map(b=>b+'C'),datasets:[{label:'MW/C',data:SNB.overall.map(o=>o.slope),backgroundColor:SNB.overall.map(o=>o.slope==null?'#33405e':o.slope>0?'rgba(255,159,90,.8)':'rgba(90,209,160,.8)')}]},options:{plugins:{legend:{display:false},tooltip:{callbacks:{afterLabel:(x)=>'n='+(SNB.overall[x.dataIndex].n||0)+', ort yuk '+fx(SNB.overall[x.dataIndex].mean_load)}}},scales:{y:AX('MW/C'),x:AX()}}});
 if(SN&&Object.keys(SN).length){const e=Object.entries(SN);ch('cs2',{type:'bar',data:{labels:e.map(x=>x[0]),datasets:[{label:'MW/C',data:e.map(x=>x[1]),backgroundColor:e.map(x=>x[1]>0?'rgba(255,159,90,.8)':'rgba(90,209,160,.8)')}]},options:{indexAxis:'y',plugins:{legend:{display:false}},scales:{x:AX('MW/C'),y:AX()}}});}
 if(Object.keys(HTE).length)ch('cs1',{type:'bar',data:{labels:hrs.map(h=>h+':00'),datasets:[{label:'MW/C',data:hrs.map(h=>HTE[h]?HTE[h].slope:0),backgroundColor:hrs.map(h=>HTE[h]&&HTE[h].slope>0?'rgba(255,159,90,.75)':'rgba(90,209,160,.75)')}]},options:{plugins:{legend:{display:false}},scales:{y:AX('MW/C'),x:AX()}}});
 const sr=document.getElementById('srange');
 // aktif sezon egimi
 const mo=+D.slice(5,7);const szName=(mo>=12||mo<=2)?'Kis':(mo<=5)?'Ilkbahar':(mo<=8)?'Yaz':'Sonbahar';
 const sl=SN[szName]||SN['Yaz']||0;
 function upd(dv){const el=document.getElementById('sval');if(el)el.textContent=(dv>0?'+':'')+dv+' C';
  const mw=Math.round(sl*dv);let o='<b>Aktif sezon ('+szName+'):</b> +1C -> '+sl+' MW<br>';
  o+='<b>'+ (dv>0?'+':'')+dv+'C sapma:</b> yuk ~<b style="color:'+(mw>0?'#ff9f5a':'#5ad1a0')+'">'+(mw>0?'+':'')+mw+' MW</b>';
  if(FC)o+=' &rarr; yeni pik ~<b>'+Math.round(Math.max(...FC)+mw)+' MWh</b>';
  document.getElementById('scenOut').innerHTML=o;}
 if(sr){sr.oninput=function(){upd(+this.value);};}upd(0);};

// ═══ TAB 4: CROSS CHECK ═══
T.cross=()=>{const k=Object.keys(CP);if(!k.length)return '<h2>Cross Check</h2><p class="lead">Veri yok</p>';
 let cells='<div class="panel"><h3>Sicaklik &times; Yuk</h3><div class="chartbox"><canvas id="cx1"></canvas></div></div>';
 if(HAS.ghi)cells+='<div class="panel"><h3>GHI &times; Yuk</h3><div class="chartbox"><canvas id="cx2"></canvas></div></div>';
 if(HAS.cloud)cells+='<div class="panel"><h3>Bulut &times; Yuk</h3><div class="chartbox"><canvas id="cx3"></canvas></div></div>';
 if(HAS.wind)cells+='<div class="panel"><h3>Ruzgar &times; Yuk</h3><div class="chartbox"><canvas id="cx4"></canvas></div></div>';
 return '<h2>Cross Check</h2><p class="lead">Referans gunlerde hava &times; yuk sacilimi</p><div class="grid2">'+cells+'</div>';};
g._i.cross=()=>{const k=Object.keys(CP);if(!k.length)return;
 function sc(id,key,ax){let ds=[];if(key==='temp'&&WX&&WX.temp&&FC)ds.push({label:'TAHMIN',data:WX.temp.map((v,i)=>({x:v,y:FC[i]})),backgroundColor:'#E53935',pointRadius:5});
  k.forEach((l,i)=>{if(CP[l][key]&&CP[l].load)ds.push({label:l,data:CP[l][key].map((v,j)=>({x:v,y:CP[l].load[j]})),backgroundColor:pal[(i+1)%pal.length],pointRadius:3});});
  if(ds.length)ch(id,{type:'scatter',data:{datasets:ds},options:{plugins:{legend:LEG},scales:{x:AX(ax),y:AX('MWh')}}});}
 sc('cx1','temp','C');if(HAS.ghi)sc('cx2','ghi','GHI W/m2');if(HAS.cloud)sc('cx3','cloud','Bulut %');if(HAS.wind)sc('cx4','wind','Ruzgar');};

// ═══ TAB 5: ONERILER (hava-kosullu) ═══
T.rec=()=>{let h='<h2>Oneriler</h2><p class="lead">Her saat: modelin tahmini vs gecmis havaya gore beklenen yuk (P95 aralik)</p>';
 if(REC&&REC.some(r=>r.exp!=null))h+='<div class="panel"><h3>Model tahmini vs beklenen (hava-kosullu) + P95 bandi</h3><div class="chartbox tall"><canvas id="cr1"></canvas></div></div>';
 h+='<div id="recList"></div>';return h;};
g._i.rec=()=>{
 if(!REC)return;
 const hasExp=REC.some(r=>r.exp!=null);
 if(hasExp)ch('cr1',{type:'line',data:{labels:hrs,datasets:[
   {label:'MODEL',data:REC.map(r=>r.fc),borderColor:'#E53935',borderWidth:3,pointRadius:3},
   {label:'Beklenen (hava)',data:REC.map(r=>r.exp),borderColor:'#5ad1a0',borderWidth:2,pointRadius:2,borderDash:[5,3]},
   {label:'P95 ust',data:REC.map(r=>r.hi),borderColor:'transparent',backgroundColor:'rgba(90,209,160,.10)',pointRadius:0,fill:'+1'},
   {label:'P95 alt',data:REC.map(r=>r.lo),borderColor:'transparent',pointRadius:0,fill:false}
  ]},options:{plugins:{legend:LEG},scales:{x:AX(),y:AX('MWh')}}});
 // oneri kartlari: en cok sapan saatler
 let flagged=REC.filter(r=>r.exp!=null&&r.fc!=null).map(r=>{
   const md=r.lastweek!=null?r.fc-r.lastweek:null, ed=r.lastweek!=null?r.exp-r.lastweek:null;
   const out=(r.hi!=null&&r.lo!=null)&&(r.fc>r.hi||r.fc<r.lo);
   const gap=Math.abs(r.fc-r.exp);
   return {...r,md,ed,out,gap};
 }).sort((a,b)=>b.gap-a.gap);
 let top=flagged.filter(r=>r.out||r.gap>15).slice(0,8);
 let html='';
 if(!top.length)html='<div class="rm ok"><h4>Model hava beklentisiyle uyumlu</h4><p>Hicbir saatte model tahmini P95 bandi disinda degil; belirgin sapma yok.</p></div>';
 top.forEach(r=>{const cls=r.out?'danger':'warn';
   html+='<div class="rm '+cls+'"><h4>Saat '+r.h+':00 '+(r.out?'<span class="pill" style="background:#3a1f1f;color:#ff9b9b">P95 DISI</span>':'<span class="pill" style="background:#3a331f;color:#ffce6a">SAPMA</span>')+'</h4>';
   html+='<div class="meta"><span>Model: <b>'+fx(r.fc)+' MWh</b></span><span>Beklenen: <b>'+fx(r.exp)+' MWh</b></span>';
   if(r.lo!=null)html+='<span>P95: '+fx(r.lo)+' &ndash; '+fx(r.hi)+' MWh</span>';html+='<span>n='+r.n+'</span></div>';
   let s='Saat '+r.h+':00 icin model <b>'+fx(r.fc)+' MWh</b> ongoruyor';
   if(r.md!=null)s+=' (gecen haftaya gore '+(r.md>0?'+':'')+fx(r.md)+' MWh)';
   s+='. Gecmiste bu saat ve mevsimde ~'+fx(r.temp_fc,1)+'C sicaklikta beklenen yuk <b>'+fx(r.exp)+' MWh</b>';
   if(r.ed!=null)s+=' (yani ~'+(r.ed>0?'+':'')+fx(r.ed)+' MWh degisim)';
   s+='. ';
   if(r.out)s+='Model tahmini <b>%95 tahmin araligi ('+fx(r.lo)+'-'+fx(r.hi)+' MWh) DISINDA</b> &mdash; '+(r.fc>r.hi?'asiri yuksek, dusurulmesi':'asiri dusuk, yukseltilmesi')+' degerlendirilebilir.';
   else s+='Fark P95 icinde ama dikkat: model ile hava-beklentisi arasinda ~'+fx(r.gap)+' MWh acik var.';
   html+='<p>'+s+'</p></div>';});
 document.getElementById('recList').innerHTML=html;};

// ═══ TAB 6: DRIFT (D -> D+2) ═══
T.drift=()=>{let h='<h2>D &rarr; D+2 Kayma</h2><p class="lead">Sicaklik ve tuketim benzerliginin son donemdeki degisimi</p>';
 h+='<div class="grid2"><div class="panel"><h3>Saatlik sicaklik: tahmin vs gecmis</h3><div class="chartbox"><canvas id="cd1"></canvas></div></div>';
 h+='<div class="panel"><h3>Gunluk ort. sicaklik trendi (son 2 hafta)</h3><div class="chartbox"><canvas id="cd2"></canvas></div></div></div>';
 if(DRIFT.sim&&DRIFT.sim.length){h+='<div class="panel"><h3>Tuketim benzerligi: D+2 tahmini vs referans gunler</h3><table class="dt"><tr><th>Referans</th><th>MAPE %</th><th>Sekil korelasyon</th><th>Yorum</th></tr>';
  DRIFT.sim.forEach(s=>{const yor=s.mape<3?'cok benzer':s.mape<7?'benzer':'farkli';h+='<tr><td>'+s.lbl+'</td><td style="color:'+(s.mape<3?'#5ad1a0':s.mape<7?'#ffce6a':'#ff6b6b')+'">'+fx(s.mape,1)+'</td><td>'+fx(s.corr,2)+'</td><td style="color:#8b97b3">'+yor+'</td></tr>';});
  h+='</table></div>';}
 if(DRIFT.hourly_dtemp_slope!=null)h+='<div class="rm"><h4>Saatler-arasi degisim (son 7 gun)</h4><p>Bir saatten digerine sicaklik +1C degistiginde tuketim ~<b>'+fx(DRIFT.hourly_dtemp_slope)+' MW</b> yonunde hareket etti.</p></div>';
 return h;};
g._i.drift=()=>{
 let ds=[];if(DRIFT.temp_fc)ds.push({label:'D+2 tahmin',data:DRIFT.temp_fc,borderColor:'#E53935',borderWidth:3,pointRadius:2});
 if(DRIFT.temp_lastweek)ds.push({label:'1 hafta once',data:DRIFT.temp_lastweek,borderColor:'#1E88E5',borderWidth:1.5,pointRadius:0,borderDash:[6,3]});
 if(DRIFT.temp_2week)ds.push({label:'2 hafta once',data:DRIFT.temp_2week,borderColor:'#43A047',borderWidth:1.5,pointRadius:0,borderDash:[6,3]});
 if(ds.length)ch('cd1',{type:'line',data:{labels:hrs,datasets:ds},options:{plugins:{legend:LEG},scales:{x:AX(),y:AX('C')}}});
 if(DRIFT.daily_temp&&DRIFT.daily_temp.length)ch('cd2',{type:'line',data:{labels:DRIFT.daily_temp.map(x=>x[0].slice(5)),datasets:[{label:'Ort C',data:DRIFT.daily_temp.map(x=>x[1]),borderColor:'#ff9f5a',borderWidth:2,pointRadius:3,tension:.3}]},options:{plugins:{legend:{display:false}},scales:{x:AX(),y:AX('C')}}});};

// ═══ TAB 7: OZEL GUNLER ═══
T.ozel=()=>{let h='<h2>Ozel Gun Etkileri</h2><p class="lead">Cuma namazi, hafta sonu ve tatil kirilimlari</p>';
 if(SPECIAL.target_is_special)h+='<div class="rm" style="border-left-color:#8E24AA"><h4>D+2 ozel gun: '+SPECIAL.target_is_special+'</h4><p>Tahmin gunu ozel gune denk geliyor &mdash; tatil/arife yuk dususu goz onunde bulundurulmali.</p></div>';
 if(SPECIAL.friday){h+='<div class="panel"><h3>Cuma namazi etkisi (12:00-13:00)</h3><p class="note">Hafta ici (Sal-Per) ayni saatlere gore fark</p>';
  h+='<p style="font-size:13px">Ortalama: <b style="color:'+(SPECIAL.friday.mw<0?'#5ad1a0':'#ff9f5a')+'">'+fx(SPECIAL.friday.mw)+' MW ('+fx(SPECIAL.friday.pct,1)+'%)</b></p>';
  if(SPECIAL.friday_by_temp&&SPECIAL.friday_by_temp.length){h+='<table class="dt"><tr><th>Sicaklik araligi</th><th>Etki (MW)</th><th>n</th></tr>';
   SPECIAL.friday_by_temp.forEach(r=>h+='<tr><td>'+r.range+'</td><td style="color:'+(r.mw<0?'#5ad1a0':'#ff9f5a')+'">'+fx(r.mw)+'</td><td>'+r.n+'</td></tr>');h+='</table>';}
  h+='</div>';}
 h+='<div class="panel"><h3>Pazar gunu saatlik etki (haftaici %)</h3><div class="chartbox"><canvas id="co1"></canvas></div></div>';
 return h;};
g._i.ozel=()=>{
 if(SPECIAL.sunday_pct){const v=hrs.map(h=>SPECIAL.sunday_pct[h]!=null?SPECIAL.sunday_pct[h]:null);
  ch('co1',{type:'bar',data:{labels:hrs.map(h=>h+':00'),datasets:[{label:'Pazar vs haftaici %',data:v,backgroundColor:v.map(x=>x==null?'#33405e':x<0?'rgba(90,209,160,.75)':'rgba(255,159,90,.75)')}]},options:{plugins:{legend:{display:false}},scales:{y:AX('%'),x:AX()}}});}};

// ═══ BOOT ═══
const NAV=[['ozet','Ozet'],['karsi','Karsilastirma'],['sens','Sicaklik & Duyarlilik'],['cross','Cross Check'],['rec','Oneriler'],['drift','D -> D+2 Kayma'],['ozel','Ozel Gunler']];
document.getElementById('nv').innerHTML=NAV.map(x=>'<button data-t="'+x[0]+'">'+x[1]+'</button>').join('');
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>g(b.dataset.t));
NAV.forEach(x=>{const d=document.createElement('div');d.className='tab';d.id='t_'+x[0];d.innerHTML=T[x[0]]();document.getElementById('mn').appendChild(d);});
g(location.hash&&document.getElementById('t_'+location.hash.slice(1))?location.hash.slice(1):'ozet');
</script></body></html>"""
