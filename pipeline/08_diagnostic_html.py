"""
08_diagnostic_html.py — FULL Diagnostic HTML (6 tabs, Chart.js)
Tab 1: Ozet (FC, MAPE, weather)
Tab 2: Karsilastirma (5 gun load profile + P95)
Tab 3: Sicaklik Etkisi (hourly temp sensitivity + day-of-week)
Tab 4: Sensitivity + Scenario (seasonal + interactive slider)
Tab 5: Cross Check (temp×load, GHI×load, cloud×load scatter)
Tab 6: Oneriler (auto-recommendations with P95 intervals)
"""
import sys, json, os, re
import pandas as pd, numpy as np
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C

TEMP_COL = "MUGLA_MenteseCenter_app_temp_actual"
GHI_COL = "GHI_ADM_Weighted"
PREFIXES = ["", "ADM_"]
EDAS = "ADM"
TITLE = "STLF DIAGNOSTIC"

# ─── Find latest forecast ──────────────────────────────────
fc_files = sorted(set().union(*[set(C.OUTPUT_DIR.glob(f"*_{p}forecast.xlsx")) for p in PREFIXES]))
fc_files = [f for f in fc_files if '_REGEN' not in f.name]
fc_date = None
for p in reversed(fc_files):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', p.name); 
    if m: fc_date = m.group(1); break
if not fc_date: fc_date = str(date.today())
TODAY = date.fromisoformat(fc_date)
print(f"[{EDAS}] Diagnostic: {fc_date}")

# ─── DATA LOADING ─────────────────────────────────────────
m = pd.read_parquet(C.MASTER_PARQUET)
m[C.RAW_DATE_COL] = pd.to_datetime(m[C.RAW_DATE_COL])
m['dt'] = m[C.RAW_DATE_COL].dt.normalize()
m['h'] = m[C.RAW_HOUR_COL]

wh = pd.read_parquet(C.WEATHER_HISTORY_PARQUET)

# Drop stale wx columns from master before merge (only actual weather cols)
wx_in_m = [c for c in m.columns if c in wh.columns and c not in (C.RAW_DATE_COL, C.RAW_HOUR_COL)]

wh['dt'] = pd.to_datetime(wh[C.RAW_DATE_COL]).dt.normalize()
wh['h'] = wh[C.RAW_HOUR_COL]

if wx_in_m: m = m.drop(columns=wx_in_m)

cloud_cols = [c for c in wh.columns if 'cloud_actual' in c]

merge_cols = [C.RAW_DATE_COL, C.RAW_HOUR_COL, TEMP_COL, GHI_COL] + cloud_cols[:2]
merged = m.merge(wh[merge_cols], on=[C.RAW_DATE_COL, C.RAW_HOUR_COL], how='left')
cloud_c = cloud_cols[0] if cloud_cols else None
print(f"  Merged: {len(merged)} rows")

# ─── FC ──────────────────────────────────────────────────
fc = None
for p in PREFIXES:
    fp = C.OUTPUT_DIR / f"{fc_date}_{p}forecast.xlsx"
    if fp.exists():
        try:
            d = pd.read_excel(fp, sheet_name="Tahmin")
            fc = d.set_index("Saat")["Tahmin_MWh"]
            break
        except Exception as e: print(f"  FC uyari: {e}")

# ─── GET SERIES ───────────────────────────────────────────
def gs(ds):
    d = pd.Timestamp(ds).date()
    day = merged[merged['dt'].dt.date == d]
    if len(day) != 24: return None
    day = day.set_index('h').sort_index()
    r = {"load": day[C.RAW_TARGET_COL].values.tolist(),
         "temp": day[TEMP_COL].values.tolist() if TEMP_COL in day else None}
    if GHI_COL in day: r["ghi"] = day[GHI_COL].values.tolist()
    if cloud_c and cloud_c in day: r["cloud"] = day[cloud_c].values.tolist()
    day_type = day.get('day_type', None) if 'day_type' in day else None
    if day_type is not None: r["day_type"] = day_type.values.tolist() if hasattr(day_type, 'values') else None
    return r

# ─── COMPARISON DAYS ─────────────────────────────────────
cp = {}
main_labels = []
if fc is not None:
    pairs = [
        (7, "1 hafta once"), (14, "2 hafta once"),
        (364, "1 yil once"), (371, "1 yil+1h once"),
    ]
    for off, lbl in pairs:
        s = gs(str(TODAY - timedelta(days=off)))
        if s and s["load"] and len([x for x in s["load"] if x is not None and not np.isnan(x)]) > 20:
            cp[lbl] = s
            main_labels.append(lbl)

# ─── P95 ─────────────────────────────────────────────────
p95 = {}
for h in range(24):
    seg = merged[merged['h'] == h].dropna(subset=[TEMP_COL, C.RAW_TARGET_COL])
    if len(seg) < 30: continue
    err = seg[C.RAW_TARGET_COL].values - seg[C.RAW_TARGET_COL].shift(168).values
    err = err[~np.isnan(err)]
    if len(err) > 50:
        ml = np.mean(seg[C.RAW_TARGET_COL].values)
        p95[h] = {"p5_ape": float(np.percentile(np.abs(err), 5) / ml * 100),
                  "p95_ape": float(np.percentile(np.abs(err), 95) / ml * 100),
                  "p50_ape": float(np.median(np.abs(err)) / ml * 100),
                  "p5_err": float(np.percentile(err, 5)),
                  "p95_err": float(np.percentile(err, 95))}

# ─── LAST 7 MAPE ─────────────────────────────────────────
last7 = []
for i in range(1, 15):
    ds = str(TODAY - timedelta(days=i))
    for p in PREFIXES:
        fp = C.OUTPUT_DIR / f"{ds}_{p}forecast.xlsx"
        if fp.exists():
            try:
                d = pd.read_excel(fp, sheet_name="Tahmin").set_index("Saat")["Tahmin_MWh"]
                s = gs(ds)
                if not s: continue
                av = np.array(s["load"]); fv = np.array([d.get(h, np.nan) for h in range(24)])
                msk = av > 0
                if msk.sum() > 12:
                    last7.append([ds, round(float(np.mean(np.abs((av[msk]-fv[msk])/av[msk]))*100), 2)])
            except: pass
            break

# ─── SENSITIVITY ─────────────────────────────────────────
sn = {}
seg = merged.dropna(subset=[TEMP_COL, C.RAW_TARGET_COL])
if len(seg) > 100:
    seg['month'] = seg['dt'].dt.month
    seg['dow'] = seg['dt'].dt.dayofweek
    # Seasonal
    for nm, ms in [("Kis", [12,1,2]), ("Ilkbahar", [3,4,5]), ("Yaz", [6,7,8]), ("Sonbahar", [9,10,11])]:
        ss = seg[seg['month'].isin(ms)]
        if len(ss) > 50 and np.std(ss[TEMP_COL].values) > 5:
            slope, _ = np.polyfit(ss[TEMP_COL].values, ss[C.RAW_TARGET_COL].values, 1)
            sn[nm] = round(float(slope), 1)
    # By hour group
    for hg, hrange in [("Gece (00-06)", range(0,7)), ("Sabah (07-10)", range(7,11)),
                        ("Ogle (11-16)", range(11,17)), ("Aksam (17-23)", range(17,24))]:
        ss = seg[seg['h'].isin(hrange)]
        if len(ss) > 50:
            slope, _ = np.polyfit(ss[TEMP_COL].values, ss[C.RAW_TARGET_COL].values, 1)
            sn[hg] = round(float(slope), 1)
    # By day type
    for dt_label, dt_mask in [("Haftaici", seg['dow'].isin([0,1,2,3,4])),
                               ("Cumartesi", seg['dow'] == 5),
                               ("Pazar", seg['dow'] == 6)]:
        ss = seg[dt_mask]
        if len(ss) > 30:
            slope, _ = np.polyfit(ss[TEMP_COL].values, ss[C.RAW_TARGET_COL].values, 1)
            sn[dt_label] = round(float(slope), 1)

# ─── HOURLY TEMP EFFECT (last 7 days) ────────────────────
hourly_temp_effect = {}
for h in range(24):
    seg_h = merged[merged['h'] == h].tail(7*24).dropna(subset=[TEMP_COL, C.RAW_TARGET_COL])
    if len(seg_h) > 20:
        slope, _ = np.polyfit(seg_h[TEMP_COL].values, seg_h[C.RAW_TARGET_COL].values, 1)
        r2 = np.corrcoef(seg_h[TEMP_COL].values, seg_h[C.RAW_TARGET_COL].values)[0,1]**2
        hourly_temp_effect[h] = {"slope": round(float(slope), 1), "r2": round(float(r2), 3), "n": len(seg_h)}

# ─── SPECIAL HOUR EFFECTS (Friday prayer) ────────────────
friday_effect = None
friday_noon = merged[(merged['dt'].dt.dayofweek == 4) & (merged['h'].isin([12, 13]))]
if len(friday_noon) > 50:
    seg_ref = merged[(merged['dt'].dt.dayofweek.isin([0,1,2,3])) & (merged['h'].isin([12, 13]))]
    if len(seg_ref) > 100:
        friday_mean = friday_noon[C.RAW_TARGET_COL].mean()
        weekday_mean = seg_ref[C.RAW_TARGET_COL].mean()
        diff_mw = round(friday_mean - weekday_mean, 0)
        diff_pct = round((friday_mean - weekday_mean) / weekday_mean * 100, 1)
        friday_effect = {"mw": diff_mw, "pct": diff_pct}

# ─── WEATHER FC ──────────────────────────────────────────
fc_wx = {}
try:
    wfc = pd.read_parquet(C.DATA_DIR / "weather_cache" / "weather_fc_live.parquet")
    wfc[C.RAW_DATE_COL] = pd.to_datetime(wfc["Tarih"]).dt.normalize()
    today_wfc = wfc[wfc[C.RAW_DATE_COL].dt.date == TODAY]
    if len(today_wfc) > 12:
        today_wfc = today_wfc.set_index(C.RAW_HOUR_COL).sort_index()
        fc_temp_col = TEMP_COL.replace("_actual", "_fc")
        fc_ghi_col = GHI_COL.replace("_actual", "_fc") if "_actual" in GHI_COL else GHI_COL
        if fc_temp_col in today_wfc.columns: fc_wx["temp"] = today_wfc[fc_temp_col].values.tolist()
        if fc_ghi_col in today_wfc.columns: fc_wx["ghi"] = today_wfc[fc_ghi_col].values.tolist()
        if fc_wx: print(f"  WX FC: {len(today_wfc)} saat")
except Exception as e: print(f"  WX FC uyari: {e}")

# ─── BUILD HTML ──────────────────────────────────────────
FC = fc.values.tolist() if fc is not None else None
FCD = fc_date
JSData = {
    "D": FCD, "FC": FC, "WX": fc_wx if fc_wx else None,
    "CP": cp, "SN": sn, "L7": last7, "P95": p95,
    "P95K": [int(k) for k in p95.keys()], "HTE": hourly_temp_effect,
    "FRIDAY": friday_effect,
}

# Generate the HTML with embedded JS data
H = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{TITLE}</title><script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#0b0e14;color:#dfe6f3;font-family:'Segoe UI',sans-serif;font-size:13.5px}}
.app{{display:flex;min-height:100vh}}
nav{{width:175px;background:linear-gradient(180deg,#0d111a,#0b0e14);border-right:1px solid #26304a;padding:12px 0;position:sticky;top:0;height:100vh;overflow:auto;flex-shrink:0}}
nav h1{{font-size:12px;color:#5ad1a0;margin:4px 12px 8px;font-weight:700}}
nav button{{display:block;width:100%;text-align:left;background:none;border:0;color:#8b97b3;padding:8px 12px;cursor:pointer;font-size:12px;border-left:3px solid transparent}}
nav button:hover{{color:#dfe6f3;background:#11151f}}
nav button.on{{color:#dfe6f3;background:#11151f;border-left-color:#5ad1a0;font-weight:600}}
main{{flex:1;padding:18px 22px 50px;max-width:1150px}}
.tab{{display:none;animation:fade .2s}}.tab.on{{display:block}}
@keyframes fade{{from{{opacity:0}}to{{opacity:1}}}}
h2{{font-size:17px;margin:0 0 3px;font-weight:700}}
.lead{{color:#8b97b3;margin:0 0 12px;font-size:12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:8px;margin-bottom:14px}}
.card{{background:#131823;border:1px solid #26304a;border-radius:9px;padding:10px 12px}}
.card .lab{{font-size:10px;color:#8b97b3;text-transform:uppercase}}
.card .val{{font-size:20px;font-weight:700;margin-top:2px}}
.chartbox{{position:relative;height:250px;margin-bottom:14px}}
.chartbox.tall{{height:320px}}
.panel{{background:#131823;border:1px solid #26304a;border-radius:10px;padding:12px 14px;margin-bottom:12px}}
.panel h3{{margin:0 0 2px;font-size:13px;color:#c7d0e3}}
.panel .note{{color:#8b97b3;font-size:11px;margin:0 0 6px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
.rm{{padding:12px 14px;margin-bottom:9px;background:#131823;border:1px solid #26304a;border-left:4px solid #5ad1a0;border-radius:8px}}
.rm.warn{{border-left-color:#ffce6a}}.rm.danger{{border-left-color:#ff6b6b}}
.rm h4{{margin:0 0 2px;font-size:13px}}.rm p{{margin:3px 0;font-size:12px;color:#c7d0e3;line-height:1.5}}
.rm .meta{{display:flex;gap:10px;font-size:10px;color:#8b97b3;margin:4px 0}}
input[type=range]{{width:100%}}
.slider-val{{display:inline-block;background:#26304a;padding:2px 8px;border-radius:4px;font-weight:700}}
@media(max-width:760px){{.grid2,.grid3{{grid-template-columns:1fr}}}}
</style></head><body><div class="app"><nav>
<h1>STLF</h1><div style="font-size:10px;color:#8b97b3;margin:-4px 12px 8px">{FCD}</div>
<div id="nv"></div></nav><main id="mn"></main></div>
<script>
const D="{FCD}";
const FC={json.dumps(FC)};
const WX={json.dumps(fc_wx if fc_wx else None)};
const CP={json.dumps(cp)};
const SN={json.dumps(sn)};
const L7={json.dumps(last7)};
const P95={json.dumps(p95)};
const P95K={json.dumps([int(k) for k in p95.keys()])};
const HTE={json.dumps(hourly_temp_effect)};
const FRI={json.dumps(friday_effect)};
const hrs=Array.from({{length:24}},(_,i)=>i);
const pal=['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA'];
function ch(i,c){{const x=document.getElementById(i);if(x)new Chart(x,c);}}
function g(id){{
 document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===id));
 document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.id==='t_'+id));
 if(g._i&&g._i[id])g._i[id]();location.hash=id;}}
g._i={{}};
const T={{}};

// ═══════════ TAB 1: OZET ═══════════
T.ozet=()=>{{
 let h='<h2>Ozet</h2><p class="lead">'+D+' tahmini</p><div class="cards">';
 if(FC){{const m=FC.reduce((a,b)=>a+b,0)/24;
 h+='<div class="card"><div class="lab">Ortalama</div><div class="val" style="color:#5ad1a0">'+m.toFixed(0)+'</div><div class="lab">MWh</div></div>';
 h+='<div class="card"><div class="lab">Pik</div><div class="val" style="color:#ffce6a">'+Math.max(...FC).toFixed(0)+'</div><div class="lab">s.'+FC.indexOf(Math.max(...FC))+':00</div></div>';}}
 if(P95K.length){{const m=Object.values(P95).reduce((a,b)=>a+b.p95_ape,0)/P95K.length;
 h+='<div class="card"><div class="lab">P95 MAPE</div><div class="val" style="color:#ff8a8a;font-size:17px">'+m.toFixed(1)+'%</div></div>';}}
 if(CP&&Object.keys(CP).length){{const k=Object.keys(CP)[0];const d=FC.map((v,i)=>v-CP[k].load[i]);const ape=d.map(Math.abs).reduce((a,b)=>a+b,0)/d.length/Math.abs(FC.reduce((a,b)=>a+b,0)/FC.length)*100;
 h+='<div class="card"><div class="lab">vs '+k+'</div><div class="val" style="color:'+(ape<3?'#5ad1a0':ape<7?'#ffce6a':'#ff6b6b')+'">'+ape.toFixed(1)+'%</div></div>';}}
 if(L7&&L7.length)h+='<div class="panel"><h3>Son 7 Gun MAPE</h3><div class="chartbox"><canvas id="c7"></canvas></div></div>';
 if(WX&&WX.temp)h+='<div class="panel"><h3>Tahmin Sicaklik & GHI</h3><div class="chartbox tall"><canvas id="cw"></canvas></div></div>';
 return h;}};
g._i.ozet=()=>{{
 if(L7&&L7.length)ch('c7',{{type:'bar',data:{{labels:L7.map(x=>x[0]),datasets:[{{label:'MAPE%',data:L7.map(x=>x[1]),backgroundColor:'rgba(90,209,160,.7)'}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{y:{{beginAtZero:true,ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}},x:{{ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}}}}}}}});
 if(WX&&WX.temp&&WX.ghi)ch('cw',{{type:'line',data:{{labels:hrs,datasets:[{{label:'Sicaklik C',data:WX.temp,borderColor:'#ff9f5a',borderWidth:2,pointRadius:2,yAxisID:'y'}},{{label:'GHI W/m2',data:WX.ghi,borderColor:'#ffce6a',borderWidth:1.5,pointRadius:1,yAxisID:'y2'}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:'#8b97b3',boxWidth:10,font:{{size:9}}}}}}}},scales:{{x:{{ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}},y:{{position:'left',title:{{display:true,text:'C',color:'#8b97b3'}},ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}},y2:{{position:'right',title:{{display:true,text:'W/m2',color:'#8b97b3'}},ticks:{{color:'#8b97b3'}},grid:{{drawOnChartArea:false}}}}}}}}}});}};

// ═══════════ TAB 2: KARSILASTIRMA ═══════════
T.karsi=()=>{{
 const k=Object.keys(CP);if(!k.length||!FC)return '<h2>Karsilastirma</h2><p class="lead">Veri yok</p>';
 let h='<h2>D+2 Karsilastirma &mdash; '+D+'</h2><p class="lead">5 gun: '+k.join(', ')+'</p>';
 h+='<div class="panel"><h3>Yuk Profili + P95 Bandi</h3><div class="chartbox tall"><canvas id="c1"></canvas></div></div>';
 h+='<div class="grid2"><div class="panel"><h3>Normalize Profil</h3><div class="chartbox"><canvas id="c2"></canvas></div></div>';
 h+='<div class="panel"><h3>Sicaklik Karsilastirmasi</h3><div class="chartbox"><canvas id="c3"></canvas></div></div></div>';
 return h;}};
g._i.karsi=()=>{{
 const k=Object.keys(CP);if(!k.length||!FC)return;
 let ds1=[{{label:'TAHMIN '+D,data:FC,borderColor:'#E53935',borderWidth:3,pointRadius:4}}];
 if(P95K.length){{const up=hrs.map(h=>P95[h]?FC[h]+P95[h].p95_err:null),lo=hrs.map(h=>P95[h]?FC[h]+P95[h].p5_err:null);
 ds1.push({{label:'P95 ust',data:up,backgroundColor:'rgba(229,57,53,.12)',borderColor:'transparent',pointRadius:0,fill:'-2'}},
 {{label:'_',data:lo,borderColor:'transparent',pointRadius:0,fill:false}});}}
 k.forEach((l,i)=>ds1.push({{label:l,data:CP[l].load,borderColor:pal[(i+1)%5],borderWidth:1.5,pointRadius:1,borderDash:[6,3]}}));
 ch('c1',{{type:'line',data:{{labels:hrs,datasets:ds1}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:'#8b97b3',boxWidth:10,font:{{size:9}}}}}}}},scales:{{x:{{ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}},y:{{title:{{display:true,text:'MWh',color:'#8b97b3'}},ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}}}}}}}});
 let fm=FC.reduce((a,b)=>a+b,0)/24,ds2=[{{label:'TAHMIN '+D,data:FC.map(v=>v/fm),borderColor:'#E53935',borderWidth:3,pointRadius:4}}];
 k.forEach((l,i)=>{{let m=CP[l].load.reduce((a,b)=>a+b,0)/24;ds2.push({{label:l,data:CP[l].load.map(v=>v/m),borderColor:pal[(i+1)%5],borderWidth:1.5,pointRadius:1,borderDash:[6,3]}});}});
 ch('c2',{{type:'line',data:{{labels:hrs,datasets:ds2}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:'#8b97b3',boxWidth:10,font:{{size:9}}}}}}}},scales:{{x:{{ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}},y:{{title:{{display:true,text:'Yuk/Ort',color:'#8b97b3'}},ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}}}}}}}});
 let ds3=[];if(WX&&WX.temp)ds3.push({{label:'TAHMIN '+D,data:WX.temp,borderColor:'#E53935',borderWidth:2,pointRadius:2}});
 k.forEach((l,i)=>{{if(CP[l].temp)ds3.push({{label:l,data:CP[l].temp,borderColor:pal[(i+1)%5],borderWidth:1.5,pointRadius:0,borderDash:[6,3]}});}});
 if(ds3.length)ch('c3',{{type:'line',data:{{labels:hrs,datasets:ds3}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:'#8b97b3',boxWidth:10,font:{{size:9}}}}}}}},scales:{{x:{{ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}},y:{{title:{{display:true,text:'C',color:'#8b97b3'}},ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}}}}}}}});}};

// ═══════════ TAB 3: SICAKLIK ETKISI ═══════════
T.sicaklik=()=>{{
 let h='<h2>Saatlik Sicaklik Etkisi</h2><p class="lead">Her saat icin: +1C -> kac MW degisir? (son 7 gun)</p>';
 if(Object.keys(HTE).length){{
 h+='<div class="panel"><h3>Saatlik MW/C (son 7 gun)</h3><div class="chartbox"><canvas id="cs1"></canvas></div></div>';
 h+='<div class="panel"><h3>Etki Tablosu</h3><table style="width:100%;font-size:11px;color:#c7d0e3;border-collapse:collapse">';
 h+='<tr style="color:#8b97b3">'+hrs.map(h=>'<th>'+h+':00</th>').join('')+'</tr><tr>';
 hrs.forEach(h=>{{let v=HTE[h]?HTE[h].slope:0;h+='<td style="text-align:center;color:'+(v>0?'#ff9f5a':'#5ad1a0')+'">'+(v>0?'+':'')+v+'</td>';}});
 h+='</tr></table></div>';
 }}
 if(FRI){{
 h+='<div class="panel"><h3>Ozel Saat Etkisi</h3><div style="color:#c7d0e3;font-size:12px">';
 h+='<b>Cuma Namazi (12:00-13:00):</b> '+FRI.mw+' MW ('+FRI.pct+'%)<br>';
 h+='<span style="font-size:10px;color:#8b97b3">Haftaici ayni saatlere gore fark</span>';
 h+='</div></div>';
 }}
 return h;}};
g._i.sicaklik=()=>{{
 if(Object.keys(HTE).length)ch('cs1',{{type:'bar',data:{{labels:hrs.map(h=>h+':00'),datasets:[{{label:'MW/C',data:hrs.map(h=>HTE[h]?HTE[h].slope:0),backgroundColor:hrs.map(h=>HTE[h]&&HTE[h].slope>0?'rgba(255,159,90,.7)':'rgba(90,209,160,.7)')}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{y:{{title:{{display:true,text:'MW/C',color:'#8b97b3'}},ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}},x:{{ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}}}}}}}});}};

// ═══════════ TAB 4: SENSITIVITY + SCENARIO ═══════════
T.sens=()=>{{
 const e=Object.entries(SN);if(!e.length)return '<h2>Sensitivity</h2><p class="lead">Veri yok</p>';
 let h='<h2>Sensitivity & Scenario</h2><p class="lead">Her +1C sicaklik -> kac MW yuk degisimi</p>';
 h+='<div class="panel"><h3>Sezonsal + Gun Tipi Sensitivity (MW/C)</h3><div class="chartbox"><canvas id="cs2"></canvas></div></div>';
 h+='<div class="panel"><h3>Interaktif Senaryo Motoru</h3><p class="note">Sicaklik sapmasina gore beklenen yuk degisimi</p>';
 h+='<div style="margin:10px 0"><label style="color:#8b97b3;font-size:12px">Sicaklik Sapmasi: <span id="sval" class="slider-val">0C</span></label><br>';
 h+='<input type="range" id="srange" min="-5" max="5" value="0" step="1" oninput="var e=document.getElementById(\\'sval\\');if(e)e.textContent=(this.value>0?\\'+\\':\\'\\')+this.value+\\'C\\';updateScenario(+this.value);" style="margin:8px 0"></div>';
 h+='<div id="scenOut" style="color:#c7d0e3;font-size:12px"></div></div>';
 return h;}};
g._i.sens=()=>{{
 const e=Object.entries(SN);if(e.length)ch('cs2',{{type:'bar',data:{{labels:e.map(x=>x[0]),datasets:[{{label:'MW/C',data:e.map(x=>x[1]),backgroundColor:['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1','#FF7043','#AB47BC']}}]}},options:{{responsive:true,maintainAspectRatio:false,indexAxis:'y',plugins:{{legend:{{display:false}}}},scales:{{x:{{title:{{display:true,text:'MW / C',color:'#8b97b3'}},ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}},y:{{ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}}}}}}}});
 window.updateScenario=function(delta){{
  let o='';const yaz=SN['Yaz']||0;
  if(yaz){{const mw=Math.round(yaz*delta);o+='<b>Yaz sezonu:</b> '+delta+'C -> <b>'+mw+' MW</b> ('+(delta>0?'artis':'azalis')+')<br>';}}
  if(FC){{const newPeak=Math.round(Math.max(...FC)+yaz*delta);o+='<b>Yeni pik tahmini:</b> '+newPeak+' MWh<br>';}}
  document.getElementById('scenOut').innerHTML=o||'Veri yok';
 }};
 updateScenario(0);}};

// ═══════════ TAB 5: CROSS CHECK ═══════════
T.cross=()=>{{
 const k=Object.keys(CP);if(!k.length)return '<h2>Cross Check</h2><p class="lead">Veri yok</p>';
 return '<h2>Cross Check</h2><p class="lead">Sicaklik, GHI, Bulut &times; Yuk</p>'
 +'<div class="grid3"><div class="panel"><h3>Sicaklik &times; Yuk</h3><div class="chartbox"><canvas id="cx1"></canvas></div></div>'
 +'<div class="panel"><h3>GHI &times; Yuk</h3><div class="chartbox"><canvas id="cx2"></canvas></div></div>'
 +'<div class="panel"><h3>Bulut &times; Yuk</h3><div class="chartbox"><canvas id="cx3"></canvas></div></div></div>';}};
g._i.cross=()=>{{
 const k=Object.keys(CP);if(!k.length||!FC)return;
 let td=[],gd=[],cd=[];
 if(WX&&WX.temp)td.push({{label:'TAHMIN '+D,data:WX.temp.map((v,i)=>({{x:v,y:FC[i]}})),backgroundColor:'#E53935',pointRadius:5}});
 k.forEach((l,i)=>{{
  if(CP[l].temp)td.push({{label:l,data:CP[l].temp.map((v,j)=>({{x:v,y:CP[l].load[j]}})),backgroundColor:pal[(i+1)%5],pointRadius:3}});
  if(CP[l].ghi)gd.push({{label:l,data:CP[l].ghi.map((v,j)=>({{x:v,y:CP[l].load[j]}})),backgroundColor:pal[(i+1)%5],pointRadius:3}});
  if(CP[l].cloud)cd.push({{label:l,data:CP[l].cloud.map((v,j)=>({{x:v,y:CP[l].load[j]}})),backgroundColor:pal[(i+1)%5],pointRadius:3}});
 }});
 ch('cx1',{{type:'scatter',data:{{datasets:td}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:'#8b97b3',boxWidth:10,font:{{size:9}}}}}}}},scales:{{x:{{title:{{display:true,text:'C',color:'#8b97b3'}},ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}},y:{{title:{{display:true,text:'MWh',color:'#8b97b3'}},ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}}}}}}}});
 if(gd.length)ch('cx2',{{type:'scatter',data:{{datasets:gd}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:'#8b97b3',boxWidth:10,font:{{size:9}}}}}}}},scales:{{x:{{title:{{display:true,text:'GHI W/m2',color:'#8b97b3'}},ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}},y:{{title:{{display:true,text:'MWh',color:'#8b97b3'}},ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}}}}}}}});
 if(cd.length)ch('cx3',{{type:'scatter',data:{{datasets:cd}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:'#8b97b3',boxWidth:10,font:{{size:9}}}}}}}},scales:{{x:{{title:{{display:true,text:'Bulut %',color:'#8b97b3'}},ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}},y:{{title:{{display:true,text:'MWh',color:'#8b97b3'}},ticks:{{color:'#8b97b3'}},grid:{{color:'#1c2334'}}}}}}}}}});}};

// ═══════════ TAB 6: ONERILER ═══════════
T.rec=()=>{{
 let h='<h2>Oneriler</h2><p class="lead">Otomatik analiz ve tavsiyeler</p>';let c=0;
 if(FC&&Object.keys(CP).length){{
  const k=Object.keys(CP)[0],lw=CP[k];
  const diff=FC.map((v,i)=>v-lw.load[i]);const md=diff.reduce((a,b)=>a+b,0)/24;
  const ap=diff.map(Math.abs).reduce((a,b)=>a+b,0)/24/Math.abs(FC.reduce((a,b)=>a+b,0)/24)*100;
  const mx=diff.indexOf(Math.max(...diff)),mn=diff.indexOf(Math.min(...diff));c++;
  h+='<div class="rm"><h4>'+k+' Karsilastirmasi</h4><div class="meta">Guven: <b style="color:'+(ap<3?'#5ad1a0':ap<7?'#ffce6a':'#ff6b6b')+'">'+(ap<3?'YUKSEK':ap<7?'ORTA':'DUSUK')+'</b> | MAPE: <b>'+ap.toFixed(1)+'%</b></div>';
  h+='<p>Tahmin gecen haftaya gore <b>'+Math.abs(md).toFixed(0)+' MWh '+(md>0?'yuksek':'dusuk')+'</b>. En buyuk fark <b>'+mx+':00</b> ('+diff[mx].toFixed(0)+' MWh), en kucuk <b>'+mn+':00</b> ('+diff[mn].toFixed(0)+' MWh).</p></div>';
 }}
 if(FRI&&FC){{
  const fri_hr_diff=FC[12]-((CP[Object.keys(CP)[0]]||{{}}).load||[])[12]||0;c++;
  h+='<div class="rm warn"><h4>Cuma Namazi Etkisi (12:00)</h4><div class="meta">Tarihsel: '+FRI.mw+' MW ('+FRI.pct+'%)</div>';
  h+='<p>D+2 gununde saat 12:00 tahmini ile gecen hafta farki fark: <b>'+fri_hr_diff.toFixed(0)+' MWh</b>. Tarihsel cuma etkisi goz onune alindiginda ek +/- dusunulebilir.</p></div>';
 }}
 if(Object.keys(SN).length&&FC){{
  const yaz=SN['Yaz']||0,ogle=SN['Ogle (11-16)']||0;c++;
  h+='<div class="rm"><h4>Scenario Engine</h4><p>Yaz sezonu: +1C ~ <b>'+yaz+' MW</b> | Ogle saatleri: +1C ~ <b>'+ogle+' MW</b><br>';
  h+='Hava +2C sicak: yuk ~<b>'+Math.round(yaz*2)+' MW</b> artar. Pik: <b>'+Math.round(Math.max(...FC)+yaz*2)+' MWh</b><br>';
  h+='Hava -2C soguk: yuk ~<b>'+Math.round(yaz*2)+' MW</b> azalir.</p></div>';
 }}
 if(Object.values(HTE).length){{
  const pk=Object.entries(HTE).reduce((a,b)=>Math.abs(a[1].slope)>Math.abs(b[1].slope)?a:b);c++;
  h+='<div class="rm warn"><h4>Saatlik Sicaklik Hassasiyeti</h4><p>En hassas saat: <b>'+pk[0]+':00</b> (+1C -> '+pk[1].slope+' MW). En yuksek r<sup>2</sup>: '+pk[1].r2.toFixed(2)+'</p></div>';
 }}
 return c?h:'<h2>Oneriler</h2><p class="lead">Henuz yeterli veri yok.</p>';}};

// ═══════════ BOOT ═══════════
document.getElementById('nv').innerHTML=[['ozet','Ozet'],['karsi','Karsilastirma'],['sicaklik','Sicaklik Etkisi'],['sens','Sensitivity'],['cross','Cross Check'],['rec','Oneriler']].map(x=>'<button data-t="'+x[0]+'">'+x[1]+'</button>').join('');
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>g(b.dataset.t));
Object.entries(T).forEach(e=>{{const d=document.createElement('div');d.className='tab';d.id='t_'+e[0];d.innerHTML=e[1]();document.getElementById('mn').appendChild(d);}});
g(location.hash?location.hash.slice(1):'ozet');
</script></body></html>"""

OUT = C.OUTPUT_DIR / f"diagnostic_{fc_date}.html"
OUT.write_text(H, 'utf-8')
js_part = H[H.find("<script>"):H.find("</script>")]
o, c = js_part.count('{') + js_part.count('('), js_part.count('}') + js_part.count(')')
print(f"SAVED: {OUT} ({len(H)} bytes) JS: {'OK' if o == c else f'BRACE: {o}/{c}'}")
print(json.dumps({"status": "ok", "file": str(OUT)}))
