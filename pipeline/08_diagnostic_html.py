"""
08_diagnostic_html.py — Interaktif Chart.js Diagnostic (P95 + Cloud + Rec)
=========================================================================
Sekmeler: Ozet, D+2 Karsilastirma, P95 Analysis, Cross Check, Sensitivity, Oneriler
"""
import sys, json, os
import pandas as pd, numpy as np
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C

OUT = C.OUTPUT_DIR / f"diagnostic_{date.today().strftime('%Y-%m-%d')}.html"
TODAY = date.today()

# ─── LOAD ────────────────────────────────────────────────────────────
print("Loading ADM data...")
m = pd.read_parquet(C.MASTER_PARQUET)
m[C.RAW_DATE_COL] = pd.to_datetime(m[C.RAW_DATE_COL])
m["dt"] = m[C.RAW_DATE_COL].dt.normalize()
m["h"] = m[C.RAW_HOUR_COL]

wh = pd.read_parquet(C.WEATHER_HISTORY_PARQUET)
wh[C.RAW_DATE_COL] = pd.to_datetime(wh[C.RAW_DATE_COL]).dt.normalize()

# Drop stale wx columns from master
wx_in_m = [c for c in m.columns if c in wh.columns and c not in (C.RAW_DATE_COL, C.RAW_HOUR_COL)]
if wx_in_m: m = m.drop(columns=wx_in_m)

merge_cols = [C.RAW_DATE_COL, C.RAW_HOUR_COL, "MUGLA_MenteseCenter_app_temp_actual",
              "GHI_ADM_Weighted"]
# Add cloud cols
cloud_cols = [c for c in wh.columns if "cloud_actual" in c]
merge_cols += cloud_cols[:2]  # first 2 cloud stations

merged = m.merge(wh[merge_cols], on=[C.RAW_DATE_COL, C.RAW_HOUR_COL], how="left")
temp_c, ghi_c, cloud_c = "MUGLA_MenteseCenter_app_temp_actual", "GHI_ADM_Weighted", cloud_cols[0] if cloud_cols else None
print(f"Merged: {len(merged)} rows")

# ─── FORECAST ────────────────────────────────────────────────────────
fc, fc7, fc14 = None, None, None
for offset, name in [(0, "fc"), (7, "fc7"), (14, "fc14")]:
    ds = str(TODAY - timedelta(days=offset))
    p = C.OUTPUT_DIR / f"{ds}_forecast.xlsx"
    if p.exists():
        try:
            d = pd.read_excel(p, sheet_name="Tahmin")
            locals()[name] = d.set_index("Saat")["Tahmin_MWh"]
        except: pass

# ─── COMPARISON DAYS ─────────────────────────────────────────────────
comp_dates = []
if fc is not None:
    for offset, label in [(7,"1 hafta once"),(14,"2 hafta once"),
                           (364,"gecen yil"),(350,"gecen yil-2h"),
                           (371,"gecen yil+1h")]:
        cd = TODAY - timedelta(days=offset)
        comp_dates.append((str(cd), label))

def get_series(ds):
    d = pd.Timestamp(ds).date()
    day = merged[merged["dt"].dt.date == d]
    if len(day) != 24: return None
    day = day.set_index(C.RAW_HOUR_COL).sort_index()
    r = {"load": day[C.RAW_TARGET_COL].values.tolist(),
         "temp": day[temp_c].values.tolist() if temp_c in day.columns else None}
    if ghi_c in day.columns: r["ghi"] = day[ghi_c].values.tolist()
    if cloud_c and cloud_c in day.columns: r["cloud"] = day[cloud_c].values.tolist()
    return r

comp_series = {}
for ds, label in comp_dates:
    s = get_series(ds)
    if s: comp_series[label] = s

# ─── P95 PREDICTION INTERVAL (from backtest error distribution) ──────
print("Computing P95 intervals...")
# Read backtest files to get historical error distribution
p95_data = []
if fc is not None:
    for btd in [TODAY - timedelta(days=i) for i in range(1, 61)]:
        bp = C.OUTPUT_DIR / f"{btd}_models_REGEN.parquet"
        if not bp.exists(): continue
        try:
            bf = pd.read_parquet(bp)
            bf["dt"] = pd.to_datetime(bf["Datetime"])
            bt2 = bf[bf["dt"].dt.date == btd]
            act_day = m[m["dt"].dt.date == btd]
            if len(bt2) != 24 or len(act_day) != 24: continue
            for h in range(24):
                pv = bt2.iloc[h]["Final_Pred"]
                av = act_day.iloc[h][C.RAW_TARGET_COL]
                if av > 0 and not np.isnan(pv):
                    p95_data.append({"h": h, "ape": abs(av-pv)/av*100, "err": pv-av})
        except: pass

p95_hourly = {}
if p95_data:
    pdf = pd.DataFrame(p95_data)
    for h in range(24):
        seg = pdf[pdf["h"] == h]
        if len(seg) >= 10:
            p95_hourly[h] = {
                "p5_ape": float(np.percentile(seg["ape"], 5)),
                "p95_ape": float(np.percentile(seg["ape"], 95)),
                "p5_err": float(np.percentile(seg["err"], 5)),
                "p95_err": float(np.percentile(seg["err"], 95)),
                "p50_ape": float(np.percentile(seg["ape"], 50)),
            }
    print(f"  P95 data: {len(pdf)} samples from {pdf['h'].nunique()} hours")

# ─── SENSITIVITY ─────────────────────────────────────────────────────
print("Sensitivity...")
seasons_map = {12:"Kis",1:"Kis",2:"Kis",3:"Ilkbahar",4:"Ilkbahar",5:"Ilkbahar",
               6:"Yaz",7:"Yaz",8:"Yaz",9:"Sonbahar",10:"Sonbahar",11:"Sonbahar"}
merged["season"] = merged["dt"].dt.month.map(seasons_map)
merged["t_valid"] = merged[temp_c].notna() & (merged[C.RAW_TARGET_COL] > 0)
sensitivity = {}
for season in ["Kis","Ilkbahar","Yaz","Sonbahar"]:
    seg = merged[(merged["season"] == season) & merged["t_valid"]]
    if len(seg) < 500: continue
    lo, hi = np.percentile(seg[temp_c], 10), np.percentile(seg[temp_c], 90)
    seg2 = seg[(seg[temp_c] >= lo) & (seg[temp_c] <= hi)]
    if len(seg2) < 200: continue
    t, y = seg2[temp_c].values, seg2[C.RAW_TARGET_COL].values
    slope = np.cov(t, y)[0,1] / np.var(t) if np.var(t) > 0 else 0
    sensitivity[season] = round(slope, 1)

# ─── LAST 7 MAPE ─────────────────────────────────────────────────────
last7 = []
for i in range(1, 8):
    d = TODAY - timedelta(days=i); ds = str(d)
    p = C.OUTPUT_DIR / f"{ds}_forecast.xlsx"
    if not p.exists(): continue
    try: fcd = pd.read_excel(p, sheet_name="Tahmin").set_index("Saat")["Tahmin_MWh"]
    except: continue
    act = get_series(ds)
    if not act: continue
    av = np.array(act["load"]); fv = np.array([fcd.get(h, np.nan) for h in range(24)])
    msk = av > 0
    if msk.sum() < 12: continue
    mape = float(np.mean(np.abs((av[msk]-fv[msk])/av[msk])) * 100)
    last7.append((ds, round(mape, 2)))

# ─── TODAY WEATHER ───────────────────────────────────────────────────
fc_wx_dict = {}
today_wx = merged[merged["dt"].dt.date == TODAY]
if len(today_wx) == 24:
    tdx = today_wx.set_index(C.RAW_HOUR_COL).sort_index()
    if temp_c in tdx.columns: fc_wx_dict["temp"] = tdx[temp_c].values.tolist()
    if ghi_c in tdx.columns: fc_wx_dict["ghi"] = tdx[ghi_c].values.tolist()
    if cloud_c and cloud_c in tdx.columns: fc_wx_dict["cloud"] = tdx[cloud_c].values.tolist()

# ─── BUILD HTML (Embed ded tiny Chart.js) ────────────────────────────
def js(v):
    if v is None: return "null"
    if isinstance(v, list): return "[" + ",".join("null" if x is None else f"{x:.1f}" for x in v) + "]"
    return json.dumps(v)

H = """<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>STLF DIAGNOSTIC</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b0e14;color:#dfe6f3;font-family:'Segoe UI',sans-serif;font-size:14px}
.app{display:flex;min-height:100vh}
nav{width:170px;background:linear-gradient(180deg,#0d111a,#0b0e14);border-right:1px solid #26304a;padding:12px 0;position:sticky;top:0;height:100vh;overflow:auto;flex-shrink:0}
nav h1{font-size:12px;color:#5ad1a0;margin:4px 12px 8px;font-weight:700}
nav button{display:block;width:100%;text-align:left;background:none;border:0;color:#8b97b3;padding:7px 12px;cursor:pointer;font-size:11.5px;border-left:3px solid transparent}
nav button:hover{color:#dfe6f3;background:#11151f}
nav button.on{color:#dfe6f3;background:#11151f;border-left-color:#5ad1a0;font-weight:600}
main{flex:1;padding:18px 22px 50px;max-width:1100px}
.tab{display:none;animation:fade .2s}.tab.on{display:block}
@keyframes fade{from{opacity:0}to{opacity:1}}
h2{font-size:17px;margin:0 0 3px;font-weight:700}
.lead{color:#8b97b3;margin:0 0 12px;font-size:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:14px}
.card{background:#131823;border:1px solid #26304a;border-radius:9px;padding:10px 12px}
.card .lab{font-size:10px;color:#8b97b3;text-transform:uppercase}
.card .val{font-size:21px;font-weight:700;margin-top:2px}
.chartbox{position:relative;height:240px;margin-bottom:14px}
.chartbox.tall{height:300px}
.panel{background:#131823;border:1px solid #26304a;border-radius:10px;padding:12px 14px;margin-bottom:12px}
.panel h3{margin:0 0 2px;font-size:13px}
.panel .note{color:#8b97b3;font-size:11.5px;margin:0 0 6px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.rm{padding:11px 13px;margin-bottom:9px;background:#131823;border:1px solid #26304a;border-left:4px solid #5ad1a0;border-radius:8px}
.rm h4{margin:0 0 2px;font-size:13px}.rm p{margin:3px 0;font-size:11.5px;color:#c7d0e3;line-height:1.45}
.rm .meta{display:flex;gap:10px;font-size:10px;color:#8b97b3;margin:4px 0}
@media(max-width:760px){.grid2{grid-template-columns:1fr}}
</style></head><body><div class="app"><nav>
<h1>STLF DIAG</h1><div style="font-size:10px;color:#8b97b3;margin:-4px 12px 8px">""" + str(TODAY) + """</div>
<div id="nv"></div></nav><main id="mn"></main></div>
<script>
const D=""" + json.dumps(str(TODAY)) + """;
const FC=""" + json.dumps(fc.values.tolist() if fc is not None else None) + """;
const WX=""" + json.dumps(fc_wx_dict if fc_wx_dict else None) + """;
const CP=""" + json.dumps(comp_series) + """;
const SN=""" + json.dumps(sensitivity) + """;
const L7=""" + json.dumps(last7) + """;
const P95=""" + json.dumps(p95_hourly) + """;
const P95_KEYS=Object.keys(P95).length?Object.keys(P95).map(Number):[];
const pal=['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1'];
const hrs=Array.from({length:24},(_,i)=>i);

function ch(id,cfg){const c=document.getElementById(id);if(!c)return;return new Chart(c,cfg);}

function go(id){
 document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===id));
 document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.id==='tab_'+id));
 if(go._inits&&go._inits[id])go._inits[id]();
 location.hash=id;}
go._inits={};

const TABS={};

TABS.ozet=()=>{
 let h=`<h2>Ozet</h2><p class="lead">Son 7 gun MAPE + bugun tahmini.</p>
 <div class="cards"><div class="card"><div class="lab">Tahmin Gunu</div><div class="val" style="font-size:15px">${D}</div></div>`;
 if(FC){const m=FC.reduce((a,b)=>a+b,0)/FC.length;
 h+=`<div class="card"><div class="lab">Ort</div><div class="val" style="color:#5ad1a0">${m.toFixed(0)}</div><div class="lab">MWh</div></div>`;
 h+=`<div class="card"><div class="lab">Pik</div><div class="val" style="color:#ffce6a">${Math.max(...FC).toFixed(0)}</div><div class="lab">s.${FC.indexOf(Math.max(...FC))}:00</div></div>`;}
 if(P95_KEYS.length){const m=Object.values(P95).reduce((a,b)=>a+b.p95_ape,0)/P95_KEYS.length;
 h+=`<div class="card"><div class="lab">P95 MAPE ort</div><div class="val" style="color:#ff8a8a;font-size:18px">${m.toFixed(1)}%</div></div>`;}
 h+=`</div>`;
 if(L7&&L7.length)h+=`<div class="panel"><h3>Son 7 Gun MAPE</h3><div class="chartbox"><canvas id="c7"></canvas></div></div>`;
 if(WX&&WX.temp)h+=`<div class="panel"><h3>Tahmin Sicaklik</h3><div class="chartbox tall"><canvas id="c_wx"></canvas></div></div>`;
 return h;};
TABS.ozet_init=()=>{
 if(L7&&L7.length)ch('c7',{type:'bar',data:{labels:L7.map(x=>x[0]),datasets:[{label:'MAPE%',data:L7.map(x=>x[1]),backgroundColor:'rgba(90,209,160,.7)'}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{beginAtZero:true,ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},x:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});
 if(WX&&WX.temp&&WX.ghi)ch('c_wx',{type:'line',data:{labels:hrs,datasets:[{label:'Sicaklik (C)',data:WX.temp,borderColor:'#ff9f5a',borderWidth:2,pointRadius:2,yAxisID:'y',type:'line'},{label:'GHI (W/m2)',data:WX.ghi,borderColor:'#ffce6a',borderWidth:1.5,pointRadius:1,yAxisID:'y2',type:'line'}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8b97b3',boxWidth:10,font:{size:10}}}},scales:{x:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},y:{position:'left',title:{display:true,text:'C',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},y2:{position:'right',title:{display:true,text:'W/m2',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{drawOnChartArea:false}}}}}});};
go._inits.ozet=TABS.ozet_init;

TABS.karsilastirma=()=>{
 const k=Object.keys(CP);if(!k.length||!FC)return `<h2>Karsilastirma</h2><p class="lead">Veri yok.</p>`;
 return `<h2>D+2 Karsilastirma — ${D}</h2><p class="lead">Tahmin vs gecmis gunler.</p>
 <div class="panel"><h3>Yuk Profili + P95 Bandi</h3><div class="chartbox tall"><canvas id="c1"></canvas></div></div>
 <div class="grid2">
  <div class="panel"><h3>Normalize Profil</h3><div class="chartbox"><canvas id="c2"></canvas></div></div>
  <div class="panel"><h3>Sicaklik</h3><div class="chartbox"><canvas id="c3"></canvas></div></div>
 </div>`;};
TABS.karsilastirma_init=()=>{
 const k=Object.keys(CP);if(!k.length||!FC)return;
 const ds1=[{label:`TAHMIN ${D}`,data:FC,borderColor:'#E53935',borderWidth:3,pointRadius:4,type:'line'}];
 // P95 band
 if(P95_KEYS.length){
  const up=hrs.map(h=>P95[h]?FC[h]+P95[h].p95_err:null);
  const lo=hrs.map(h=>P95[h]?FC[h]+P95[h].p5_err:null);
  ds1.push({label:'P95 band',data:up,backgroundColor:'rgba(229,57,53,.12)',borderColor:'transparent',pointRadius:0,fill:'-2',type:'line'});
  ds1.push({label:'_',data:lo,backgroundColor:'rgba(229,57,53,.12)',borderColor:'transparent',pointRadius:0,fill:false,type:'line'});}
 k.forEach((l,i)=>ds1.push({label:l,data:CP[l].load,borderColor:pal[(i+1)%6],borderWidth:1.5,pointRadius:1,type:'line',borderDash:[6,3]}));
 ch('c1',{type:'line',data:{labels:hrs,datasets:ds1},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8b97b3',boxWidth:10,font:{size:9}}}},scales:{x:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},y:{title:{display:true,text:'MWh',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});
 const fm=FC.reduce((a,b)=>a+b,0)/FC.length;
 const ds2=[{label:`TAHMIN ${D}`,data:FC.map(v=>v/fm),borderColor:'#E53935',borderWidth:3,pointRadius:4,type:'line'}];
 k.forEach((l,i)=>{const m=CP[l].load.reduce((a,b)=>a+b,0)/CP[l].load.length;ds2.push({label:l,data:CP[l].load.map(v=>v/m),borderColor:pal[(i+1)%6],borderWidth:1.5,pointRadius:1,type:'line',borderDash:[6,3]});});
 ch('c2',{type:'line',data:{labels:hrs,datasets:ds2},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8b97b3',boxWidth:10,font:{size:9}}}},scales:{x:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},y:{title:{display:true,text:'Yuk/Ort',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});
 const ds3=[];if(WX&&WX.temp)ds3.push({label:`TAHMIN ${D}`,data:WX.temp,borderColor:'#E53935',borderWidth:2,pointRadius:2,type:'line'});
 k.forEach((l,i)=>{if(CP[l].temp)ds3.push({label:l,data:CP[l].temp,borderColor:pal[(i+1)%6],borderWidth:1.5,pointRadius:0,type:'line',borderDash:[6,3]});});
 if(ds3.length)ch('c3',{type:'line',data:{labels:hrs,datasets:ds3},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8b97b3',boxWidth:10,font:{size:9}}}},scales:{x:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},y:{title:{display:true,text:'C',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});};
go._inits.karsilastirma=TABS.karsilastirma_init;

TABS.p95=()=>{
 if(!P95_KEYS.length)return `<h2>P95 Prediction Interval</h2><p class="lead">Henuz hesaplanamadi (60 gun backtest verisi gerek).</p>`;
 return `<h2>P95 Prediction Interval</h2><p class="lead">Gecmis ${Object.keys(P95).length*24} saatlik hata dagilimindan hesaplanan P5-P95 araligi.</p>
 <div class="panel"><h3>Saatlik APE Dagilimi (ortalama + P5/P95)</h3><div class="chartbox tall"><canvas id="cp1"></canvas></div></div>
 <div class="grid2">
  <div class="panel"><h3>P50 APE (medyan hata)</h3><div class="chartbox"><canvas id="cp2"></canvas></div></div>
  <div class="panel"><h3>Tahmin Hata Bandi (MWh)</h3><div class="chartbox"><canvas id="cp3"></canvas></div></div>
 </div>`;};
TABS.p95_init=()=>{
 if(!P95_KEYS.length)return;
 const ds=[{label:'P95 APE%',data:hrs.map(h=>P95[h]?P95[h].p95_ape:null),backgroundColor:'rgba(255,107,107,.3)',borderColor:'#ff6b6b',borderWidth:1.5,type:'line'},{label:'P50 APE%',data:hrs.map(h=>P95[h]?P95[h].p50_ape:null),borderColor:'#ffce6a',borderWidth:2,pointRadius:3,type:'line'},{label:'P5 APE%',data:hrs.map(h=>P95[h]?P95[h].p5_ape:null),borderColor:'#5ad1a0',borderWidth:1.5,type:'line'}];
 ch('cp1',{type:'line',data:{labels:hrs,datasets:ds},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8b97b3',boxWidth:10,font:{size:9}}}},scales:{x:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},y:{title:{display:true,text:'APE%',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});
 ch('cp2',{type:'bar',data:{labels:hrs.map(h=>h+':00'),datasets:[{label:'P50 APE%',data:hrs.map(h=>P95[h]?P95[h].p50_ape:0),backgroundColor:hrs.map(h=>P95[h]&&P95[h].p50_ape>5?'rgba(255,107,107,.7)':'rgba(90,209,160,.7)')}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},x:{ticks:{color:'#8b97b3',maxTicksLimit:12},grid:{color:'#1c2334'}}}}});
 const up=hrs.map(h=>FC[h]+(P95[h]?P95[h].p95_err:0));
 const lo=hrs.map(h=>FC[h]+(P95[h]?P95[h].p5_err:0));
 ch('cp3',{type:'line',data:{labels:hrs,datasets:[{label:'P95 ust',data:up,borderColor:'transparent',backgroundColor:'rgba(72,213,151,.15)',pointRadius:0,fill:false,type:'line'},{label:'P5 alt',data:lo,borderColor:'transparent',pointRadius:0,fill:'-2',backgroundColor:'rgba(72,213,151,.15)',type:'line'},{label:'TAHMIN',data:FC,borderColor:'#E53935',borderWidth:3,pointRadius:4,type:'line'}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8b97b3',boxWidth:10,font:{size:9}}}},scales:{x:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},y:{title:{display:true,text:'MWh',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});
 };
go._inits.p95=TABS.p95_init;

TABS.cross=()=>{
 const k=Object.keys(CP);if(!k.length)return `<h2>Cross Check</h2><p class="lead">Veri yok.</p>`;
 return `<h2>Cross Check — Sicaklik, Bulut, GHI</h2><p class="lead">Her karsilastirma gunu icin cevresel faktorlerin yukle iliskisi.</p>
 <div class="grid2">
  <div class="panel"><h3>Sicaklik x Yuk (scatter)</h3><div class="chartbox tall"><canvas id="cx_t"></canvas></div></div>
  <div class="panel"><h3>GHI x Yuk (scatter)</h3><div class="chartbox tall"><canvas id="cx_g"></canvas></div></div>
 </div>
 <div class="grid2">
  <div class="panel"><h3>Bulut x Yuk (scatter)</h3><div class="chartbox tall"><canvas id="cx_c"></canvas></div></div>
  <div class="panel"><h3>Normalize Sicaklik overlay</h3><div class="chartbox tall"><canvas id="cx_s"></canvas></div></div>
 </div>`;};
TABS.cross_init=()=>{
 const k=Object.keys(CP);if(!k.length)return;
 // Temp scatter
 let tds=[];
 if(WX&&WX.temp)tds.push({label:`TAHMIN ${D}`,data:WX.temp.map((v,i)=>({x:v,y:FC[i]})),backgroundColor:'#E53935',pointRadius:5});
 k.forEach((l,i)=>{if(CP[l].temp)tds.push({label:l,data:CP[l].temp.map((v,j)=>({x:v,y:CP[l].load[j]})),backgroundColor:pal[(i+1)%6],pointRadius:3});});
 ch('cx_t',{type:'scatter',data:{datasets:tds},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8b97b3',boxWidth:10,font:{size:9}}}},scales:{x:{title:{display:true,text:'Sicaklik (C)',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},y:{title:{display:true,text:'MWh',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});
 // GHI scatter
 let gds=[];k.forEach((l,i)=>{if(CP[l].ghi)gds.push({label:l,data:CP[l].ghi.map((v,j)=>({x:v,y:CP[l].load[j]})),backgroundColor:pal[(i+1)%6],pointRadius:3});});
 if(gds.length)ch('cx_g',{type:'scatter',data:{datasets:gds},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8b97b3',boxWidth:10,font:{size:9}}}},scales:{x:{title:{display:true,text:'GHI (W/m2)',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},y:{title:{display:true,text:'MWh',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});
 // Cloud scatter
 let cds=[];k.forEach((l,i)=>{if(CP[l].cloud)cds.push({label:l,data:CP[l].cloud.map((v,j)=>({x:v,y:CP[l].load[j]})),backgroundColor:pal[(i+1)%6],pointRadius:3});});
 if(cds.length)ch('cx_c',{type:'scatter',data:{datasets:cds},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8b97b3',boxWidth:10,font:{size:9}}}},scales:{x:{title:{display:true,text:'Bulut (%)',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},y:{title:{display:true,text:'MWh',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});
 // Temp overlay
 let sds=[];k.forEach((l,i)=>{if(CP[l].temp)sds.push({label:l,data:CP[l].temp.map(v=>v-(CP[l].temp.reduce((a,b)=>a+b,0)/CP[l].temp.length)),borderColor:pal[(i+1)%6],borderWidth:1.5,pointRadius:1,type:'line',borderDash:[6,3]});});
 if(sds.length)ch('cx_s',{type:'line',data:{labels:hrs,datasets:sds},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8b97b3',boxWidth:10,font:{size:9}}}},scales:{x:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},y:{title:{display:true,text:'Sicaklik anomalisi (C)',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});
};
go._inits.cross=TABS.cross_init;

TABS.sensitivity=()=>{
 const e=Object.entries(SN);if(!e.length)return `<h2>Sensitivity</h2><p class="lead">Yetersiz veri.</p>`;
 let h=`<h2>Sensitivity - Sicaklik / Yuk Iliskisi</h2><p class="lead">Her 1C sicaklik artisinda beklenen yuk degisimi (MW/C).</p>
 <div class="panel"><h3>Sezonsal Sensitivity</h3><div class="chartbox"><canvas id="cs1"></canvas></div></div>
 <div class="panel"><h3>Nasil Kullanilir?</h3><p class="note" style="font-size:12.5px;line-height:1.6">`;
 e.forEach(([s,v])=>{h+=`<b>${s}:</b> +1C ~ <b>${v} MW</b><br>`;});
 const yaz=e.find(([s])=>s==='Yaz');
 if(yaz)h+=`<br><b>Senaryo:</b> Yarin sicaklik tahmini 2C yuksek cikarsa, ADM yuku ~<b>${Math.round(yaz[1]*2)} MW</b> artar.`;
 h+=`<br><b>Uyari:</b> Bu basit dogrusal modeldir. GHI, bulut, nem gibi faktorler sabit kabul edilmistir.</p></div>`;
 return h;};
TABS.sensitivity_init=()=>{
 const e=Object.entries(SN);
 if(e.length)ch('cs1',{type:'bar',data:{labels:e.map(x=>x[0]),datasets:[{label:'MW/C',data:e.map(x=>x[1]),backgroundColor:['#E53935','#1E88E5','#43A047','#FB8C00']}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{title:{display:true,text:'MW / C',color:'#8b97b3'},ticks:{color:'#8b97b3'},beginAtZero:true,grid:{color:'#1c2334'}},x:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});};
go._inits.sensitivity=TABS.sensitivity_init;

TABS.rec=()=>{
 let h=`<h2>Oneriler</h2><p class="lead">Tahmin guvenilirligi ve senaryo analizi.</p>`;
 const k=Object.keys(CP);
 let count=0;
 if(FC&&k.length){
  const lw=CP[k[0]];
  if(lw){
   const diff=FC.map((v,i)=>v-lw.load[i]);
   const md=diff.reduce((a,b)=>a+b,0)/diff.length;
   const ape=diff.map(Math.abs).reduce((a,b)=>a+b,0)/diff.length/Math.abs(FC.reduce((a,b)=>a+b,0)/FC.length)*100;
   const mx=diff.indexOf(Math.max(...diff)), mn=diff.indexOf(Math.min(...diff));
   count++;
   h+=`<div class="rm"><h4>Haftalik Degerlendirme</h4>
    <div class="meta">Guven: <b style="color:${ape<3?'#5ad1a0':ape<7?'#ffce6a':'#ff6b6b'}">${ape<3?'YUKSEK':ape<7?'ORTA':'DUSUK'}</b></div>
    <p>Tahmin ortalamasi gecen haftaya gore <b>${Math.abs(md).toFixed(0)} MWh ${md>0?'yuksek':'dusuk'}</b>.
    MAPE: <b>${ape.toFixed(1)}%</b>. En buyuk fark saat <b>${mx}:00</b> (${diff[mx].toFixed(0)} MWh),
    en kucuk fark saat <b>${mn}:00</b> (${diff[mn].toFixed(0)} MWh).</p></div>`;}}
 if(FC&&WX&&WX.temp&&k.length){
  const lw=CP[k[0]];
  if(lw&&lw.temp){
   const tdiff=WX.temp[13]-lw.temp[13]; // 14:00 temp diff
   count++;
   h+=`<div class="rm" style="border-left-color:#6aa9ff"><h4>Sicaklik Analizi</h4>
    <div class="meta">14:00 sicaklik: Tahmin=<b>${WX.temp[13].toFixed(1)}C</b> | Gecen hafta=<b>${lw.temp[13].toFixed(1)}C</b></div>
    <p>Ogle sicakligi gecen haftaya gore <b>${tdiff>0?'+':''}${tdiff.toFixed(1)}C</b>.
    ${Math.abs(tdiff)>3?'Bu buyuk bir fark — tahminin guvenilirligi dusuk olabilir.':'Fark makul — tahmin guvenilir.'}</p></div>`;}}
 if(P95_KEYS.length){
  const maxP95=Math.max(...P95_KEYS.map(h=>P95[h].p95_ape));
  const maxHr=P95_KEYS.find(h=>P95[h].p95_ape===maxP95);
  count++;
  h+=`<div class="rm" style="border-left-color:#ffce6a"><h4>Prediction Interval</h4>
   <div class="meta">P95 MAPE: <b>${(Object.values(P95).reduce((a,b)=>a+b.p95_ape,0)/P95_KEYS.length).toFixed(1)}%</b></div>
   <p>En genis hata bandi saat <b>${maxHr}:00</b> (P95 APE=${maxP95.toFixed(1)}%).
   ${maxP95>10?'Bu saatte tahmin guvenilirligi dusuk — ek dikkat gerek.':'Tahmin genel olarak guvenilir.'}</p></div>`;}
 if(Object.keys(SN).length){
  const yaz=Object.entries(SN).find(([s])=>s==='Yaz');
  if(yaz){
   count++;
   h+=`<div class="rm" style="border-left-color:#5ad1a0"><h4>Senaryo Motoru</h4>
    <div class="meta">Sensitivity: Yazin +1C ~ ${yaz[1]} MW</div>
    <p><b>Alternatif senaryo:</b> Yarinki hava tahminine gore sicaklik +3C saparsa, ADM yuku yaklasik
    <b>${Math.round(yaz[1]*3)} MW</b> artar. Bu durumda pik talep <b>${Math.round(Math.max(...FC)+yaz[1]*3)} MWh</b>'ye ulasabilir.</p>
    <p><b>Dusuk sicaklik senaryosu:</b> -3C saparsa, yuk <b>${Math.round(Math.max(...FC)-yaz[1]*3)} MWh</b>'ye duser.</p></div>`;}}
 if(count===0)h=`<h2>Oneriler</h2><p class="lead">Tahmin verisi yok.</p>`;
 return h;};
TABS.rec_init=()=>{};
go._inits.rec=TABS.rec_init;

// ── BOOT ────────────────────────────────────────────────────────
document.getElementById('nv').innerHTML=[
 ['ozet','Ozet'],['karsilastirma','D+2 Karsilastirma'],
 ['p95','P95 Interval'],['cross','Cross Check'],
 ['sensitivity','Sensitivity'],['rec','Oneriler']
].map(([i,l])=>`<button data-t="${i}">${l}</button>`).join('');
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>go(b.dataset.t));

Object.entries(TABS).forEach(([id,tab])=>{
 const d=document.createElement('div');d.className='tab';d.id='tab_'+id;
 d.innerHTML=tab();document.getElementById('mn').appendChild(d);});
go(location.hash?location.hash.slice(1):'ozet');
</script></body></html>
"""

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(H, encoding="utf-8")
print(f"SAVED: {OUT} ({len(H):,} bytes)")

def run() -> dict:
    return {"status": "ok", "file": str(OUT)}

if __name__ == "__main__":
    print(run())
