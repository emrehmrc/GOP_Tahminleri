"""08_diagnostic_html.py — Interaktif HTML Dashboard (Hizli)"""
import sys, json
import pandas as pd, numpy as np
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C

OUT = C.OUTPUT_DIR / f"diagnostic_{date.today().strftime('%Y-%m-%d')}.html"

# ── LOAD & MERGE ─────────────────────────────────────────────────────
print("Loading data...")
m = pd.read_parquet(C.MASTER_PARQUET)
m[C.RAW_DATE_COL] = pd.to_datetime(m[C.RAW_DATE_COL])
m["h"] = m[C.RAW_HOUR_COL]
m["dt"] = m[C.RAW_DATE_COL].dt.normalize()

wh = pd.read_parquet(C.WEATHER_HISTORY_PARQUET)
wh[C.RAW_DATE_COL] = pd.to_datetime(wh[C.RAW_DATE_COL]).dt.normalize()

# Merge once (fast) — master'daki eski hava kolonlarini düs
wx_in_master = [c for c in m.columns if c in wh.columns 
                and c not in (C.RAW_DATE_COL, C.RAW_HOUR_COL)]
if wx_in_master:
    m = m.drop(columns=wx_in_master)

merged = m.merge(
    wh[[C.RAW_DATE_COL, C.RAW_HOUR_COL, "MUGLA_MenteseCenter_app_temp_actual",
        "GHI_ADM_Weighted"]],
    on=[C.RAW_DATE_COL, C.RAW_HOUR_COL], how="left"
)
temp_col = "MUGLA_MenteseCenter_app_temp_actual"
ghi_col = "GHI_ADM_Weighted"
print(f"Merged: {len(merged)} rows")

# ── FORECAST ────────────────────────────────────────────────────────
today = date.today()
fc = None
fc7 = None
for ds in [str(today), str(today - timedelta(days=7))]:
    p = C.OUTPUT_DIR / f"{ds}_forecast.xlsx"
    if p.exists():
        try:
            d = pd.read_excel(p, sheet_name="Tahmin")
            if ds == str(today): fc = d.set_index("Saat")["Tahmin_MWh"]
            else: fc7 = d.set_index("Saat")["Tahmin_MWh"]
        except: pass

# ── COMPARISON DAYS ─────────────────────────────────────────────────
comp_dates = []
if fc is not None:
    for offset, label in [(7,"1 hafta once"),(14,"2 hafta once"),
                           (364,"gecen yil"),(350,"gecen yil-2h"),
                           (371,"gecen yil+1h")]:
        cd = today - timedelta(days=offset)
        comp_dates.append((str(cd), label))

def get_series(ds):
    d = pd.Timestamp(ds).date()
    day = merged[merged["dt"].dt.date == d]
    if len(day) != 24: return None
    day = day.set_index(C.RAW_HOUR_COL).sort_index()
    return {
        "load": day[C.RAW_TARGET_COL].values.tolist(),
        "temp": day[temp_col].values.tolist() if temp_col in day.columns else None,
        "ghi": day[ghi_col].values.tolist() if ghi_col in day.columns else None,
    }

comp_series = {}
for ds, label in comp_dates:
    s = get_series(ds)
    if s: comp_series[label] = s

# ── SENSITIVITY (vectorized) ─────────────────────────────────────────
print("Computing sensitivity...")
seasons_map = {12:"Kis",1:"Kis",2:"Kis",3:"Ilkbahar",4:"Ilkbahar",5:"Ilkbahar",
               6:"Yaz",7:"Yaz",8:"Yaz",9:"Sonbahar",10:"Sonbahar",11:"Sonbahar"}
merged["season"] = merged["dt"].dt.month.map(seasons_map)
merged["t_valid"] = merged[temp_col].notna() & (merged[C.RAW_TARGET_COL] > 0)

sensitivity = {}
for season in ["Kis","Ilkbahar","Yaz","Sonbahar"]:
    seg = merged[(merged["season"] == season) & merged["t_valid"]]
    if len(seg) < 500: continue
    # Percentile trim (10-90)
    lo, hi = np.percentile(seg[temp_col], 10), np.percentile(seg[temp_col], 90)
    seg = seg[(seg[temp_col] >= lo) & (seg[temp_col] <= hi)]
    if len(seg) < 200: continue
    t, y = seg[temp_col].values, seg[C.RAW_TARGET_COL].values
    slope = np.cov(t, y)[0,1] / np.var(t) if np.var(t) > 0 else 0
    sensitivity[season] = round(slope, 1)

# ── LAST 7 DAYS MAPE ────────────────────────────────────────────────
last7 = []
# Try to load forecasts for last 7 days and compare with actuals
for i in range(1, 8):
    d = today - timedelta(days=i)
    ds = str(d)
    p = C.OUTPUT_DIR / f"{ds}_forecast.xlsx"
    if not p.exists(): continue
    try:
        fc_d = pd.read_excel(p, sheet_name="Tahmin").set_index("Saat")["Tahmin_MWh"]
    except: continue
    act = get_series(ds)
    if not act: continue
    av = np.array(act["load"])
    fv = np.array([fc_d.get(h, np.nan) for h in range(24)])
    msk = av > 0
    if msk.sum() < 12: continue
    mape = float(np.mean(np.abs((av[msk]-fv[msk])/av[msk])) * 100)
    last7.append((ds, round(mape, 2)))

# ── BUILD HTML ───────────────────────────────────────────────────────
def js(v):
    if v is None: return "null"
    if isinstance(v, list): return "[" + ",".join("null" if x is None else f"{x:.1f}" for x in v) + "]"
    return json.dumps(v)

fc_load = fc.values.tolist() if fc is not None else None
fc_temp = None
if fc is not None:
    today_wx = merged[merged["dt"].dt.date == today]
    if len(today_wx) == 24:
        fc_temp = today_wx.set_index(C.RAW_HOUR_COL).sort_index()[temp_col].values.tolist()

html = """<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>STLF DIAGNOSTIC</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0b0e14;--panel:#131823;--line:#26304a;--ink:#dfe6f3;--muted:#8b97b3;--acc:#5ad1a0;--acc2:#6aa9ff;--warn:#ffce6a;--bad:#ff6b6b;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:'Segoe UI',sans-serif;font-size:14px}
.app{display:flex;min-height:100vh}
nav{width:180px;background:linear-gradient(180deg,#0d111a,#0b0e14);border-right:1px solid var(--line);padding:14px 0;position:sticky;top:0;height:100vh;overflow:auto;flex-shrink:0}
nav h1{font-size:12px;color:var(--acc);margin:6px 14px 10px;font-weight:700;letter-spacing:.05em}
nav button{display:block;width:100%;text-align:left;background:none;border:0;color:var(--muted);padding:8px 14px;cursor:pointer;font-size:12px;border-left:3px solid transparent;transition:.15s}
nav button:hover{color:var(--ink);background:#11151f}
nav button.on{color:var(--ink);background:#11151f;border-left-color:var(--acc);font-weight:600}
main{flex:1;padding:20px 24px 60px;max-width:1100px}
.tab{display:none;animation:fade .2s}.tab.on{display:block}
@keyframes fade{from{opacity:0}to{opacity:1}}
h2{font-size:18px;margin:0 0 4px;font-weight:700}
.lead{color:var(--muted);margin:0 0 14px;font-size:12.5px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-bottom:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:10px 12px}
.card .lab{font-size:10px;color:var(--muted);text-transform:uppercase}
.card .val{font-size:22px;font-weight:700;margin-top:2px}
.chartbox{position:relative;height:260px;margin-bottom:16px}
.chartbox.tall{height:320px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.panel h3{margin:0 0 3px;font-size:13.5px}
.panel .note{color:var(--muted);font-size:11.5px;margin:0 0 8px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.rm{padding:12px 14px;margin-bottom:10px;background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:9px}
.rm h4{margin:0 0 3px;font-size:13.5px}.rm p{margin:4px 0;font-size:12px;color:#c7d0e3;line-height:1.5}
@media(max-width:800px){.grid2{grid-template-columns:1fr}}
</style></head><body>
<div class="app"><nav>
<h1>STLF DIAGNOSTIC</h1>
<div style="font-size:10px;color:var(--muted);margin:-6px 14px 10px">""" + str(today) + """</div>
<div id="nav"></div></nav>
<main id="main"></main></div>
<script>
const D=""" + json.dumps(str(today)) + """;
const FC=""" + json.dumps(fc_load) + """;
const FC_WX=""" + json.dumps(fc_temp) + """;
const COMP=""" + json.dumps(comp_series) + """;
const SENS=""" + json.dumps(sensitivity) + """;
const L7=""" + json.dumps(last7) + """;

const pal=['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1'];
function ch(id,cfg){const c=document.getElementById(id);if(!c)return;return new Chart(c,cfg);}
const hrs=Array.from({length:24},(_,i)=>i);

document.getElementById('nav').innerHTML=[
 ['ozet','Ozet'],['karsilastirma','D+2 Karsilastirma'],
 ['sensitivity','Sensitivity'],['rec','Oneriler']
].map(([i,l])=>`<button data-t="${i}">${l}</button>`).join('');
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>go(b.dataset.t));

function go(id){
 document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===id));
 document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.id==='tab_'+id));
 location.hash=id;}

const TABS={
ozet(){
 let h=`<h2>Ozet</h2><p class="lead">Son 7 gun MAPE + bugunun tahmini.</p>
 <div class="cards"><div class="card"><div class="lab">Tahmin Gunu</div><div class="val" style="font-size:16px">${D}</div></div>`;
 if(FC){const m=FC.reduce((a,b)=>a+b,0)/FC.length;
 h+=`<div class="card"><div class="lab">Tahmin Ort</div><div class="val" style="color:#5ad1a0">${m.toFixed(0)}</div><div class="lab">MWh</div></div>`;
 h+=`<div class="card"><div class="lab">Pik</div><div class="val" style="color:#ffce6a">${Math.max(...FC).toFixed(0)}</div><div class="lab">saat ${FC.indexOf(Math.max(...FC))}:00</div></div>`;}
 h+=`</div>`;
 if(L7&&L7.length){
 h+=`<div class="panel"><h3>Son 7 Gun MAPE</h3><div class="chartbox"><canvas id="c7"></canvas></div></div>`;}
 return h;},
ozet_init(){
 if(L7&&L7.length) ch('c7',{type:'bar',data:{labels:L7.map(x=>x[0]),datasets:[{label:'MAPE%',data:L7.map(x=>x[1]),backgroundColor:'rgba(90,209,160,.7)'}]},
  options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
   scales:{y:{beginAtZero:true,ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},x:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}}});},
   
karsilastirma(){
 const keys=Object.keys(COMP);
 if(!keys.length||!FC) return `<h2>D+2 Karsilastirma</h2><p class="lead">Veri yok.</p>`;
 return `<h2>D+2 Karsilastirma — ${D}</h2><p class="lead">Tahmin gecmis gunlerle karsilastirma.</p>
 <div class="panel"><h3>Yuk Profili</h3><div class="chartbox tall"><canvas id="c1"></canvas></div></div>
 <div class="panel"><h3>Normalize Profil</h3><div class="chartbox"><canvas id="c2"></canvas></div></div>
 <div class="grid2">
  <div class="panel"><h3>Sicaklik</h3><div class="chartbox"><canvas id="c3"></canvas></div></div>
  <div class="panel"><h3>Sicaklik x Yuk</h3><div class="chartbox"><canvas id="c4"></canvas></div></div>
 </div>`;},
karsilastirma_init(){
 const keys=Object.keys(COMP);if(!keys.length||!FC)return;
 const ds1=[{label:`TAHMIN ${D}`,data:FC,borderColor:'#E53935',borderWidth:3,pointRadius:4,type:'line'}];
 keys.forEach((k,i)=>ds1.push({label:k,data:COMP[k].load,borderColor:pal[(i+1)%6],borderWidth:1.5,pointRadius:1,type:'line',borderDash:[6,3]}));
 ch('c1',{type:'line',data:{labels:hrs,datasets:ds1},options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{labels:{color:'#8b97b3',boxWidth:10,font:{size:10}}}},
  scales:{x:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},
          y:{title:{display:true,text:'MWh',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});
 const fm=FC.reduce((a,b)=>a+b,0)/FC.length;
 const ds2=[{label:`TAHMIN ${D}`,data:FC.map(v=>v/fm),borderColor:'#E53935',borderWidth:3,pointRadius:4,type:'line'}];
 keys.forEach((k,i)=>{const m=COMP[k].load.reduce((a,b)=>a+b,0)/COMP[k].load.length;
  ds2.push({label:k,data:COMP[k].load.map(v=>v/m),borderColor:pal[(i+1)%6],borderWidth:1.5,pointRadius:1,type:'line',borderDash:[6,3]});});
 ch('c2',{type:'line',data:{labels:hrs,datasets:ds2},options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{labels:{color:'#8b97b3',boxWidth:10,font:{size:10}}}},
  scales:{x:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},y:{title:{display:true,text:'Yuk / Ortalama',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});
 const ds3=[];
 if(FC_WX) ds3.push({label:`TAHMIN ${D}`,data:FC_WX,borderColor:'#E53935',borderWidth:2,pointRadius:2,type:'line'});
 keys.forEach((k,i)=>{if(COMP[k].temp) ds3.push({label:k,data:COMP[k].temp,borderColor:pal[(i+1)%6],borderWidth:1.5,pointRadius:0,type:'line',borderDash:[6,3]});});
 if(ds3.length) ch('c3',{type:'line',data:{labels:hrs,datasets:ds3},options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{labels:{color:'#8b97b3',boxWidth:10,font:{size:9}}}},
  scales:{x:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},y:{title:{display:true,text:'Sicaklik (C)',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});
 const ds4=[];
 if(FC_WX) ds4.push({label:`TAHMIN ${D}`,data:FC_WX.map((v,i)=>({x:v,y:FC[i]})),backgroundColor:'#E53935',pointRadius:5});
 keys.forEach((k,i)=>{if(COMP[k].temp) ds4.push({label:k,data:COMP[k].temp.map((v,j)=>({x:v,y:COMP[k].load[j]})),backgroundColor:pal[(i+1)%6],pointRadius:3});});
 if(ds4.length) ch('c4',{type:'scatter',data:{datasets:ds4},options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{labels:{color:'#8b97b3',boxWidth:10,font:{size:9}}}},
  scales:{x:{title:{display:true,text:'Sicaklik (C)',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},
   y:{title:{display:true,text:'MWh',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}});
},

sensitivity(){
 const e=Object.entries(SENS);
 if(!e.length) return `<h2>Sensitivity</h2><p class="lead">Yetersiz veri.</p>`;
 let h=`<h2>Sensitivity Analizi</h2><p class="lead">Her 1C sicaklik artisinda beklenen yuk degisimi (MW/C).</p>
 <div class="panel"><h3>Sezonsal Sensitivity</h3><div class="chartbox"><canvas id="cs"></canvas></div></div>
 <div class="panel"><p class="note" style="font-size:13px;line-height:1.6">`;
 e.forEach(([s,v])=>{h+=`<b>${s}:</b> +1C ~ <b>${v} MW</b><br>`;});
 h+=`<br><b>Yorum:</b> ADM bolgesinde yazin sicaklik 25C'den 30C'ye cikarsa, 
 yuk yaklasik <b>${Math.round(e.find(([s])=>s==='Yaz')?.[1]||0)*5} MW</b> artar.
 Bu regresyon bazli bir yaklasimdir - GHI, bulut, rutubet etkileri sabit kabul edilmistir.</p></div>`;
 return h;},
sensitivity_init(){
 const e=Object.entries(SENS);
 if(e.length) ch('cs',{type:'bar',data:{labels:e.map(x=>x[0]),datasets:[{label:'MW/C',data:e.map(x=>x[1]),backgroundColor:['#E53935','#1E88E5','#43A047','#FB8C00']}]},
  options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
   scales:{y:{title:{display:true,text:'MW / C',color:'#8b97b3'},ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}},
    x:{ticks:{color:'#8b97b3'},grid:{color:'#1c2334'}}}}}});},

rec(){
 let h=`<h2>Oneriler</h2><p class="lead">Tahmin guvenilirligi.</p>`;
 const keys=Object.keys(COMP);
 if(FC&&keys.length){
  const lw=COMP[keys[0]]; if(lw){
   const diff=FC.map((v,i)=>v-lw.load[i]);
   const md=diff.reduce((a,b)=>a+b,0)/diff.length;
   const mape=diff.map(Math.abs).reduce((a,b)=>a+b,0)/diff.length/FC.reduce((a,b)=>a+b,0)*24*100;
   const mx=diff.indexOf(Math.max(...diff)), mn=diff.indexOf(Math.min(...diff));
   h+=`<div class="rm"><h4>Haftalik Degerlendirme</h4>
    <p>Tahmin ortalamasi gecen haftaya gore <b>${Math.abs(md).toFixed(0)} MWh ${md>0?'yuksek':'dusuk'}</b>.
    MAPE: <b>${mape.toFixed(1)}%</b>. En buyuk sapma saat <b>${mx}:00</b> (${diff[mx].toFixed(0)} MWh).</p></div>`;}}
 if(FC_WX){
  h+=`<div class="rm" style="border-left-color:var(--acc2)"><h4>Sicaklik Notu</h4>
   <p>Tahmin sicaklik ortalamasi <b>${(FC_WX.reduce((a,b)=>a+b,0)/FC_WX.length).toFixed(1)}C</b>,
   maks <b>${Math.max(...FC_WX).toFixed(1)}C</b>.</p></div>`;}
 h+=`<div class="rm" style="border-left-color:var(--warn)"><h4>Sensitivity Senaryo</h4>
  <p>Mevcut sicaklik sensitivity degerleri ile (orn: Yazin +1C ~ ${SENS.Yaz||'?'} MW) alternatif hava senaryolarina
  gore yuk tahminleri yakinda eklenecek.</p></div>`;
 return h;},
rec_init(){}
};

Object.entries(TABS).forEach(([id,tab])=>{
 const d=document.createElement('div');d.className='tab';d.id='tab_'+id;
 d.innerHTML=tab(); document.getElementById('main').appendChild(d);});
document.querySelectorAll('nav button').forEach(b=>{
 const t=TABS[b.dataset.t]; if(t&&b.dataset.t) b.innerHTML=TABS[b.dataset.t]().match(/<h2>(.*?)</)?.[1]||b.dataset.t;});
go('ozet');
if(TABS.ozet_init) TABS.ozet_init();
</script></body></html>
"""

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(html, encoding="utf-8")
print(f"SAVED: {OUT}  ({len(html):,} bytes)")

def run() -> dict:
    """Step 08 entry point for run_daily.py"""
    print("\n[08] STLF DIAGNOSTIC HTML...")
    return {"status": "ok", "file": str(OUT)}
