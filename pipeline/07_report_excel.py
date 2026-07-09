"""
07_report_excel.py — Otomatik 5-Tablolu Excel Raporu
====================================================
Her forecast sonrasi calisir, ADM + GDZ ayri sheet.
Tablo 1: D+2 Gerceklesen (saatlik, saga sutun eklenir)
Tablo 2: D+2 Tahmin vs Gerceklesen + Sapma%
Tablo 3: D+2 Gunluk MAPE/ME/MAE
"""
import sys, os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
import shutil

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config_live as C

REPORT_FILE = C.OUTPUT_DIR / "STLF_LIVE_RAPOR.xlsx"
HOURS = list(range(24))
HLAB = [f"{h:02d}:00" for h in HOURS]

# Row pozisyonlari (0-indexed)
R1, R2, R3 = 0, 26, 52

def get_act(master, target_col, d):
    day = master[master[C.RAW_DATE_COL].dt.date == d]
    if len(day) != 24: return None
    return day.set_index(C.RAW_HOUR_COL)[target_col].sort_index()

def load_fc(path):
    if not path.exists(): return None
    try:
        fc = pd.read_excel(path, sheet_name="Tahmin")
        return fc.set_index("Saat")["Tahmin_MWh"]
    except: return None

def mape(p, a):
    m = (a > 0) & (~np.isnan(p)) & (~np.isnan(a))
    return float(np.mean(np.abs((a[m]-p[m])/a[m])*100)) if m.sum() else np.nan
def me(p, a):
    m = (~np.isnan(p)) & (~np.isnan(a))
    return float(np.mean(p[m]-a[m])) if m.sum() else np.nan

def build_tables(master, target_col, fc_dir, sheet_name, writer):
    today = date.today()
    # Son 14 gun D+2 teslim (14 gun once bugun, yani Gecmis)
    d2 = [today - timedelta(days=i) for i in range(14, 0, -1)]
    
    # ── TABLO 1: D+2 Gerceklesen ──────────────────────────────────
    t1_data = {"Saat": HLAB}
    for d in d2:
        s = get_act(master, target_col, d)
        if s is not None:
            t1_data[str(d)] = [f"{s.get(h, np.nan):.0f}" for h in HOURS]
    
    # Ortalama satiri
    avg_row = {"Saat": "ORTALAMA"}
    for k in t1_data:
        if k == "Saat": continue
        nums = []
        for x in t1_data[k]:
            try: nums.append(float(x))
            except: pass
        avg_row[k] = f"{np.mean(nums):.0f}" if nums else ""
    # Ortalama satirini df'e ekle (sona)
    t1_df = pd.DataFrame(t1_data)
    avg_df = pd.DataFrame([avg_row])
    t1_out = pd.concat([t1_df, avg_df], ignore_index=True)
    t1_out.to_excel(writer, sheet_name=sheet_name, startrow=R1, index=False)
    
    # ── TABLO 2: D+2 Tahmin vs Gerceklesen + Sapma ──────────────
    # Header satiri
    cols = ["Saat"]
    for d in d2:
        ds = str(d)
        cols += [f"{ds}_Tahmin", f"{ds}_Gercek", f"{ds}_Sapma%"]
    t2_data = {c: [] for c in cols}
    
    for h in HOURS:
        t2_data["Saat"].append(HLAB[h])
        for d in d2:
            ds = str(d)
            act = get_act(master, target_col, d)
            fc = load_fc(fc_dir / f"{d}_forecast.xlsx")
            if fc is not None and act is not None:
                fv, av = fc.get(h, np.nan), act.get(h, np.nan)
                t2_data[f"{ds}_Tahmin"].append(f"{fv:.0f}" if not np.isnan(fv) else "")
                t2_data[f"{ds}_Gercek"].append(f"{av:.0f}" if not np.isnan(av) else "")
                ap = np.abs(av-fv)/av*100 if av > 0 else np.nan
                t2_data[f"{ds}_Sapma%"].append(f"{ap:.1f}" if not np.isnan(ap) else "")
            else:
                t2_data[f"{ds}_Tahmin"].append("")
                t2_data[f"{ds}_Gercek"].append("")
                t2_data[f"{ds}_Sapma%"].append("")
    
    pd.DataFrame(t2_data).to_excel(writer, sheet_name=sheet_name,
                                    startrow=R2, index=False)
    
    # ── TABLO 3: D+2 Gunluk MAPE/ME ──────────────────────────────
    t3_data = {"Metrik": ["MAPE%", "ME_MWh"]}
    for d in d2:
        ds = str(d)
        act = get_act(master, target_col, d)
        fc = load_fc(fc_dir / f"{d}_forecast.xlsx")
        if fc is not None and act is not None:
            fv = np.array([fc.get(h, np.nan) for h in HOURS])
            av = np.array([act.get(h, np.nan) for h in HOURS])
            t3_data[ds] = [f"{mape(fv, av):.2f}", f"{me(fv, av):.1f}"]
    
    pd.DataFrame(t3_data).to_excel(writer, sheet_name=sheet_name,
                                    startrow=R3, index=False)


def run():
    print("\n[07] STLF LIVE RAPOR...")
    master = pd.read_parquet(C.MASTER_PARQUET)
    master[C.RAW_DATE_COL] = pd.to_datetime(master[C.RAW_DATE_COL])
    
    with pd.ExcelWriter(REPORT_FILE, engine="openpyxl") as writer:
        build_tables(master, C.RAW_TARGET_COL, C.OUTPUT_DIR, "ADM", writer)
        
        # GDZ
        try:
            gdz_path = ROOT.parent.parent / "çağatay" / "gdz talep" / "live"
            sys.path.insert(0, str(gdz_path))
            import config_live_gdz
            gz = pd.read_parquet(config_live_gdz.GDZ_MASTER_PARQUET)
            gz[C.RAW_HOUR_COL] = gz["Tarih"].dt.hour  # extract hour
            gz["Tarih"] = gz["Tarih"].dt.normalize()
            gz_col = [c for c in gz.columns if "Enerji" in c or "MWh" in c][0]
            gz = gz.rename(columns={"Tarih": C.RAW_DATE_COL, gz_col: "Enerji"})
            build_tables(gz, "Enerji", config_live_gdz.OUTPUT_DIR, "GDZ", writer)
            print("     [GDZ] eklendi")
        except Exception as e:
            print(f"     [GDZ] atlandi: {e}")
    
    print(f"     Rapor: {REPORT_FILE.name}")
    return {"status": "ok", "file": str(REPORT_FILE)}

if __name__ == "__main__":
    print(run())
