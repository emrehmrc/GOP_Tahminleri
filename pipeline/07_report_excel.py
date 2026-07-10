"""
07_report_excel.py — Rapor + Diagnostic
========================================
Tek script, 3 cikti:
  1. forecast.xlsx (musteri teslim)
  2. STLF_LIVE_RAPOR.xlsx (5 tablo, auto-append)
  3. diagnostic.html (interaktif)
"""
import sys, os, json
import pandas as pd, numpy as np
from pathlib import Path
from datetime import date, timedelta
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Border, Side

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C

H = list(range(24))
HL = [f"{h:02d}:00" for h in H]
TODAY = date.today()
REPORT = C.OUTPUT_DIR / "STLF_LIVE_RAPOR.xlsx"
GREEN = PatternFill("solid", fgColor="C6EFCE")
YELLOW = PatternFill("solid", fgColor="FFEB9C")
RED = PatternFill("solid", fgColor="FFC7CE")
HDR = PatternFill("solid", fgColor="4472C4")
HDR_F = Font(color="FFFFFF", bold=True)
BOLD = Font(bold=True)
THIN = Border(left=Side('thin'),right=Side('thin'),top=Side('thin'),bottom=Side('thin'))

def ga(m, tc, d):
    day = m[m[C.RAW_DATE_COL].dt.date == d]
    if len(day) != 24: return None
    return day.set_index(C.RAW_HOUR_COL)[tc].sort_index()

def lf(p):
    if not p.exists(): return None
    try:
        d = pd.read_excel(p, sheet_name="Tahmin")
        return d.set_index("Saat")["Tahmin_MWh"]
    except: return None

def mape(p, a):
    m = (a>0)&(~np.isnan(p))&(~np.isnan(a))
    return float(np.mean(np.abs((a[m]-p[m])/a[m])*100)) if m.sum() else None

def me(p, a):
    m = (~np.isnan(p))&(~np.isnan(a))
    return float(np.mean(p[m]-a[m])) if m.sum() else None

def make_report(master, tc, fc_dir, sheet_name):
    """Return 5 DataFrames for the 5 tables"""
    now = TODAY; d2 = [now - timedelta(days=i) for i in range(14, 0, -1)]
    ft2 = now + timedelta(days=2); ft1 = now + timedelta(days=1)
    realized = sorted(set(d for d in d2 if ga(master, tc, d) is not None))
    dates = sorted(set(realized + [ft1, ft2]))
    fc0 = lf(fc_dir / f"{TODAY}_forecast.xlsx")
    ds = [str(d) for d in dates]
    
    def vcol(cols, rows):
        """rows: list of (hour, [values per date]), cols: [date strings]"""
        out = [["Saat"] + cols + ["ORT"]]
        vals_all = {c: [] for c in cols}
        for h, vals in rows:
            row = [HL[h]]
            nv = []
            for c, v in zip(cols, vals):
                row.append(v if v else "")
                if v: nv.append(v)
            for c, v in zip(cols, vals):
                if v: vals_all[c].append(v)
            row.append(f"{np.nanmean([x for x in vals if x]):.0f}" if any(x for x in vals if x) else "")
            out.append(row)
        # avg row
        avg = ["ORTALAMA"]
        for c in cols:
            v = vals_all.get(c, [])
            avg.append(f"{np.nanmean(v):.0f}" if v else "")
        # total avg
        all_v = [x for lst in vals_all.values() for x in lst]
        avg.append(f"{np.nanmean(all_v):.0f}" if all_v else "")
        out.append(avg)
        return out
    
    # T1: Realized
    t1_rows = []
    for h in H:
        vals = []
        for d in dates:
            if d > now: vals.append(None); continue
            s = ga(master, tc, d)
            vals.append(s[h] if s is not None and h in s.index and not np.isnan(s[h]) else None)
        t1_rows.append((h, vals))
    t1 = vcol(ds, t1_rows)
    
    # T2: D+1 Forecast
    t2_rows = []
    for h in H:
        vals = []
        for d in dates:
            fc = lf(fc_dir / f"{d}_forecast.xlsx")
            if fc0 is not None and d == ft1: v = fc0.get(h, None)
            elif fc is not None: v = fc.get(h, None)
            else: v = None
            vals.append(v)
        t2_rows.append((h, vals))
    t2 = vcol(ds, t2_rows)
    
    # T3: D+1 Deviation %
    t3_rows = []
    for h in H:
        vals = []
        for d in dates:
            if d > now: vals.append(None); continue
            s = ga(master, tc, d)
            fc = lf(fc_dir / f"{d}_forecast.xlsx")
            if fc is not None and s is not None and h in fc.index and h in s.index and s[h] > 0:
                vals.append(round(abs(s[h]-fc[h])/s[h]*100, 1))
            else: vals.append(None)
        t3_rows.append((h, vals))
    t3 = vcol(ds, t3_rows)
    # MAPE/ME rows for T3
    mp_row = ["MAPE%"]; me_row = ["ME_MWh"]
    for d in dates:
        if d > now: mp_row.append(""); me_row.append(""); continue
        s = ga(master, tc, d); fc = lf(fc_dir / f"{d}_forecast.xlsx")
        if fc is not None and s is not None:
            fv = np.array([fc.get(h,np.nan) for h in H]); av = np.array([s.get(h,np.nan) for h in H])
            mp_row.append(f"{mape(fv,av):.2f}" if mape(fv,av) is not None else "")
            me_row.append(f"{me(fv,av):.1f}" if me(fv,av) is not None else "")
        else: mp_row.append(""); me_row.append("")
    mp_row.append(""); me_row.append("")
    t3.append(mp_row); t3.append(me_row)
    
    # T4: D+2 Forecast
    t4_rows = []
    for h in H:
        vals = []
        for d in dates:
            fc = lf(fc_dir / f"{d}_forecast_REGEN.xlsx")
            if fc is None: fc = lf(fc_dir / f"{d}_forecast.xlsx")
            if fc is not None: v = fc.get(h, None)
            elif fc0 is not None and d == ft2: v = fc0.get(h, None)
            else: v = None
            vals.append(v)
        t4_rows.append((h, vals))
    t4 = vcol(ds, t4_rows)
    
    # T5: D+2 Deviation %
    t5_rows = []
    for h in H:
        vals = []
        for d in dates:
            if d > now: vals.append(None); continue
            s = ga(master, tc, d)
            fc = lf(fc_dir / f"{d}_forecast_REGEN.xlsx")
            if fc is None: fc = lf(fc_dir / f"{d}_forecast.xlsx")
            if fc is not None and s is not None and h in fc.index and h in s.index and s[h] > 0:
                vals.append(round(abs(s[h]-fc[h])/s[h]*100, 1))
            else: vals.append(None)
        t5_rows.append((h, vals))
    t5 = vcol(ds, t5_rows)
    # MAPE/ME
    mp_row2 = ["MAPE%"]; me_row2 = ["ME_MWh"]
    for d in dates:
        if d > now: mp_row2.append(""); me_row2.append(""); continue
        s = ga(master, tc, d)
        fc = lf(fc_dir / f"{d}_forecast_REGEN.xlsx")
        if fc is None: fc = lf(fc_dir / f"{d}_forecast.xlsx")
        if fc is not None and s is not None:
            fv = np.array([fc.get(h,np.nan) for h in H]); av = np.array([s.get(h,np.nan) for h in H])
            mp_row2.append(f"{mape(fv,av):.2f}" if mape(fv,av) is not None else "")
            me_row2.append(f"{me(fv,av):.1f}" if me(fv,av) is not None else "")
        else: mp_row2.append(""); me_row2.append("")
    mp_row2.append(""); me_row2.append("")
    t5.append(mp_row2); t5.append(me_row2)
    
    return {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5}

def write_excel(tables, sheet_name):
    """Write 5 tables to Excel, auto-append columns"""
    import openpyxl
    # Try to load existing
    wb = None
    if REPORT.exists():
        try: wb = openpyxl.load_workbook(REPORT)
        except: pass
    if wb is None: wb = openpyxl.Workbook()
    
    # Get or create sheet
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
    else:
        ws = wb.create_sheet(sheet_name)
    
    # Row starts
    r0 = {"T1": 0, "T2": 26, "T3": 52, "T4": 78, "T5": 104}
    
    for tid, rows in tables.items():
        start_row = r0[tid]
        # Clear old content
        for r in range(start_row, start_row + len(rows) + 5):
            for c in range(1, 30):
                ws.cell(row=r+1, column=c).value = None
        
        for ri, row in enumerate(rows):
            for ci, val in enumerate(row):
                cell = ws.cell(row=start_row + ri + 1, column=ci + 1)
                cell.value = val
                cell.border = THIN
                if ri == 0:
                    cell.fill = HDR; cell.font = HDR_F
                elif val and isinstance(val, str) and val.endswith("%"):
                    try:
                        pct = float(val.rstrip("%"))
                        cell.fill = GREEN if pct < 3 else YELLOW if pct < 6 else RED
                        cell.number_format = '0.0"%"'
                    except: pass
    
    wb.save(REPORT)
    print(f"     Rapor: {REPORT.name}")


def run():
    print("\n[07] RAPOR + DIAGNOSTIC...")
    master = pd.read_parquet(C.MASTER_PARQUET)
    master[C.RAW_DATE_COL] = pd.to_datetime(master[C.RAW_DATE_COL])
    
    tables = make_report(master, C.RAW_TARGET_COL, C.OUTPUT_DIR, "ADM")
    write_excel(tables, "ADM")
    
    # GDZ
    try:
        gdz_path = ROOT.parent.parent / "çağatay" / "gdz talep" / "live"
        sys.path.insert(0, str(gdz_path))
        import config_live_gdz
        gz = pd.read_parquet(config_live_gdz.GDZ_MASTER_PARQUET)
        gz["Tarih"] = pd.to_datetime(gz["Tarih"])
        gz[C.RAW_HOUR_COL] = gz["Tarih"].dt.hour
        gz = gz.rename(columns={"Tarih": C.RAW_DATE_COL, 
                                config_live_gdz.GDZ_RAW_TARGET_COL: "Enerji"})
        gz_tables = make_report(gz, "Enerji", config_live_gdz.OUTPUT_DIR, "GDZ")
        write_excel(gz_tables, "GDZ")
        print("     [GDZ] eklendi")
    except Exception as e:
        print(f"     [GDZ] atlandi: {e}")
    
    # Diagnostic HTML
    print("\n     [08] DIAGNOSTIC HTML...")
    diag_script = ROOT / "pipeline" / "08_diagnostic_html.py"
    if diag_script.exists():
        ret = os.system(f'"{sys.executable}" "{diag_script}"')
        if ret == 0: print("     [08] OK")
        else: print(f"     [08] HATA: donus kodu {ret}")
    
    return {"status": "ok", "file": str(REPORT)}

if __name__ == "__main__":
    print(run())
