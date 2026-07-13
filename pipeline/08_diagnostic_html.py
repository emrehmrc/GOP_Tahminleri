"""
08_diagnostic_html.py — ADM interaktif diagnostic (7 sekme, Chart.js)
Ince kabuk: ADM kolonlarini kanonik isimlere map eder ve
src/diagnostic_core.py motorunu cagirir. Tum hesap+render orada
(GDZ ile birebir ayni motor → iki EDAS asla ayrismaz).

Pipeline sozlesmesi: run() -> dict (run_daily / UI import ederek cagirir).
Standalone: python pipeline/08_diagnostic_html.py
"""
import sys, re, json
import pandas as pd, numpy as np
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C
import diagnostic_core as core
from diagnostic_support import load_hourly_performance, load_model_signals
from src.output_paths import dated_output_path, glob_output_files, resolve_output_file, DELIVERY_ROOT

TEMP_COL = "MUGLA_MenteseCenter_app_temp_actual"
GHI_COL = "GHI_ADM_Weighted"
EDAS = "ADM"


def _mean_columns(frame, contains, excludes=()):
    columns = [c for c in frame.columns
               if all(token.lower() in c.lower() for token in contains)
               and not any(token.lower() in c.lower() for token in excludes)]
    return frame[columns].apply(pd.to_numeric, errors='coerce').mean(axis=1) if columns else pd.Series(np.nan, index=frame.index)


def _aggregate_weather(frame, forecast=False):
    suffix = "_fc" if forecast else "_actual"
    out = pd.DataFrame(index=frame.index)
    out['temp'] = _mean_columns(frame, ('app_temp', suffix))
    preferred_ghi = f"GHI_ADM_Weighted{suffix if forecast else ''}"
    out['ghi'] = (pd.to_numeric(frame[preferred_ghi], errors='coerce') if preferred_ghi in frame
                  else _mean_columns(frame, ('GHI', suffix)))
    out['cloud'] = _mean_columns(frame, ('cloud', suffix))
    out['humidity'] = _mean_columns(frame, ('humidity', suffix))
    out['wind'] = _mean_columns(frame, ('wind_speed', suffix), excludes=('gust',))
    out['precip'] = _mean_columns(frame, ('precip', suffix))
    return out


def run():
    # ── En guncel forecast tarihi ─────────────────────────────────────
    # DELIVERY_ROOT ADM+GDZ ortak; ADM dosya adi "ADM_forecast_..." ile baslar
    # (bkz. config_live.OUTPUT_FILENAME_TEMPLATE) — GDZ'nin "GDZ_forecast_..."
    # dosyalariyla asla karismaz.
    fc_files = [f for f in glob_output_files(DELIVERY_ROOT, C.OUTPUT_FILENAME_TEMPLATE.format(date="*"))
                if '_REGEN' not in f.name]
    fc_date = None
    for p in sorted(fc_files, reverse=True):
        m = re.search(r'(\d{4}-\d{2}-\d{2})', p.name)
        if m: fc_date = m.group(1); break
    if not fc_date: fc_date = str(date.today())
    TODAY = date.fromisoformat(fc_date)
    print(f"[{EDAS}] Diagnostic: {fc_date}")

    # ── Master + hava gecmisi → kanonik merged ───────────────────────
    m = pd.read_parquet(C.MASTER_PARQUET)
    m[C.RAW_DATE_COL] = pd.to_datetime(m[C.RAW_DATE_COL])
    wh = pd.read_parquet(C.WEATHER_HISTORY_PARQUET)
    wx_in_m = [c for c in m.columns if c in wh.columns and c not in (C.RAW_DATE_COL, C.RAW_HOUR_COL)]
    if wx_in_m: m = m.drop(columns=wx_in_m)
    weather = _aggregate_weather(wh, forecast=False)
    weather[C.RAW_DATE_COL] = pd.to_datetime(wh[C.RAW_DATE_COL])
    weather[C.RAW_HOUR_COL] = wh[C.RAW_HOUR_COL].astype(int)
    merged = m.merge(weather, on=[C.RAW_DATE_COL, C.RAW_HOUR_COL], how='left')

    can = pd.DataFrame({
        'dt': merged[C.RAW_DATE_COL].dt.normalize(),
        'h': merged[C.RAW_HOUR_COL].astype(int),
        'load': merged[C.RAW_TARGET_COL],
        'temp': merged['temp'], 'ghi': merged['ghi'],
        'cloud': merged['cloud'], 'humidity': merged['humidity'],
        'wind': merged['wind'], 'precip': merged['precip'],
    })
    if 'ÖzelGün_Adı' in merged.columns: can['special'] = merged['ÖzelGün_Adı']
    print(f"  Merged: {len(can)} satir")

    # ── FC (D+2 tahmini) ─────────────────────────────────────────────
    fc = None
    fp = resolve_output_file(DELIVERY_ROOT, C.OUTPUT_FILENAME_TEMPLATE.format(date=fc_date))
    if fp.exists():
        try:
            fc = pd.read_excel(fp, sheet_name="Tahmin").set_index("Saat")["Tahmin_MWh"].reindex(range(24)).tolist()
        except Exception as e:
            print(f"  FC uyari: {e}")

    # ── FC hava (tahmin sicaklik/GHI) ────────────────────────────────
    fc_wx = {}
    try:
        wfc = pd.read_parquet(C.DATA_DIR / "weather_cache" / "weather_fc_live.parquet")
        wfc[C.RAW_DATE_COL] = pd.to_datetime(wfc["Tarih"]).dt.normalize()
        tw = wfc[wfc[C.RAW_DATE_COL].dt.date == TODAY].set_index(C.RAW_HOUR_COL).sort_index()
        if len(tw) > 12:
            aggregated = _aggregate_weather(tw, forecast=True).reindex(range(24))
            for key in ('temp', 'ghi', 'cloud', 'humidity', 'wind', 'precip'):
                if aggregated[key].notna().any():
                    fc_wx[key] = aggregated[key].tolist()
            if fc_wx: print(f"  WX FC: {len(tw)} saat")
    except Exception as e:
        print(f"  WX FC uyari: {e}")

    # ── Son gunler MAPE (teslim tahmin vs gerceklesme) ───────────────
    last7 = []
    for i in range(1, 15):
        ds = str(TODAY - timedelta(days=i))
        fp = resolve_output_file(DELIVERY_ROOT, C.OUTPUT_FILENAME_TEMPLATE.format(date=ds))
        if not fp.exists(): continue
        try:
            pred = pd.read_excel(fp, sheet_name="Tahmin").set_index("Saat")["Tahmin_MWh"]
            key = pd.Timestamp(ds).normalize()
            actual_day = (can[can['dt'] == key].dropna(subset=['load'])
                          .groupby('h')['load'].last())
            av = np.array([actual_day.get(h, np.nan) for h in range(24)], dtype=float)
            fv = np.array([pred.get(h, np.nan) for h in range(24)])
            msk = (av > 0) & ~np.isnan(fv)
            if msk.sum() > 12:
                last7.append([ds, round(float(np.mean(np.abs((av[msk] - fv[msk]) / av[msk])) * 100), 2)])
        except Exception:
            pass
    last7.reverse()

    # ── Compute + render ─────────────────────────────────────────────
    model_signals = load_model_signals(C.FORECAST_LOG_DIR, EDAS, fc_date)
    hourly_performance = load_hourly_performance(
        C.FORECAST_LOG_DIR, C.ACTUALS_LOG_DIR, EDAS, fc_date,
    )
    data = core.compute(
        can, fc, fc_wx, fc_date, EDAS,
        model_signals=model_signals, hourly_performance=hourly_performance,
    )
    html = core.render(data, "ADM STLF DIAGNOSTIC", last7=last7)
    OUT = dated_output_path(DELIVERY_ROOT, fc_date, f"diagnostic_{fc_date}.html", create=True)
    OUT.write_text(html, 'utf-8')
    DATA_OUT = dated_output_path(DELIVERY_ROOT, fc_date, f"diagnostic_{fc_date}.json", create=True)
    DATA_OUT.write_text(json.dumps({**data, "L7": last7}, ensure_ascii=False, indent=2), encoding="utf-8")
    rec_n = sum(1 for r in data['REC'] if r['exp'] is not None)
    print(f"SAVED: {OUT} ({len(html)} bytes) | CP={len(data['CP'])} SN={len(data['SN'])} REC={rec_n} L7={len(last7)}")
    return {"status": "ok", "file": str(OUT), "data_file": str(DATA_OUT), "date": fc_date,
            "cp": len(data['CP']), "rec": rec_n, "mape_days": len(last7)}


if __name__ == "__main__":
    run()
