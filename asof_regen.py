"""asof_regen.py — Eski (pre-fix) teslim dosyalarını GÜNCEL pipeline ile "geçmişteymişiz
gibi" yeniden üret (manuel, bir kez tetiklenir).
====================================================================================
NEDEN: 07-02 ve 07-04 teslimleri, pipeline'ın Faz -1 öncesi döneminde üretilmiş; o
dönemde Chronos her run'da sessizce XGB'ye düşüyor ve T+2 günü düz çizgiye çöküyordu
(bkz. oturum teşhisi). Güncel pipeline bunu düzeltti. Bu script, master'ı issue
tarihine KESİP güncel kod + modellerle o günleri yeniden tahmin eder.

YAKLAŞIKLIK (şeffaf): Orijinal run'ların forecast havası kaybolmuş; ufuk havası
weather_history'deki GERÇEKLEŞEN (reanalysis) havadan kurulur = "perfect-prog" hava.
Yani sonuç, o gün üretilecek FAİTHFUL tahminle birebir aynı değildir; MODEL davranışının
(düz T+2 vs sağlıklı T+2) düzeldiğini gösterir ve makul düzeltilmiş sayı verir.

GÜVENLİK: Canlı model dosyaları + master + weather_fc + ara parquet'ler çalışmadan önce
yedeklenir, SONUNDA (hata olsa da) geri yüklenir. Çıktı ayrı *_REGEN.xlsx'e yazılır;
orijinal teslim dosyasına DOKUNULMAZ. monitoring/forecast_log'a yazmaz (03-06 doğrudan).

Kullanım:
    python asof_regen.py            # 07-02 ve 07-04'ü üretir
"""
from __future__ import annotations
import sys, os, shutil, importlib, glob
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C

WC = C.DATA_DIR / "weather_cache"
# Yedeklenecek canlı durum dosyaları
LIVE_FILES = [
    C.MASTER_PARQUET, C.WEATHER_FC_PARQUET, WC / "feature_matrix.parquet",
    WC / "raw_predictions.parquet", WC / "postprocessed_predictions.parquet",
    WC / "raw_predictions_meta.json", C.OOF_HISTORY_PATH,
]
MODEL_GLOBS = ["live_*.json", "live_*.txt", "stacking_ridge.joblib", "live_catboost.cbm"]

BACKUP = WC / "_asof_backup"


def _backup():
    BACKUP.mkdir(parents=True, exist_ok=True)
    for f in LIVE_FILES:
        if f.exists(): shutil.copy2(f, BACKUP / f.name)
    (BACKUP / "models").mkdir(exist_ok=True)
    for g in MODEL_GLOBS:
        for f in C.MODELS_DIR.glob(g):
            shutil.copy2(f, BACKUP / "models" / f.name)
    # ORİJİNAL teslim dosyaları: 06 bunların üstüne yazacak — koru, sonra geri yükle.
    # *_REGEN.xlsx'leri YEDEKLEME: bu run'ın ürettiği yeni REGEN'i geri-yükleme ezmesin.
    (BACKUP / "output").mkdir(exist_ok=True)
    for f in C.OUTPUT_DIR.glob("*.xlsx"):
        if f.name.endswith("_REGEN.xlsx"):
            continue
        shutil.copy2(f, BACKUP / "output" / f.name)
    print(f"[yedek] {BACKUP}")


def _restore():
    for f in LIVE_FILES:
        b = BACKUP / f.name
        if b.exists(): shutil.copy2(b, f)
        elif f.exists(): f.unlink()  # regen'in yarattığı, orijinalde olmayan dosyayı sil
    for g in MODEL_GLOBS:
        for f in C.MODELS_DIR.glob(g):
            b = BACKUP / "models" / f.name
            if b.exists(): shutil.copy2(b, f)
    # orijinal teslim dosyalarını geri koy (06'nın üstüne yazdıklarını geri al).
    # *_REGEN.xlsx'e ASLA dokunma — bu run'ın ürettiği yeni REGEN korunmalı.
    original_names = {b.name for b in (BACKUP / "output").glob("*.xlsx")}
    for b in (BACKUP / "output").glob("*.xlsx"):
        shutil.copy2(b, C.OUTPUT_DIR / b.name)
    # backtest'in ürettiği, backup ANINDA var OLMAYAN teslim dosyalarını temizle
    # (ör. 7-gün backtest'te as-of hedefler için 06_deliver gerçek dosya adına
    # yazıyor — bunlar orijinal teslim değil, backtest kalıntısı; sızıp production
    # output/ klasörünü kirletmesin).
    for f in C.OUTPUT_DIR.glob("*.xlsx"):
        if f.name.endswith("_REGEN.xlsx"):
            continue
        if f.name not in original_names:
            f.unlink()
            print(f"     [temizlik] backtest kalıntısı silindi: {f.name}")
    print("[geri yükle] canlı durum + orijinal teslim dosyaları eski haline döndü")


def _build_synth_fc(horizon_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """weather_history'deki gerçekleşen havadan, weather_fc_live şemasında 3-günlük
    (72h) synthetic forecast kur. 03'ün merge'inin kullandığı kolonlar app_temp/precip/
    cloud/Dark_Fraction/GHI_* — hepsi history'de var. Eksik gün (ör. 07-03) komşu
    günlerden saat-bazında lineer interpolasyonla doldurulur."""
    fc_template = pd.read_parquet(C.WEATHER_FC_PARQUET)
    wh = pd.read_parquet(C.WEATHER_HISTORY_PARQUET)
    wh[C.RAW_DATE_COL] = pd.to_datetime(wh[C.RAW_DATE_COL])

    def hist_col_for(fccol):
        for suf in ("app_temp", "precip", "cloud"):
            if fccol.endswith(f"{suf}_fc"):
                cand = fccol.replace(f"{suf}_fc", f"{suf}_actual")
                return cand if cand in wh.columns else None
        if fccol.endswith("_fc"):
            base = fccol[:-3]
            return base if base in wh.columns else None
        return None

    rows = []
    for d in horizon_dates:
        for h in range(24):
            rows.append({"Tarih": pd.Timestamp(d), "Saat": h})
    out = pd.DataFrame(rows)

    # her fc kolonu için history'den değer çek (gün+saat), eksik günü interpole et
    wh_idx = wh.set_index([C.RAW_DATE_COL, C.RAW_HOUR_COL])
    for c in fc_template.columns:
        if c in ("Tarih", "Saat"):
            continue
        hc = hist_col_for(c)
        vals = []
        for d in horizon_dates:
            for h in range(24):
                v = np.nan
                if hc is not None:
                    try: v = wh_idx.loc[(pd.Timestamp(d), h), hc]
                    except KeyError: v = np.nan
                vals.append(v)
        out[c] = vals
    # eksik gün (tüm-NaN 24h blok) -> zaman-ekseninde interpolasyon + uçları doldur
    for c in out.columns:
        if c in ("Tarih", "Saat"): continue
        out[c] = out[c].interpolate(limit_direction="both")
    return out


def _horizon_for(target_date: str) -> tuple[str, list[pd.Timestamp]]:
    """target (teslim=T+2) için: 3-günlük blok = [T-2(T+0), T-1(T+1), target(T+2)].
    master, blok başından bir gün öncesine (last actual) kesilir."""
    tgt = pd.Timestamp(target_date)
    t0 = tgt - pd.Timedelta(days=2)   # T+0 (teslim edilmez)
    t1 = tgt - pd.Timedelta(days=1)   # T+1
    last_actual_date = (t0 - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return last_actual_date, [t0, t1, tgt]


def regen_one(target_date: str) -> dict:
    last_actual_date, horizon = _horizon_for(target_date)
    print(f"\n=== REGEN {target_date}  (last_actual<={last_actual_date}, ufuk={[d.date() for d in horizon]}) ===")

    # 1) master'ı kes
    master = pd.read_parquet(BACKUP / C.MASTER_PARQUET.name)
    master[C.RAW_DATE_COL] = pd.to_datetime(master[C.RAW_DATE_COL])
    trunc = master[master[C.RAW_DATE_COL] <= pd.Timestamp(last_actual_date)].copy()
    trunc.to_parquet(C.MASTER_PARQUET, index=False)
    print(f"     master kesildi: {len(trunc)} satir (son {trunc[C.RAW_DATE_COL].max().date()})")

    # 1b) OOF geçmişini de aynı cutoff'a kes (aksi halde Rolling Ridge, backtest
    # edilen as-of noktasının GELECEĞİNDEKİ gerçek OOF günlerini görür — sızıntı).
    if C.OOF_HISTORY_PATH.exists():
        oof_bak = BACKUP / C.OOF_HISTORY_PATH.name
        if oof_bak.exists():
            oof = pd.read_parquet(oof_bak)
            oof["date"] = pd.to_datetime(oof["date"])
            oof_trunc = oof[oof["date"] <= pd.Timestamp(last_actual_date)].copy()
            oof_trunc.to_parquet(C.OOF_HISTORY_PATH, index=False)
            print(f"     OOF kesildi: {len(oof_trunc)} satir")

    # 2) synthetic weather_fc
    synth = _build_synth_fc(horizon)
    synth.to_parquet(C.WEATHER_FC_PARQUET, index=False)
    print(f"     synth weather_fc: {len(synth)} satir")

    # 3) 03 -> 04 -> 05 -> 06
    arch_before = set(glob.glob(str(C.ARCHIVE_DIR / "*.parquet")))
    for modname in ["03_build_features", "04_predict_48h", "05_postprocess", "06_deliver"]:
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"p{modname}", str(ROOT / "pipeline" / f"{modname}.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        if modname == "06_deliver":
            res = mod.run(target_date=target_date)
        else:
            res = mod.run()
        print(f"     [{modname}] {res.get('status')}")

    # 4) çıktıyı REGEN'e taşı, 06'nın yarattığı archive parquet'ini temizle
    orig_out = C.OUTPUT_DIR / C.OUTPUT_FILENAME_TEMPLATE.format(date=target_date)
    regen_out = C.OUTPUT_DIR / f"{target_date}_forecast_REGEN.xlsx"
    if orig_out.exists():
        # 06 orijinalin üstüne yazdı; regen'i REGEN'e KOPYALA (orijinal _restore ile
        # yedekten geri gelecek).
        shutil.copy2(str(orig_out), str(regen_out))

    # 4b) model-bazlı ayrıntı (XGB/LGBM/CAT/CHRONOS/Ensemble/Final) — analiz için,
    # 05'in ürettiği postprocessed_predictions.parquet bir sonraki hedefte
    # ezilmeden önce ayrı bir dosyaya kopyala.
    models_out = C.OUTPUT_DIR / f"{target_date}_models_REGEN.parquet"
    postproc_path = C.DATA_DIR / "weather_cache" / "postprocessed_predictions.parquet"
    if postproc_path.exists():
        shutil.copy2(str(postproc_path), str(models_out))
    for f in set(glob.glob(str(C.ARCHIVE_DIR / "*.parquet"))) - arch_before:
        Path(f).unlink(missing_ok=True)  # regen'in kirlettiği archive dosyasını sil

    # 5) sonucu incele
    df = pd.read_excel(regen_out, sheet_name="Tahmin")
    xgb_std = df["Tahmin_MWh"].std()
    return {"target": target_date, "regen_file": regen_out.name, "n": len(df),
            "min": round(df["Tahmin_MWh"].min(), 1), "max": round(df["Tahmin_MWh"].max(), 1),
            "std": round(xgb_std, 1), "mean": round(df["Tahmin_MWh"].mean(), 1)}


if __name__ == "__main__":
    if sys.argv[1:2] == ["--restore-only"]:
        # Kurtarma: süreç dış etkenle (oturum kopması vb.) öldürülüp finally hiç
        # çalışmadıysa, BACKUP hâlâ diskte duruyorsa bununla manuel kurtar.
        if not BACKUP.exists():
            print("BACKUP yok — kurtaracak bir şey bulunamadı."); sys.exit(1)
        _restore()
        sys.exit(0)

    targets = sys.argv[1:] or ["2026-07-02", "2026-07-04"]
    _backup()
    results = []
    try:
        for t in targets:
            results.append(regen_one(t))
    finally:
        _restore()
    print("\n=== REGEN SONUÇ ===")
    for r in results:
        print(r)
