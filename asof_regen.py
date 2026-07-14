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

GÜVENLİK (2026-07-10 yeniden yazıldı — bkz. gerekçe aşağıda): 03→04→05→06 adımları
GERÇEK master/weather_fc/model/output dosyalarına DOKUNMAZ. config_live modül
niteliklerinin (DATA_DIR, MASTER_PARQUET, WEATHER_FC_PARQUET, OOF_HISTORY_PATH,
MODELS_DIR + günlük-yeniden-eğitilen model yolları, OUTPUT_DIR, ARCHIVE_DIR) bu
script'in ömrü boyunca izole bir `_asof_sandbox/` alt ağacına yönlendirilmesiyle
çalışır (pipeline modülleri asof_regen tarafından HER regen_one() çağrısında
importlib ile taze exec edildiği için `from config_live import X` satırları o an
config_live üzerinde ne varsa onu okur). Donmuş/salt-okunur kaynaklar (weather_history,
Chronos adapter, stacking_ridge.joblib, holiday/pv-bias kalibrasyonları) HİÇ
yönlendirilmez — gerçek dosyadan okunmaya devam eder (mutasyona uğramadıkları için
izolasyona gerek yok). Sonuç: gerçek master/model/teslim dosyaları bu script
çalışırken ASLA açılıp üzerine yazılmaz — yarıda kesilse (oturum kopması vb.) bile
canlı durum bozulmaz, "restore" adımına hiç ihtiyaç kalmaz.

ESKİ TASARIM (kaldırıldı): önceki sürüm gerçek master.parquet'i YERİNDE kesip
gerçek model/teslim dosyalarının üzerine yazıyor, sonunda `finally` bloğuyla
yedekten geri yüklüyordu. Bu restore adımı en az iki gerçek olayda güvenilmez
çıktı: 2026-07-06 master.parquet korupsiyonu ve 2026-07-08'de ADM'nin bir günlük
(T+2=07-10) forecast_log/arşiv kaybı — archregen sırasında incelenen arşiv dosyası
gerçek 07-09/07-10 içeriği yerine bayat 07-07/07-08 içeriği taşıyordu, restore'un
gerçek durumu doğru geri yüklemediğinin izi. Sandbox izolasyonu bu risk sınıfını
kökten kapatır (mutasyon hiç gerçek dosyaya gitmediği için geri yüklenecek bir şey
de yok).

Kullanım:
    python asof_regen.py            # 07-02 ve 07-04'ü üretir
    python asof_regen.py 2026-07-01 2026-07-03
"""
from __future__ import annotations
import sys
import shutil
import time
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config_live as C
from src.output_paths import dated_output_path

SANDBOX = ROOT / "data" / "_asof_sandbox"   # git-ignored, tek kullanımlık izole alan


def _rmtree_retry(path: Path, attempts: int = 8, delay_s: float = 1.5) -> None:
    """shutil.rmtree wrapper — OneDrive senkron kilidi, klasör silme/oluşturma
    hemen ardından WinError 5 (Erişim engellendi) fırlatabiliyor, bazen 15s+
    sürebiliyor (2026-07-14 F1.A-cal koşusunda gözlemlendi — hem SANDBOX hem
    SANDBOX/delivery/<ay> alt klasöründe, boş klasörler üzerinde). Artan backoff
    ile yeniden dener; TÜM denemeler başarısız olursa BEST-EFFORT: hatayı
    yutar, uyarı basar, devam eder — sandbox git-ignored/tek-kullanımlık
    olduğu için bir sonraki mkdir(exist_ok=True) zaten üzerine yazar, artık
    kalıntı dosyalar (varsa) çalışma doğruluğunu bozmaz, sadece disk çöpü."""
    last_err: OSError | None = None
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError as e:
            last_err = e
            time.sleep(delay_s * (i + 1))
    print(f"     [uyari] sandbox temizligi basarisiz (OneDrive kilidi olasi), devam ediliyor: {last_err}")

# Regen sırasında YAZILAN (mutasyona uğrayan) config_live nitelikleri — bunlar
# sandbox'a yönlendirilir. Geri kalan her şey (frozen kalibrasyon, Chronos adapter,
# weather_history) GERÇEK yoldan okunmaya devam eder; kopyalanmasına bile gerek yok
# çünkü 03/04/05/06 hiçbiri bunların üzerine yazmıyor.
_REDIRECT_ATTRS = [
    "DATA_DIR", "MASTER_PARQUET", "WEATHER_FC_PARQUET", "OOF_HISTORY_PATH",
    "MODELS_DIR", "MODEL_XGB_PATH", "MODEL_LGBM_PATH", "MODEL_LGBM_WD_SAT",
    "MODEL_LGBM_WE", "MODEL_XGB_WD_SAT", "MODEL_XGB_WE",
    "OUTPUT_DIR", "ARCHIVE_DIR", "DELIVERY_ROOT",
]

_originals: dict[str, Path] = {}


def _sandbox_path(real_path: Path) -> Path:
    rel = Path(real_path).relative_to(C.LIVE_DIR)
    return SANDBOX / rel


def _enter_sandbox() -> None:
    """config_live niteliklerini sandbox yollarına yönlendirir. Pipeline modülleri
    (03-06) fresh-exec edildikleri için `from config_live import X` bundan sonraki
    her çağrıda buradaki sandbox değerlerini görür — gerçek dosyalar bu blok
    içinde asla açılmaz.

    DELIVERY_ROOT özel durum: proje dışı paylaşılan müşteri teslim klasörü —
    LIVE_DIR altında DEĞİL, bu yüzden _sandbox_path()'in relative_to() hesabı
    burada patlar. Sabit bir sandbox alt klasörüne yönlendirilir (06_deliver.py
    regen sırasında GERÇEK müşteri klasörüne yazmasın diye — bkz. yukarıdaki
    GÜVENLİK notu)."""
    if SANDBOX.exists():
        _rmtree_retry(SANDBOX)
    for d in ["data/weather_cache", "models", "output/archive"]:
        (SANDBOX / d).mkdir(parents=True, exist_ok=True)

    for name in _REDIRECT_ATTRS:
        real = getattr(C, name)
        _originals[name] = real
        if name == "DELIVERY_ROOT":
            sandboxed = SANDBOX / "delivery"
        else:
            sandboxed = _sandbox_path(real)
            sandboxed.parent.mkdir(parents=True, exist_ok=True)
        setattr(C, name, sandboxed)


def _exit_sandbox() -> None:
    for name, real in _originals.items():
        setattr(C, name, real)
    _originals.clear()


def _build_synth_fc(horizon_dates: list[pd.Timestamp], template_path: Path) -> pd.DataFrame:
    """weather_history'deki gerçekleşen havadan, weather_fc_live şemasında 3-günlük
    (72h) synthetic forecast kur. 03'ün merge'inin kullandığı kolonlar app_temp/precip/
    cloud/Dark_Fraction/GHI_* — hepsi history'de var. Eksik gün (ör. 07-03) komşu
    günlerden saat-bazında lineer interpolasyonla doldurulur.

    `template_path`: GERÇEK weather_fc_live.parquet — sadece kolon şeması için
    okunur (salt-okunur, sandbox'a yönlendirilmeden ÖNCE çağrılmalı)."""
    fc_template = pd.read_parquet(template_path)
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
                    try:
                        v = wh_idx.loc[(pd.Timestamp(d), h), hc]
                    except KeyError:
                        v = np.nan
                vals.append(v)
        out[c] = vals

    # Bütünlük kontrolü (2026-07-11): bir günün TÜMÜ weather_history'de karşılıksız
    # kalırsa (o gün henüz gerçekleşmemiş / backfill gelmemiş), interpolate() alttaki
    # `limit_direction="both"` yüzünden o günü komşu günün son saatinin DÜZ ÇİZGİSİYLE
    # dolduruyor (07-08 archregen olayı — bkz. modül docstring'i). Sessizce geçmesin.
    temp_cols = [c for c in out.columns if c not in ("Tarih", "Saat") and c.endswith("_fc")
                 and hist_col_for(c) is not None]
    for d in horizon_dates:
        day_mask = out["Tarih"] == pd.Timestamp(d)
        if temp_cols and out.loc[day_mask, temp_cols].isna().all(axis=None):
            print(f"     [UYARI] {d.date()} için weather_history'de HİÇ gerçekleşme yok — "
                  f"bu günün sentetik havası komşu günün son saatinin düz çizgisiyle "
                  f"dolduruluyor (interpolate ffill/bfill). Bu günün model hataları "
                  f"GERÇEK hava değil, düz-hava artefaktı yansıtıyor olabilir.")

    for c in out.columns:
        if c in ("Tarih", "Saat"):
            continue
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

    # 1) GERÇEK (henüz yönlendirilmemiş) kaynaklardan oku — salt okunur.
    real_master_path = C.MASTER_PARQUET
    real_weather_fc_path = C.WEATHER_FC_PARQUET
    real_oof_path = C.OOF_HISTORY_PATH
    real_output_dir = C.OUTPUT_DIR

    master = pd.read_parquet(real_master_path)
    master[C.RAW_DATE_COL] = pd.to_datetime(master[C.RAW_DATE_COL])
    trunc = master[master[C.RAW_DATE_COL] <= pd.Timestamp(last_actual_date)].copy()
    print(f"     master kesildi: {len(trunc)} satir (son {trunc[C.RAW_DATE_COL].max().date()})")

    oof_trunc = None
    if real_oof_path.exists():
        oof = pd.read_parquet(real_oof_path)
        oof["date"] = pd.to_datetime(oof["date"])
        oof_trunc = oof[oof["date"] <= pd.Timestamp(last_actual_date)].copy()
        print(f"     OOF kesildi: {len(oof_trunc)} satir")

    synth = _build_synth_fc(horizon, real_weather_fc_path)
    print(f"     synth weather_fc: {len(synth)} satir")

    # 2) SANDBOX'A GİR — bundan sonra config_live.* mutasyona uğrayan yollar
    #    gerçek değil, izole sandbox'ı gösterir.
    _enter_sandbox()
    try:
        trunc.to_parquet(C.MASTER_PARQUET, index=False)
        synth.to_parquet(C.WEATHER_FC_PARQUET, index=False)
        if oof_trunc is not None:
            oof_trunc.to_parquet(C.OOF_HISTORY_PATH, index=False)

        # 3) 03 -> 04 -> 05 -> 06 (fresh-exec: patched config_live'ı görürler)
        for modname in ["03_build_features", "04_predict_48h", "05_postprocess", "06_deliver"]:
            spec = importlib.util.spec_from_file_location(f"p{modname}", str(ROOT / "pipeline" / f"{modname}.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if modname == "06_deliver":
                # issue_date = horizon[0] (T+0, bkz. _horizon_for) — regen'in
                # WALL-CLOCK date.today()'si degil, simule ettigi GERCEK issue
                # gunu. Arsive bununla damgalanmazsa rebuild_forecast_log_
                # from_archives() yanlis (bugunun tarihi) issue_date okur.
                res = mod.run(target_date=target_date, issue_date=horizon[0].date())
            else:
                res = mod.run()
            print(f"     [{modname}] {res.get('status')}")

        # 4) sandbox çıktısını gerçek output/'a *_REGEN olarak kopyala — orijinal
        #    teslim dosyasına (artık DELIVERY_ROOT'ta) hiç dokunulmadı, kopyalamaya
        #    gerek yok. 06_deliver.py sandboxed DELIVERY_ROOT'a dated_output_path
        #    ile yazdığı için burada da aynı fonksiyonla aranır (flat DEĞİL).
        sandbox_out = dated_output_path(C.DELIVERY_ROOT, target_date, C.OUTPUT_FILENAME_TEMPLATE.format(date=target_date))
        regen_out = dated_output_path(real_output_dir, target_date, f"{target_date}_forecast_REGEN.xlsx", create=True)
        if sandbox_out.exists():
            shutil.copy2(str(sandbox_out), str(regen_out))

        sandbox_postproc = C.DATA_DIR / "weather_cache" / "postprocessed_predictions.parquet"
        models_out = dated_output_path(real_output_dir, target_date, f"{target_date}_models_REGEN.parquet", create=True)
        if sandbox_postproc.exists():
            shutil.copy2(str(sandbox_postproc), str(models_out))

        # raw_predictions_meta.json da (meta_method/meta_w_*/chronos_ok/cat_present)
        # kopyalanır — write_forecast_log() bu regen'e ait meta'yı gerçek (bayat)
        # sidecar yerine buradan okuyabilsin diye (bkz. backtest_walkforward.py).
        sandbox_meta = C.DATA_DIR / "weather_cache" / "raw_predictions_meta.json"
        meta_out = dated_output_path(real_output_dir, target_date, f"{target_date}_meta_REGEN.json", create=True)
        if sandbox_meta.exists():
            shutil.copy2(str(sandbox_meta), str(meta_out))
    finally:
        # 5) SANDBOX'TAN ÇIK — config_live gerçek değerlerine döner. Sandbox
        #    klasörü bir sonraki regen_one() çağrısının başında zaten silinir;
        #    burada bırakmak debug için zararsız (gerçek dosya değil).
        _exit_sandbox()

    df = pd.read_excel(regen_out, sheet_name="Tahmin")
    return {"target": target_date, "regen_file": regen_out.name, "n": len(df),
            "min": round(df["Tahmin_MWh"].min(), 1), "max": round(df["Tahmin_MWh"].max(), 1),
            "std": round(df["Tahmin_MWh"].std(), 1), "mean": round(df["Tahmin_MWh"].mean(), 1),
            "models_path": str(models_out) if models_out.exists() else None,
            "meta_path": str(meta_out) if meta_out.exists() else None}


if __name__ == "__main__":
    targets = sys.argv[1:] or ["2026-07-02", "2026-07-04"]
    results = []
    for t in targets:
        results.append(regen_one(t))
    if SANDBOX.exists():
        _rmtree_retry(SANDBOX)
    print("\n=== REGEN SONUÇ ===")
    for r in results:
        print(r)
