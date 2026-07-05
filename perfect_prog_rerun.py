"""
perfect_prog_rerun.py — Faz 1 v1: Model vs Meteoroloji Ayrıştırma (manuel tetiklenen)
========================================================================================
Tasarım: stlf_faz1_scorecard_tasarim.md §4.

Amaç: bir teslim gününde hatanın ne kadarı MODEL'den, ne kadarı METEOROLOJİ tahmin
hatasından geldiğini ayırt etmek. weather_history.parquet gerçekleşen (reanalysis)
havayı tutuyor (~D+6 gecikmeyle); bu gerçek havayı canlı run'ın kullandığı forecast
havanın YERİNE koyup aynı modellerle yeniden tahmin üretiyoruz.

GÜVENLİK KISITI (kritik): pipeline/04_predict_48h.py:run() GBDT modellerini HER
ÇAĞRILDIĞINDA yeniden eğitip canlı model dosyalarının (MODEL_XGB_PATH vb.) ÜZERİNE
YAZIYOR. Bu script o fonksiyonu ASLA çağırmaz — sadece models/archive/<run_id>/
altındaki (o günü üreten, donmuş) model dosyalarını `.predict()` için yükler.
Yeniden eğitim YOK, canlı modellere dokunma YOK.

Bilinen yaklaşıklıklar (v1):
  - T+2 lag-zinciri (_recompute_lags_for_t2), canlı run'ın ORİJİNAL T+1 tahminleriyle
    kurulur, yeniden hesaplanmaz — yalnızca hava etkisi izole edilir. T+1 için bu
    yaklaşıklık YOK: T+1 lag feature'ları her zaman gerçek geçmiş yüke dayanır.
  - Ensemble = arşivlenmiş GBDT modellerinin (XGB+LGBM+CAT varsa) BASİT ORTALAMASI.
    Chronos-2 (LoRA inference, ağır/yavaş) ve gerçek meta-stacker (rolling ridge,
    ayrı bir OOF geçmişi ister) v1'de YOK. Sonuç: `perfect_prog_mape` ile canlı
    `live_mape_final` (tam 4-model + stacker + corrector zinciri) DOĞRUDAN eşit
    şartlarda kıyaslanamaz — `weather_attribution_pp` bu yüzden bir ÜST SINIR/kaba
    işaret olarak okunmalı, kesin sayı değil. Tam sadakat (Chronos + stacker dahil)
    Faz 4 backlog.

Kullanım:
    python perfect_prog_rerun.py 2026-07-01
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config_live as C
from src.forecast_logger import compute_calendar_fields, WEATHER_STATION_TEMP_COLS

WX_TEMP_COLS = WEATHER_STATION_TEMP_COLS


def _find_run_id_for_target_date(target_date: str) -> str:
    """forecast_log'dan o target_date'i teslim eden (en güncel issue_ts) run_id'yi bul."""
    con = duckdb.connect(str(C.MONITORING_DB), read_only=True)
    try:
        row = con.execute(
            "SELECT run_id FROM forecast_log_v WHERE target_date = ? "
            "ORDER BY issue_ts DESC LIMIT 1",
            [target_date],
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise ValueError(f"forecast_log'da {target_date} için satır yok — önce build_daily_scorecard/backfill çalıştırılmalı.")
    return row[0]


def _load_archived_feature_matrix(run_id: str) -> pd.DataFrame:
    path = C.MODEL_ARCHIVE_DIR / run_id / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Arşivlenmiş feature_matrix yok: {path}. "
            "archive_models() feature_snapshot_ref'i sadece Faz -1 sonrası run'lar için tutuyor — "
            "eski run'larda perfect-prog rerun mümkün değil."
        )
    df = pd.read_parquet(path)
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]
    return df


def _substitute_actual_weather(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """feature_matrix'in gelecek satırlarındaki forecast hava kolonlarını (`*_app_temp_actual`,
    `GHI_ADM_Weighted` — 03'ün uniform şeması gereği forecast de bu adlarla tutuluyor)
    weather_history.parquet'teki GERÇEKLEŞEN değerle değiştirir. Sadece o saat için
    reanalysis DOLMUŞSA (~D+6) değiştirilir; boşsa orijinal forecast kalır."""
    if not C.WEATHER_HISTORY_PARQUET.exists():
        raise FileNotFoundError("weather_history.parquet yok.")

    wh = pd.read_parquet(C.WEATHER_HISTORY_PARQUET)
    wh[C.RAW_DATE_COL] = pd.to_datetime(wh[C.RAW_DATE_COL])
    wh_ts = wh[C.RAW_DATE_COL] + pd.to_timedelta(wh[C.RAW_HOUR_COL], unit="h")
    wh = wh.set_index(wh_ts)

    df = feature_df.copy()
    temp_cols = [c for c in WX_TEMP_COLS if c in df.columns and c in wh.columns]
    has_ghi = "GHI_ADM_Weighted" in df.columns and "GHI_ADM_Weighted" in wh.columns

    common_ts = df.index.intersection(wh.index)
    n_replaced = 0
    for c in temp_cols:
        valid = wh.loc[common_ts, c].dropna()
        df.loc[valid.index, c] = valid
        n_replaced = max(n_replaced, len(valid))
    if has_ghi:
        valid = wh.loc[common_ts, "GHI_ADM_Weighted"].dropna()
        df.loc[valid.index, "GHI_ADM_Weighted"] = valid
        n_replaced = max(n_replaced, len(valid))

    return df, n_replaced


def _get_feature_cols(feature_df: pd.DataFrame) -> list:
    cols = feature_df.select_dtypes(include=["number", "category", "bool"]).columns.tolist()
    if C.RAW_TARGET_COL in cols:
        cols.remove(C.RAW_TARGET_COL)
    return cols


def _t1_delivery_index(feature_df: pd.DataFrame):
    """04_predict_48h.py:split_train_predict ile birebir aynı mantık: NaN bloğu 72
    ise ilk 24 satır TESLİM EDİLMEYEN 'T+0 valid-lag' adımıdır — gerçek T+1 teslim
    penceresi steps[1] (`all_nan[24:48]`). Bunu `all_nan[:24]` sanmak T+0'ı T+1 olarak
    işlemek olurdu (bulunup düzeltildi)."""
    all_nan = feature_df.index[feature_df[C.RAW_TARGET_COL].isna()]
    n_nan = len(all_nan)
    if n_nan >= 72:
        return all_nan[24:48]
    if n_nan >= 48:
        return all_nan[:24]
    return all_nan[:24] if n_nan >= 24 else None


def _lgbm_booster_from_file(path: Path):
    """`lgb.Booster(model_file=path)` doğrudan verildiğinde arşivlenmiş dosyalarda
    'Model format error, expect a tree here' ile patlıyor (bulunup teşhis edildi):
    src/lightgbm_manager.py:save_model, booster metnini Python text-mode `open(path,'w')`
    ile yazıyor — Windows'ta bu `\\n`'i `\\r\\n`'e çeviriyor ve LightGBM'in kendi metin
    formatı satır sonlarına duyarlı olduğu için parser'ı bozuyor. Production bunu hiç
    fark etmedi çünkü canlı `04` HİÇBİR ZAMAN kayıtlı bir LGBM modelini geri yüklemiyor
    (her gün sıfırdan eğitiyor) — bu script arşivden yükleyen ilk kod yolu. Kalıcı
    düzeltme `lightgbm_manager.py`'de (Faz 1 kapsamı dışı, ayrıca not edilecek); burada
    salt-okunur bir normalize-edip-geçici-dosyaya-yaz atlatması yeterli."""
    import lightgbm as lgb
    import tempfile

    raw = path.read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, newline="\n", encoding="utf-8") as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        return lgb.Booster(model_file=tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _load_lgbm_predictor(path: Path):
    """LightGBMManager.save_model, XGB'deki HybridXGBModel ile aynı desende
    (bkz. src/lightgbm_manager.py:HybridLightGBMModel) hafta-içi/hafta-sonu ayrı
    booster'ları bir 'HYBRID_LGBM_SPLIT' pointer dosyasına yazabiliyor. LightGBMManager'ın
    kendisi load_model sunmuyor (yalnızca train/save) — bu yüzden pointer'ı burada
    elle çözüyoruz."""
    from src.lightgbm_manager import HybridLightGBMModel

    first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
    if first_line == "HYBRID_LGBM_SPLIT":
        lines = path.read_text(encoding="utf-8").splitlines()
        wd_sat_file = lines[1].split(": ")[1]
        we_file = lines[2].split(": ")[1]
        model_wd_sat = _lgbm_booster_from_file(path.parent / wd_sat_file)
        model_we = _lgbm_booster_from_file(path.parent / we_file)
        return HybridLightGBMModel(model_wd_sat, model_we)
    return _lgbm_booster_from_file(path)


def _predict_t1_archived(feature_df: pd.DataFrame, t1_idx, feature_cols: list, run_id: str) -> dict:
    """T+1: recursive zincir gerekmez (lag'ler hep gerçek geçmiş) — arşivlenmiş
    (donmuş, bu script tarafından ASLA yeniden eğitilmeyen) modellerle doğrudan tahmin."""
    archive_dir = C.MODEL_ARCHIVE_DIR / run_id
    X = feature_df.loc[t1_idx, feature_cols]
    preds = {}

    xgb_path = archive_dir / C.MODEL_XGB_PATH.name
    if xgb_path.exists():
        from src.model_manager import ModelManager
        mm = ModelManager()
        mm.load_model(str(xgb_path))
        preds["XGB_Pred"] = pd.Series(mm.model.predict(X), index=t1_idx)

    lgbm_path = archive_dir / C.MODEL_LGBM_PATH.name
    if lgbm_path.exists():
        predictor = _load_lgbm_predictor(lgbm_path)
        preds["LGBM_Pred"] = pd.Series(predictor.predict(X), index=t1_idx)

    cat_path = archive_dir / "live_catboost.cbm"
    if cat_path.exists():
        from src.catboost_manager import CatBoostManager
        cat = CatBoostManager()
        cat.load_model(str(cat_path))
        preds["CAT_Pred"] = pd.Series(cat.model.predict(X), index=t1_idx)

    return preds


def run_perfect_prog_rerun(target_date: str) -> dict:
    run_id = _find_run_id_for_target_date(target_date)
    feature_df = _load_archived_feature_matrix(run_id)
    feature_df_pp, n_wx_replaced = _substitute_actual_weather(feature_df)

    if n_wx_replaced == 0:
        return {
            "status": "no_weather_actuals_yet",
            "note": "weather_history'de bu tarih için henüz reanalysis (~D+6) dolmamış.",
        }

    feature_cols = _get_feature_cols(feature_df_pp)
    t1_idx = _t1_delivery_index(feature_df_pp)
    if t1_idx is None:
        return {"status": "insufficient_future_rows"}

    recon_datetime = pd.DatetimeIndex(
        feature_df_pp.loc[t1_idx, C.RAW_DATE_COL].values
        + pd.to_timedelta(feature_df_pp.loc[t1_idx, C.RAW_HOUR_COL].values, unit="h")
    )
    is_t1_target_date = recon_datetime.strftime("%Y-%m-%d") == target_date
    if not is_t1_target_date.any():
        return {
            "status": "target_date_not_in_t1_window",
            "note": "v1 sadece T+1 penceresini (recursive zincirsiz) destekliyor; "
                    f"bu run'ın T+1 penceresi {target_date}'i kapsamıyor.",
        }

    preds = _predict_t1_archived(feature_df_pp, t1_idx, feature_cols, run_id)
    if not preds:
        return {"status": "no_archived_models", "archive_dir": str(C.MODEL_ARCHIVE_DIR / run_id)}

    perfect_ensemble = pd.DataFrame(preds).mean(axis=1)

    con = duckdb.connect(str(C.MONITORING_DB), read_only=True)
    try:
        actual_rows = con.execute(
            "SELECT target_ts, y_actual FROM actuals_log_v WHERE edas_id = ? AND target_date = ?",
            [C.EDAS_ID, target_date],
        ).df()
        live_rows = con.execute(
            "SELECT target_ts, y_pred_final FROM forecast_log_v "
            "WHERE edas_id = ? AND target_date = ? AND horizon_day = 'T+1' AND run_id = ?",
            [C.EDAS_ID, target_date, run_id],
        ).df()
    finally:
        con.close()

    if actual_rows.empty:
        return {"status": "no_actuals_yet"}

    actual_rows["target_ts"] = pd.to_datetime(actual_rows["target_ts"])
    actual = actual_rows.set_index("target_ts")["y_actual"]

    perfect_ape = np.abs((actual.reindex(recon_datetime).to_numpy() - perfect_ensemble.to_numpy())
                          / (actual.reindex(recon_datetime).to_numpy() + 1e-10)) * 100
    perfect_mape = float(np.nanmean(perfect_ape))

    live_mape = None
    if not live_rows.empty:
        live_rows["target_ts"] = pd.to_datetime(live_rows["target_ts"])
        live = live_rows.set_index("target_ts")["y_pred_final"].reindex(recon_datetime)
        a = actual.reindex(recon_datetime).to_numpy()
        live_ape = np.abs((a - live.to_numpy()) / (a + 1e-10)) * 100
        live_mape = float(np.nanmean(live_ape))

    result = {
        "status": "ok",
        "run_id": run_id,
        "target_date": target_date,
        "n_hours": int(is_t1_target_date.sum()),
        "n_weather_actuals_replaced": n_wx_replaced,
        "perfect_prog_mape": round(perfect_mape, 2),
        "live_mape_final": round(live_mape, 2) if live_mape is not None else None,
        "weather_attribution_pp": (
            round(live_mape - perfect_mape, 2) if live_mape is not None else None
        ),
        "ensemble_method": "simple_mean (GBDT only, Chronos/stacker YOK — v1 kısıtı)",
        "note": "weather_attribution_pp = live_mape - perfect_mape; KABA bir üst-sınır işaretidir, "
                "kesin sayı değil (bkz. dosya başlığı 'Bilinen yaklaşıklıklar'). Sadece T+1 penceresi; "
                "T+2 v1 kapsamı dışında.",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perfect-prog rerun (Faz 1 v1, sadece T+1)")
    parser.add_argument("target_date", help="YYYY-MM-DD")
    args = parser.parse_args()
    print(json.dumps(run_perfect_prog_rerun(args.target_date), ensure_ascii=False, indent=2, default=str))
