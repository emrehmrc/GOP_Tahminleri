"""
forecast_logger.py — Faz 0 Loglama Katmanı
=============================================
Şema kaynağı: stlf_forecast_log_tasarim.md §2 (forecast_log), §3 (actuals_log).
Kilitli kararlar (§6): her düzeltme adımı ayrı delta; known_event CSV; perfect-prog
mümkün (weather_history = gerçekleşme); DuckDB (parquet üzerinde view, türetilmiş);
günlük partition.

Depo: config_live.FORECAST_LOG_DIR / ACTUALS_LOG_DIR (OneDrive DIŞI, %LOCALAPPDATA%).
MONITORING_DB tamamen türetilmiş — silinip parquet'ten rebuild edilebilir.

Çağrı sırası:
  04_predict_48h  -> compute_calendar_fields(), compute_horizon_fields() (yardımcı)
  05_postprocess  -> (kendi delta hesaplarını yapar, forecast_logger'a dokunmaz)
  run_daily/UI    -> write_forecast_log(ctx)      (06'dan sonra)
  01_ingest       -> update_actuals_log(...)      (D+1 yük dalgası)
  02_fetch_weather-> update_actuals_log_weather()  (~D+6 hava dalgası)
"""

from __future__ import annotations

import json
import shutil
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config_live as C

log = logging.getLogger("adm_live")

DATA_DIR = C.DATA_DIR
RAW_TARGET_COL = C.RAW_TARGET_COL
RAW_DATE_COL = C.RAW_DATE_COL
RAW_HOUR_COL = C.RAW_HOUR_COL

RAW_PREDICTIONS_PATH = DATA_DIR / "weather_cache" / "raw_predictions.parquet"
RAW_PREDICTIONS_META_PATH = DATA_DIR / "weather_cache" / "raw_predictions_meta.json"
POSTPROC_PATH = DATA_DIR / "weather_cache" / "postprocessed_predictions.parquet"

WEATHER_STATION_TEMP_COLS = [f"{s}_app_temp_actual" for s in C.WEATHER_STATIONS]

# ── pyarrow şemaları (stlf_forecast_log_tasarim.md §2 / §3 birebir) ───────────

FORECAST_LOG_SCHEMA = pa.schema([
    ("edas_id", pa.string()),
    ("run_id", pa.string()),
    ("config_hash", pa.string()),
    ("issue_ts", pa.timestamp("us")),
    ("target_ts", pa.timestamp("us")),
    ("target_date", pa.string()),
    ("horizon_day", pa.string()),          # "T+1" / "T+2"
    ("lead_time_h", pa.int32()),
    # Alt-model ham tahminleri
    ("y_pred_xgb", pa.float64()),
    ("y_pred_lgbm", pa.float64()),
    ("y_pred_cat", pa.float64()),
    ("y_pred_chronos", pa.float64()),
    ("cat_present", pa.bool_()),
    ("chronos_ok", pa.bool_()),
    # Stacking / meta katman
    ("y_pred_ens_raw", pa.float64()),
    ("meta_method", pa.string()),
    ("meta_w_xgb", pa.float64()),
    ("meta_w_lgbm", pa.float64()),
    ("meta_w_cat", pa.float64()),
    ("meta_w_chronos", pa.float64()),
    ("meta_intercept", pa.float64()),
    # Düzeltme zinciri
    ("override_active", pa.bool_()),
    ("override_delta", pa.float64()),
    ("subst_active", pa.bool_()),
    ("subst_delta", pa.float64()),
    ("pv_bias_delta", pa.float64()),
    ("y_pred_final", pa.float64()),
    # Kullanılan dış girdiler
    ("wx_temp_fcst", pa.float64()),
    ("wx_ghi_fcst", pa.float64()),
    # Takvim & segment
    ("day_type", pa.string()),
    ("flag_holiday", pa.bool_()),
    ("flag_bridge", pa.bool_()),
    ("flag_ramadan", pa.bool_()),
    # Sürüm & tekrarlanabilirlik
    ("model_versions", pa.string()),        # json
    ("feature_snapshot_ref", pa.string()),
])

ACTUALS_LOG_SCHEMA = pa.schema([
    ("edas_id", pa.string()),
    ("target_ts", pa.timestamp("us")),
    ("target_date", pa.string()),
    ("y_actual", pa.float64()),
    ("wx_temp_actual", pa.float64()),
    ("wx_ghi_actual", pa.float64()),
    ("data_quality_flag", pa.string()),
    ("known_event", pa.string()),
])


def _empty_frame(schema: pa.Schema) -> pd.DataFrame:
    return schema.empty_table().to_pandas()


def _write_typed_parquet(df: pd.DataFrame, schema: pa.Schema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.reindex(columns=[f.name for f in schema])
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path)


# ── Yardımcılar: 04'ün ürettiği calendar/horizon alanları ─────────────────────

def compute_calendar_fields(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """day_type / flag_holiday / flag_bridge / flag_ramadan — 04'te ÇAĞRILIRKEN
    `deliver_idx` (DataManager'ın kendi DatetimeIndex'i) DEĞİL, Tarih+Saat'ten
    yeniden kurulan gerçek `target_ts` verilmeli.

    DİKKAT (bulundu, düzeltildi): DataManager'ın kendi index'i Saat=0 satırlarında
    bir gün İLERİ kayıyor (eski "hour-ending" konvansiyonu kalıntısı — bkz.
    03_build_features.py:_backfill_calendar_columns'daki aynı gözlem). feature_df'in
    KENDİ Yilbasi/Milli_Bayram/Is_Ramadan kolonları da bu kaymış index'e göre
    hizalı olduğundan, onlara güvenmek yerine takvimi doğrudan statik
    holiday_calendar tablosundan (Datetime'dan bağımsız, saf tarih hesabı)
    türetiyoruz — target_ts ile her koşulda tutarlı kalır.

    day_type kategorileri (roadmap §2.2): hafta_ici / cumartesi / pazar /
    hafta_ici_tatil / hafta_sonu_tatil.
    """
    from src.holiday_calendar import build_holiday_calendar, classify_de_facto_bridge_day, BRIDGE_DATES

    years = list(range(idx.year.min() - 1, idx.year.max() + 2))
    cal = build_holiday_calendar(years)

    flag_holiday, flag_bridge, flag_ramadan = [], [], []
    for ts in idx:
        d = ts.date()
        meta = cal.get(d)
        flag_holiday.append(meta is not None and meta["holiday_type"] in ("religious", "official"))
        flag_ramadan.append(meta is not None and meta["holiday_type"] == "ramadan_period")
        flag_bridge.append(d in BRIDGE_DATES or classify_de_facto_bridge_day(d) is not None)

    flag_holiday = np.array(flag_holiday)
    flag_ramadan = np.array(flag_ramadan)
    flag_bridge = np.array(flag_bridge)

    dow = idx.dayofweek.to_numpy()
    day_type = np.where(
        flag_holiday & (dow < 5), "hafta_ici_tatil",
        np.where(
            flag_holiday & (dow >= 5), "hafta_sonu_tatil",
            np.where(dow == 5, "cumartesi", np.where(dow == 6, "pazar", "hafta_ici"))
        )
    )

    return pd.DataFrame({
        "day_type": day_type,
        "flag_holiday": flag_holiday,
        "flag_bridge": flag_bridge,
        "flag_ramadan": flag_ramadan,
    }, index=idx)


def compute_horizon_fields(idx: pd.DatetimeIndex, issue_ts: datetime, test_size: int) -> pd.DataFrame:
    """horizon_day (T+1/T+2) + lead_time_h — is_t2 maskesi 05'teki ile aynı (TEST_SIZE//2)."""
    pos = np.arange(len(idx))
    horizon_day = np.where(pos < test_size // 2, "T+1", "T+2")
    issue_ts = pd.Timestamp(issue_ts)
    lead_time_h = ((idx - issue_ts) / pd.Timedelta(hours=1)).to_numpy().round().astype("int32")
    return pd.DataFrame({"horizon_day": horizon_day, "lead_time_h": lead_time_h}, index=idx)


# ── forecast_log yazımı ────────────────────────────────────────────────────────

def _forecast_log_path(edas_id: str, target_date: str, run_id: str) -> Path:
    return (C.FORECAST_LOG_DIR / f"edas_id={edas_id}" / f"target_date={target_date}"
            / f"run_{run_id}.parquet")


def write_forecast_log(ctx: dict) -> dict:
    """postprocessed_predictions.parquet + raw_predictions_meta.json sidecar + manifest.json
    -> forecast_log parquet (target_date başına partition, 06'dan sonra çağrılır).

    Girdi kolonları 04/05 tarafından üretilir (bkz. dosya başlığı). Eksikse
    (henüz enstrümante edilmemiş run) NaN/None ile doldurulur — sessizce atlanmaz,
    ama pipeline'ı bozmaz.
    """
    if not POSTPROC_PATH.exists():
        return {"status": "no_postproc"}

    df = pd.read_parquet(POSTPROC_PATH)
    if "Datetime" not in df.columns:
        return {"status": "no_datetime_col"}
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    meta = {}
    if RAW_PREDICTIONS_META_PATH.exists():
        meta = json.loads(RAW_PREDICTIONS_META_PATH.read_text(encoding="utf-8"))

    manifest = {}
    manifest_path = C.MODEL_ARCHIVE_DIR / ctx["run_id"] / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_snapshot = manifest.get("feature_snapshot_ref")

    n = len(df)

    def col(name, default=None):
        return df[name] if name in df.columns else pd.Series([default] * n, index=df.index)

    out = pd.DataFrame({
        "edas_id": ctx["edas_id"],
        "run_id": ctx["run_id"],
        "config_hash": ctx["config_hash"],
        "issue_ts": pd.Timestamp(ctx["started_at"]),
        "target_ts": df["Datetime"],
        "target_date": df["Datetime"].dt.strftime("%Y-%m-%d"),
        "horizon_day": col("horizon_day"),
        "lead_time_h": col("lead_time_h"),
        "y_pred_xgb": col("XGB_Pred"),
        "y_pred_lgbm": col("LGBM_Pred"),
        "y_pred_cat": col("CAT_Pred"),
        "y_pred_chronos": col("CHRONOS_Pred"),
        "cat_present": bool(meta.get("cat_present")) if "cat_present" in meta else col("cat_present", False),
        "chronos_ok": bool(meta.get("chronos_ok")) if "chronos_ok" in meta else col("chronos_ok", False),
        "y_pred_ens_raw": col("Ensemble_Pred_Raw"),
        "meta_method": meta.get("meta_method"),
        "meta_w_xgb": meta.get("meta_w_xgb"),
        "meta_w_lgbm": meta.get("meta_w_lgbm"),
        "meta_w_cat": meta.get("meta_w_cat"),
        "meta_w_chronos": meta.get("meta_w_chronos"),
        "meta_intercept": meta.get("meta_intercept"),
        "override_active": col("override_active", False),
        "override_delta": col("override_delta", 0.0),
        "subst_active": col("subst_active", False),
        "subst_delta": col("subst_delta", 0.0),
        "pv_bias_delta": col("pv_bias_delta", 0.0),
        "y_pred_final": col("Final_Pred"),
        "wx_temp_fcst": col("wx_temp_fcst"),
        "wx_ghi_fcst": col("wx_ghi_fcst"),
        "day_type": col("day_type"),
        "flag_holiday": col("flag_holiday", False),
        "flag_bridge": col("flag_bridge", False),
        "flag_ramadan": col("flag_ramadan", False),
        "model_versions": json.dumps(manifest.get("model_versions", {}), ensure_ascii=False),
        "feature_snapshot_ref": feature_snapshot,
    })

    n_written = 0
    for target_date, part in out.groupby("target_date"):
        path = _forecast_log_path(ctx["edas_id"], target_date, ctx["run_id"])
        _write_typed_parquet(part, FORECAST_LOG_SCHEMA, path)
        n_written += len(part)

    log.info(f"[ForecastLog] {n_written} satır yazıldı -> {C.FORECAST_LOG_DIR}")
    return {"status": "ok", "rows": n_written, "target_dates": sorted(out["target_date"].unique().tolist())}


# ── actuals_log: D+1 yük dalgası ───────────────────────────────────────────────

def _actuals_log_path(edas_id: str, target_date: str) -> Path:
    return C.ACTUALS_LOG_DIR / f"edas_id={edas_id}" / f"target_date={target_date}.parquet"


def _load_known_events() -> pd.DataFrame | None:
    if not C.KNOWN_EVENTS_CSV.exists():
        return None
    try:
        ev = pd.read_csv(C.KNOWN_EVENTS_CSV)
        if ev.empty:
            return None
        ev["ts_start"] = pd.to_datetime(ev["ts_start"])
        ev["ts_end"] = pd.to_datetime(ev["ts_end"])
        return ev
    except Exception as e:
        log.warning(f"[ActualsLog] known_events.csv okunamadı: {e}")
        return None


def _known_event_for(edas_id: str, ts: pd.Timestamp, events: pd.DataFrame | None) -> Optional[str]:
    if events is None:
        return None
    m = events[(events["edas_id"] == edas_id) & (events["ts_start"] <= ts) & (events["ts_end"] >= ts)]
    if m.empty:
        return None
    return f"{m.iloc[0]['category']}:{m.iloc[0].get('note', '')}"


def _upsert_by_date(edas_id: str, updates: pd.DataFrame) -> int:
    """updates: 'target_ts' dahil her kolon. Tarihe göre partition edip upsert eder.

    DİKKAT: target_date'i her zaman `updates["target_ts"]`'in KENDİSİNDEN türet
    (dışarıdan ayrı bir Series/array değil). `updates` çoğu zaman `.to_numpy()`
    ile taze bir RangeIndex'le kurulur; eğer tarih dizisi başka bir (filtrelenmiş,
    süreksiz index'li) kaynaktan geliyorsa `.assign()`/`groupby` index'e göre
    hizalar ve TÜM satırlar NaN key'e düşüp sessizce atlanır (canlıda 0 satır
    yazan bir upsert — bu fonksiyon tam bunu önlemek için var)."""
    target_date = pd.to_datetime(updates["target_ts"]).dt.strftime("%Y-%m-%d")
    n = 0
    for d, part in updates.groupby(target_date):
        _upsert_actuals(edas_id, d, part)
        n += len(part)
    return n


def _upsert_actuals(edas_id: str, target_date: str, updates: pd.DataFrame) -> None:
    """target_ts anahtarıyla upsert: mevcut dosyada olmayan kolonlar korunur."""
    path = _actuals_log_path(edas_id, target_date)
    if path.exists():
        existing = pd.read_parquet(path)
        existing["target_ts"] = pd.to_datetime(existing["target_ts"])
        merged = existing.set_index("target_ts")
        upd = updates.set_index("target_ts")
        for c in upd.columns:
            if c not in merged.columns:
                merged[c] = None
            merged.loc[upd.index, c] = upd[c]
        merged = merged.reset_index()
    else:
        merged = updates.copy()

    merged["edas_id"] = edas_id
    merged["target_date"] = target_date
    _write_typed_parquet(merged, ACTUALS_LOG_SCHEMA, path)


def derive_data_quality_flags(target_col: pd.Series) -> pd.Series:
    """Faz 0 basit kontrol: eksik / sıfır değer işaretle. Detaylı anomali
    tespiti (spike/flat-line) Faz 1 triyaj kapsamında."""
    flags = pd.Series("", index=target_col.index, dtype=object)
    flags[target_col.isna()] = "missing"
    flags[(target_col == 0) & (~target_col.isna())] = "zero_value"
    return flags


def update_actuals_log(day_df: pd.DataFrame, edas_id: Optional[str] = None) -> dict:
    """01_ingest_actual'dan çağrılır. day_df: RAW_DATE_COL/RAW_HOUR_COL/RAW_TARGET_COL,
    tek bir günün (genelde 24) satırı — validate() sonrası, henüz upsert edilmemiş hali de olur.
    """
    edas_id = edas_id or C.EDAS_ID
    if day_df.empty:
        return {"status": "no_data"}

    target_ts = pd.to_datetime(day_df[RAW_DATE_COL]) + pd.to_timedelta(day_df[RAW_HOUR_COL], unit="h")
    quality = derive_data_quality_flags(day_df[RAW_TARGET_COL])
    events = _load_known_events()

    updates = pd.DataFrame({
        "target_ts": target_ts.to_numpy(),
        "y_actual": day_df[RAW_TARGET_COL].to_numpy(),
        "data_quality_flag": quality.to_numpy(),
        "known_event": [_known_event_for(edas_id, ts, events) for ts in target_ts],
    })

    n_written = _upsert_by_date(edas_id, updates)

    log.info(f"[ActualsLog] {n_written} satır (y_actual) upsert edildi")
    return {"status": "ok", "rows": n_written}


# ── actuals_log: hava gerçekleşme dalgası (~D+6) ──────────────────────────────

def update_actuals_log_weather(edas_id: Optional[str] = None, lookback_days: int = 30) -> dict:
    """02_fetch_weather'dan çağrılır. weather_history.parquet'teki dolu (_actual)
    satırları actuals_log'a upsert eder. Sadece son `lookback_days` günü tarar
    (performans — tam geçmiş backfill_logs.py'nin işi)."""
    edas_id = edas_id or C.EDAS_ID
    if not C.WEATHER_HISTORY_PARQUET.exists():
        return {"status": "no_weather_history"}

    wh = pd.read_parquet(C.WEATHER_HISTORY_PARQUET)
    wh[RAW_DATE_COL] = pd.to_datetime(wh[RAW_DATE_COL])

    cutoff = pd.Timestamp(date.today() - timedelta(days=lookback_days))
    wh = wh[wh[RAW_DATE_COL] >= cutoff].copy()

    temp_cols = [c for c in WEATHER_STATION_TEMP_COLS if c in wh.columns]
    if not temp_cols or "GHI_ADM_Weighted" not in wh.columns:
        return {"status": "missing_weather_cols"}

    wx_temp_actual = wh[temp_cols].mean(axis=1)
    wx_ghi_actual = wh["GHI_ADM_Weighted"]
    has_actual = wx_temp_actual.notna() | wx_ghi_actual.notna()
    if not has_actual.any():
        return {"status": "no_actuals_yet"}

    wh = wh[has_actual]
    target_ts = wh[RAW_DATE_COL] + pd.to_timedelta(wh[RAW_HOUR_COL], unit="h")

    updates = pd.DataFrame({
        "target_ts": target_ts.to_numpy(),
        "wx_temp_actual": wx_temp_actual[has_actual].to_numpy(),
        "wx_ghi_actual": wx_ghi_actual[has_actual].to_numpy(),
    })

    n_written = _upsert_by_date(edas_id, updates)

    log.info(f"[ActualsLog] {n_written} satır (wx_actual) upsert edildi")
    return {"status": "ok", "rows": n_written}


# ── DuckDB view (türetilmiş) ──────────────────────────────────────────────────

def rebuild_duckdb_views() -> dict:
    """monitoring.duckdb'yi parquet üzerinden (yeniden) kur. Tamamen türetilmiş —
    dosya silinip bu fonksiyon tekrar çağrılırsa aynı sonuç elde edilir."""
    import duckdb

    C.MONITORING_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(C.MONITORING_DB))
    try:
        fc_glob = str(C.FORECAST_LOG_DIR / "**" / "*.parquet")
        ac_glob = str(C.ACTUALS_LOG_DIR / "**" / "*.parquet")

        created = []
        if any(C.FORECAST_LOG_DIR.rglob("*.parquet")):
            # Dedup: aynı (edas_id, target_ts, horizon_day) için birden çok run
            # birikebilir (backfill farklı run_id ile gerçek run'ın üstüne yazmaz).
            # Her hücrede tek satır bırak; gerçek run'ı backfill'e tercih et
            # (backfill=true sona düşer), sonra en güncel issue_ts'yi seç. Aksi
            # halde scorecard, actual geldiğinde gerçek + backfill satırlarını
            # ortalayıp yanlış MAPE üretir.
            con.execute(
                f"CREATE OR REPLACE VIEW forecast_log_v AS "
                f"SELECT * FROM read_parquet('{fc_glob}', hive_partitioning=1) "
                f"QUALIFY row_number() OVER ("
                f"  PARTITION BY edas_id, target_ts, horizon_day "
                f"  ORDER BY (run_id LIKE '%backfill%') ASC, issue_ts DESC, run_id DESC"
                f") = 1"
            )
            created.append("forecast_log_v")
        if any(C.ACTUALS_LOG_DIR.rglob("*.parquet")):
            con.execute(
                f"CREATE OR REPLACE VIEW actuals_log_v AS "
                f"SELECT * FROM read_parquet('{ac_glob}', hive_partitioning=1)"
            )
            created.append("actuals_log_v")
    finally:
        con.close()

    log.info(f"[DuckDB] view'lar kuruldu: {created}")
    return {"status": "ok", "views": created}


# ── Yedekleme (OneDrive'a günlük zip) ─────────────────────────────────────────

def backup_logs_zip(keep_days: int = 30) -> Path:
    """LOG_ROOT (forecast_log + actuals_log + known_events.csv — DuckDB HARİÇ,
    türetilmiş) için günlük zip -> logs/backup/ (OneDrive altı, git-ignored)."""
    C.LOG_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    dest_base = C.LOG_BACKUP_DIR / f"adm_logs_{stamp}"

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if C.FORECAST_LOG_DIR.exists():
            shutil.copytree(C.FORECAST_LOG_DIR, tmp / "forecast_log")
        if C.ACTUALS_LOG_DIR.exists():
            shutil.copytree(C.ACTUALS_LOG_DIR, tmp / "actuals_log")
        if C.KNOWN_EVENTS_CSV.exists():
            shutil.copy2(C.KNOWN_EVENTS_CSV, tmp / "known_events.csv")
        zip_path = shutil.make_archive(str(dest_base), "zip", root_dir=str(tmp))

    for f in C.LOG_BACKUP_DIR.glob("adm_logs_*.zip"):
        try:
            d = date.fromisoformat(f.stem.replace("adm_logs_", ""))
        except ValueError:
            continue
        if (date.today() - d).days > keep_days:
            f.unlink(missing_ok=True)

    log.info(f"[Backup] {zip_path}")
    return Path(zip_path)
