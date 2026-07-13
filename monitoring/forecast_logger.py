"""
monitoring/forecast_logger.py — ADM + GDZ ortak izleme yardımcıları (Faz 2).

BİLEREK burada OLMAYAN: write_forecast_log / update_actuals_log /
update_actuals_log_weather — bunlar GDZ'nin T1/T2 ayrı-kolon (coalesce
gerektiren) şeması ile ADM'nin doğrudan-kolon şeması arasındaki GERÇEK
yapısal farktan dolayı her tenant'ın kendi src/forecast_logger.py'sinde
tenant-spesifik kalır (bkz. paket __init__.py).

Burada olan her şey İKİ TENANT'TA DA BİREBİR AYNI mantık — tek kopya,
`TenantConfig` ile parametrize.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from monitoring.schema import FORECAST_LOG_SCHEMA, ACTUALS_LOG_SCHEMA
from monitoring.tenant_config import TenantConfig
# ADM ve GDZ kendi `holiday_calendar.py` kopyalarını bare modül olarak öncelikli
# src dizininden sunar. `src.holiday_calendar` kullanmak GDZ subprocess'inde
# araştırma tarafındaki farklı `src` paketini kilitleyip importu bozuyordu.
from holiday_calendar import build_holiday_calendar, classify_de_facto_bridge_day, BRIDGE_DATES

ARCHIVE_FULL48H_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_run_(.+)_full48h\.parquet$")


# ── Ortak dosya-yazma yardımcıları ──────────────────────────────────────────────

def _write_typed_parquet(df: pd.DataFrame, schema: pa.Schema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.reindex(columns=[f.name for f in schema])
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path)


def _forecast_log_path(config: TenantConfig, target_date: str, run_id: str) -> Path:
    return (config.forecast_log_dir / f"edas_id={config.edas_id}"
            / f"target_date={target_date}" / f"run_{run_id}.parquet")


def _actuals_log_path(config: TenantConfig, target_date: str) -> Path:
    return config.actuals_log_dir / f"edas_id={config.edas_id}" / f"target_date={target_date}.parquet"


# ── Takvim alanları (ADM/GDZ birebir aynı — holiday_calendar.py da byte-identical) ──

def compute_calendar_fields(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """day_type / flag_holiday / flag_bridge / flag_ramadan — statik
    holiday_calendar tablosundan türetilir (target_ts'e bağlı, feature_df'in
    kendi takvim kolonlarına DEĞİL — DataManager'ın hour=0 kayması riskinden
    bağımsız kalır)."""
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


# ── DuckDB view (türetilmiş) ──────────────────────────────────────────────────

def rebuild_duckdb_views(config: TenantConfig) -> dict:
    """monitoring.duckdb'yi parquet üzerinden (yeniden) kur. Tamamen türetilmiş —
    dosya silinip bu fonksiyon tekrar çağrılırsa aynı sonuç elde edilir."""
    import duckdb

    config.monitoring_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.monitoring_db))
    try:
        fc_glob = str(config.forecast_log_dir / "**" / "*.parquet")
        ac_glob = str(config.actuals_log_dir / "**" / "*.parquet")

        created = []
        if any(config.forecast_log_dir.rglob("*.parquet")):
            # union_by_name=1: Faz 1 şemaya issue_date/horizon_days/source ekledi
            # — eski dosyalarda yok, union_by_name eksik kolonları NULL yapar.
            #
            # hive_partitioning=0 (BULUNDU 2026-07-10): dizin adı 'target_date=...'
            # hive-partitioning=1 ile DATE tipi tahmin ediyor, dosya İÇİNDEKİ
            # target_date VARCHAR — union_by_name bu çakışmada DuckDB internal
            # hatasına düşüyordu. edas_id/target_date her satırda zaten gerçek
            # kolon, partition inference'a gerek yok.
            #
            # Dedup: aynı (edas_id, target_ts, horizon_day) için birden çok run
            # birikebilir — gerçek run'ı backfill/archive_heal'e tercih et,
            # sonra en güncel issue_ts.
            #
            # run_count (Faz 1, 2026-07-13): kaç run bu hücreye yazmış — dedup
            # SONUCU tek satır göründüğü için birden fazla canlı run aynı günü
            # gölgeliyorsa (örn. aynı gün iki kez koşturulmuş) bu sessizce
            # kaybolurdu; artık winning satırın kendisinde görünür.
            con.execute(
                f"CREATE OR REPLACE VIEW forecast_log_v AS "
                f"SELECT *, count(*) OVER (PARTITION BY edas_id, target_ts, horizon_day) AS run_count "
                f"FROM read_parquet('{fc_glob}', hive_partitioning=0, union_by_name=1) "
                f"QUALIFY row_number() OVER ("
                f"  PARTITION BY edas_id, target_ts, horizon_day "
                f"  ORDER BY (run_id LIKE '%backfill%' OR run_id LIKE '%archheal%' "
                f"            OR run_id LIKE '%archregen%') ASC, issue_ts DESC, run_id DESC"
                f") = 1"
            )
            created.append("forecast_log_v")
        if any(config.actuals_log_dir.rglob("*.parquet")):
            con.execute(
                f"CREATE OR REPLACE VIEW actuals_log_v AS "
                f"SELECT * FROM read_parquet('{ac_glob}', hive_partitioning=0, union_by_name=1)"
            )
            created.append("actuals_log_v")
    finally:
        con.close()

    logging.getLogger(config.logger_name).info(f"[DuckDB] view'lar kuruldu: {created}")
    return {"status": "ok", "views": created}


# ── Faz 1: arşivden idempotent boşluk doldurma ─────────────────────────────────

def _normalize_hour_grid(df: pd.DataFrame, ts_col: str = "Datetime") -> pd.DataFrame:
    """Bazı arşivler hour-ending grid'inde (ilk satır saat=1), bazıları temiz
    hour-beginning (saat=0) — actuals_log HER ZAMAN hour-beginning, yanlış
    grid join'i sessizce kırar. Tek yer, tek karar."""
    df = df.sort_values(ts_col).reset_index(drop=True)
    first_hour = df[ts_col].iloc[0].hour
    if first_hour == 0:
        return df
    if first_hour == 1:
        df = df.copy()
        df[ts_col] = df[ts_col] - pd.Timedelta(hours=1)
        return df
    raise ValueError(f"beklenmeyen ilk saat={first_hour} (0 veya 1 olmalı)")


def _issue_date_for_archive(path: Path, df: pd.DataFrame) -> date | None:
    """issue_date'i ÖNCE dosyanın kendi 'issue_date' kolonundan oku (06_deliver.py
    2026-07-10'dan beri yazıyor). Yoksa dosya adının İKİNCİ tarih grubuna düş
    (BULUNDU 2026-07-10: ilk grup toplu regen/backtest çıktılarında hepsi AYNI
    kaydetme-günüyle damgalı, issue_date DEĞİL — saçma horizon_days üretiyordu.
    İkinci grup HER ZAMAN 06_deliver.py'nin target_str'i = teslim günü =
    issue_date+1 — hem gerçek run'larda hem toplu dosyalarda güvenilir)."""
    if "issue_date" in df.columns and df["issue_date"].notna().any():
        return pd.Timestamp(df["issue_date"].iloc[0]).date()
    m = ARCHIVE_FULL48H_RE.match(path.name)
    if m:
        try:
            return date.fromisoformat(m.group(2)) - timedelta(days=1)
        except ValueError:
            return None
    return None


def _scan_archive_coverage(config: TenantConfig) -> dict:
    """Arşivdeki her (target_date, horizon_days) hücresi için EN İYİ adayı
    (issue_date kolonu olan > dosya-adından-çıkarılan, aynı öncelikte EN YENİ
    issue_date kazanır) seçer. Döner: {(target_date_str, horizon_days): (...)}."""
    log = logging.getLogger(config.logger_name)
    candidates: dict[tuple, tuple] = {}
    for f in sorted(config.archive_dir.glob("*_full48h.parquet")):
        try:
            raw = pd.read_parquet(f)
        except Exception as e:
            log.warning(f"[Heal] arşiv okunamadı, atlandı: {f.name} ({e})")
            continue
        if "Datetime" not in raw.columns:
            continue
        raw["Datetime"] = pd.to_datetime(raw["Datetime"])

        issue_dt = _issue_date_for_archive(f, raw)
        if issue_dt is None:
            log.warning(f"[Heal] issue_date belirlenemedi, atlandı: {f.name}")
            continue
        has_explicit_issue_col = "issue_date" in raw.columns and raw["issue_date"].notna().any()

        try:
            normalized = _normalize_hour_grid(raw, "Datetime")
        except ValueError as e:
            log.warning(f"[Heal] saat gridi belirsiz, atlandı: {f.name} ({e})")
            continue

        normalized["target_date"] = normalized["Datetime"].dt.strftime("%Y-%m-%d")
        normalized["horizon_days"] = (normalized["Datetime"].dt.date - issue_dt).apply(lambda d: d.days)

        priority = 1 if has_explicit_issue_col else 0
        for (tdate, hdays), part in normalized.groupby(["target_date", "horizon_days"]):
            if len(part) != 24:
                continue  # bu dosyada bu hücre zaten kısmi -- güvenilir aday değil
            if not (-1 <= hdays <= 5):
                log.warning(f"[Heal] mantık dışı horizon_days={hdays} ({f.name}, target={tdate}), atlandı")
                continue
            key = (tdate, int(hdays))
            cur = candidates.get(key)
            if cur is None or (priority, issue_dt) > (cur[0], cur[1]):
                candidates[key] = (priority, issue_dt, f, part)
    return candidates


def heal_forecast_log_gaps(config: TenantConfig) -> dict:
    """Her run sonunda çağrılır. forecast_log_v'de EKSİK ya da KISMİ (24'ten az
    satır) olan (target_date, horizon_days) hücrelerini arşivden doldurur.
    Zaten tam (24 satır) olan hiçbir hücreye dokunmaz — canlı run'ın zengin
    metadata'lı satırlarını asla ezmez."""
    log = logging.getLogger(config.logger_name)
    if not config.archive_dir.exists():
        return {"status": "no_archive_dir"}
    if not config.monitoring_db.exists():
        return {"status": "no_monitoring_db"}

    import duckdb

    con = duckdb.connect(str(config.monitoring_db), read_only=True)
    try:
        tables = con.execute("SHOW TABLES").df()["name"].tolist()
        # DIKKAT (bulundu 2026-07-10): existing_counts'u horizon_days (int) ile
        # anahtarlamak YANLIS -- Faz 1 semasindan ONCEKI 'live' satirlar (gercek
        # gunluk run'lar) dedup view'da HER ZAMAN kazanir (archheal/backfill'e
        # karsi oncelikli) ama horizon_days=NULL tasir (eski dosyalarda o kolon
        # yok). "horizon_days IS NOT NULL" filtresiyle bu satirlar hic
        # sayilmiyor, heal onlari HER calistirmada 'eksik' saniyor ve zararsiz
        # ama gereksiz archheal kopyasi yaziyordu. horizon_day STRING alani
        # (Faz 1 ONCESI de dahil TUM satirlarda var) guvenilir anahtar.
        existing_counts: dict[tuple, int] = {}
        if "forecast_log_v" in tables:
            df_counts = con.execute(
                "SELECT target_date, horizon_day, count(*) n FROM forecast_log_v "
                "WHERE edas_id = ? GROUP BY 1,2",
                [config.edas_id],
            ).df()
            existing_counts = {
                (row.target_date, row.horizon_day): row.n for row in df_counts.itertuples()
            }
    finally:
        con.close()

    candidates = _scan_archive_coverage(config)

    now = pd.Timestamp.now()
    healed_cells = []
    for (tdate, hdays), (priority, issue_dt, src_file, part) in candidates.items():
        horizon_day_label = f"T+{hdays + config.horizon_day_label_offset}"
        if existing_counts.get((tdate, horizon_day_label), 0) >= 24:
            continue

        n = len(part)

        def col(name, default=None, _part=part, _n=n):
            return _part[name] if name in _part.columns else pd.Series([default] * _n, index=_part.index)

        def coalesce(t1_col, t2_col, _part=part, _n=n):
            t1 = col(t1_col, None, _part, _n)
            t2 = col(t2_col, None, _part, _n)
            return t1.combine_first(t2) if (t1_col in _part.columns or t2_col in _part.columns) else t1

        run_id = f"{issue_dt}_archheal"
        lead_time_h = ((part["Datetime"] - now) / pd.Timedelta(hours=1)).round().astype("int32")
        cal = compute_calendar_fields(pd.DatetimeIndex(part["Datetime"])).reset_index(drop=True)

        # y_pred_* kolon adları tenant'a göre değişir: ADM doğrudan XGB_Pred/
        # LGBM_Pred vb, GDZ T1/T2 ayrı (coalesce gerekir). Her ikisini de dener,
        # hangisi varsa o kullanılır.
        y_xgb = coalesce("XGB_T1_Pred", "XGB_T2_Pred") if "XGB_T1_Pred" in part.columns or "XGB_T2_Pred" in part.columns else col("XGB_Pred")
        y_lgbm = coalesce("LGBM_T1_Pred", "LGBM_T2_Pred") if "LGBM_T1_Pred" in part.columns or "LGBM_T2_Pred" in part.columns else col("LGBM_Pred")
        y_cat = coalesce("CAT_T1_Pred", "CAT_T2_Pred") if "CAT_T1_Pred" in part.columns or "CAT_T2_Pred" in part.columns else col("CAT_Pred")
        y_chronos = coalesce("CHRONOS_T1_Pred", "CHRONOS_T2_Pred") if "CHRONOS_T1_Pred" in part.columns or "CHRONOS_T2_Pred" in part.columns else col("CHRONOS_Pred")
        y_ens_raw = col("Ensemble_Pred_Raw") if "Ensemble_Pred_Raw" in part.columns else col("Ensemble_Pred")

        out = pd.DataFrame({
            "edas_id": config.edas_id,
            "run_id": run_id,
            "config_hash": "archheal",
            "issue_ts": now,
            "target_ts": part["Datetime"].to_numpy(),
            "target_date": tdate,
            "horizon_day": horizon_day_label,
            "issue_date": issue_dt,
            "horizon_days": hdays,
            "source": "archive_heal",
            "lead_time_h": lead_time_h.to_numpy(),
            "y_pred_xgb": y_xgb.to_numpy(),
            "y_pred_lgbm": y_lgbm.to_numpy(),
            "y_pred_cat": y_cat.to_numpy(),
            "y_pred_chronos": y_chronos.to_numpy(),
            "cat_present": True,
            "chronos_ok": bool(y_chronos.notna().any()),
            "y_pred_ens_raw": y_ens_raw.to_numpy(),
            "meta_method": "archive_heal",
            "meta_w_xgb": None, "meta_w_lgbm": None, "meta_w_cat": None,
            "meta_w_chronos": None, "meta_intercept": None,
            "override_active": col("override_active", False).to_numpy(),
            "override_delta": col("override_delta", 0.0).to_numpy(),
            "subst_active": col("subst_active", False).to_numpy(),
            "subst_delta": col("subst_delta", 0.0).to_numpy(),
            "pv_bias_delta": col("pv_bias_delta", 0.0).to_numpy(),
            "y_pred_final": col("Final_Pred").to_numpy(),
            "wx_temp_fcst": col("wx_temp_fcst").to_numpy() if "wx_temp_fcst" in part.columns else None,
            "wx_ghi_fcst": col("wx_ghi_fcst").to_numpy() if "wx_ghi_fcst" in part.columns else None,
            "day_type": cal["day_type"].to_numpy(),
            "flag_holiday": cal["flag_holiday"].to_numpy(),
            "flag_bridge": cal["flag_bridge"].to_numpy(),
            "flag_ramadan": cal["flag_ramadan"].to_numpy(),
            "model_versions": json.dumps({"note": f"archive_heal <- {src_file.name}"}, ensure_ascii=False),
            "feature_snapshot_ref": None,
        })

        path = _forecast_log_path(config, tdate, run_id)
        _write_typed_parquet(out, FORECAST_LOG_SCHEMA, path)
        healed_cells.append({"target_date": tdate, "horizon_days": hdays, "source_file": src_file.name})

    if healed_cells:
        log.info(f"[Heal] {len(healed_cells)} hücre arşivden onarıldı: {healed_cells}")
        rebuild_duckdb_views(config)

    return {"status": "ok", "healed_cells": healed_cells, "n_healed": len(healed_cells)}


# ── actuals_log ortak yardımcıları (upsert) ────────────────────────────────────

def derive_data_quality_flags(target_col: pd.Series) -> pd.Series:
    """Faz 0 basit kontrol: eksik / sıfır değer işaretle."""
    flags = pd.Series("", index=target_col.index, dtype=object)
    flags[target_col.isna()] = "missing"
    flags[(target_col == 0) & (~target_col.isna())] = "zero_value"
    return flags


def load_known_events(config: TenantConfig) -> pd.DataFrame | None:
    if not config.known_events_csv.exists():
        return None
    try:
        ev = pd.read_csv(config.known_events_csv)
        if ev.empty:
            return None
        ev["ts_start"] = pd.to_datetime(ev["ts_start"])
        ev["ts_end"] = pd.to_datetime(ev["ts_end"])
        return ev
    except Exception as e:
        logging.getLogger(config.logger_name).warning(f"[ActualsLog] known_events.csv okunamadı: {e}")
        return None


def known_event_for(edas_id: str, ts: pd.Timestamp, events: pd.DataFrame | None) -> Optional[str]:
    if events is None:
        return None
    m = events[(events["edas_id"] == edas_id) & (events["ts_start"] <= ts) & (events["ts_end"] >= ts)]
    if m.empty:
        return None
    return f"{m.iloc[0]['category']}:{m.iloc[0].get('note', '')}"


def upsert_actuals(config: TenantConfig, target_date: str, updates: pd.DataFrame) -> None:
    """target_ts anahtarıyla upsert: mevcut dosyada olmayan kolonlar korunur."""
    path = _actuals_log_path(config, target_date)
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

    merged["edas_id"] = config.edas_id
    merged["target_date"] = target_date
    _write_typed_parquet(merged, ACTUALS_LOG_SCHEMA, path)


def upsert_by_date(config: TenantConfig, updates: pd.DataFrame) -> int:
    """updates: 'target_ts' dahil her kolon. Tarihe göre partition edip upsert eder.

    DİKKAT: target_date'i her zaman `updates["target_ts"]`'in KENDİSİNDEN türet
    (dışarıdan ayrı bir Series/array değil) — aksi halde .assign()/groupby
    index'e göre hizalar ve TÜM satırlar NaN key'e düşüp sessizce atlanır."""
    target_date = pd.to_datetime(updates["target_ts"]).dt.strftime("%Y-%m-%d")
    n = 0
    for d, part in updates.groupby(target_date):
        upsert_actuals(config, d, part)
        n += len(part)
    return n


# ── Yedekleme (OneDrive'a günlük zip) ─────────────────────────────────────────

def backup_logs_zip(config: TenantConfig, keep_days: int = 30) -> Path:
    """LOG_ROOT (forecast_log + actuals_log + known_events.csv — DuckDB HARİÇ,
    türetilmiş) için günlük zip -> logs/backup/ (OneDrive altı, git-ignored)."""
    config.log_backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    dest_base = config.log_backup_dir / f"{config.edas_id.lower()}_logs_{stamp}"

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if config.forecast_log_dir.exists():
            shutil.copytree(config.forecast_log_dir, tmp / "forecast_log")
        if config.actuals_log_dir.exists():
            shutil.copytree(config.actuals_log_dir, tmp / "actuals_log")
        if config.known_events_csv.exists():
            shutil.copy2(config.known_events_csv, tmp / "known_events.csv")
        zip_path = shutil.make_archive(str(dest_base), "zip", root_dir=str(tmp))

    for f in config.log_backup_dir.glob(f"{config.edas_id.lower()}_logs_*.zip"):
        try:
            d = date.fromisoformat(f.stem.replace(f"{config.edas_id.lower()}_logs_", ""))
        except ValueError:
            continue
        if (date.today() - d).days > keep_days:
            f.unlink(missing_ok=True)

    logging.getLogger(config.logger_name).info(f"[Backup] {zip_path}")
    return Path(zip_path)
