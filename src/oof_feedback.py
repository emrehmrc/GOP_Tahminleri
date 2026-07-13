"""
oof_feedback.py — OOF (Out-of-Fold) Feedback Loop
====================================================
Dün tahmini ile bugünün actual'ını karşılaştırır.
Rolling Ridge stacker'ı OOF üzerinde retrains.
PV bias / holiday alpha refit için residual kaydeder.

Çağrı sırası:
  1. update_oof_history() — step 01'den sonra (actual ingest sonrası)
  2. get_rolling_ridge()  — step 04'te stack_predictions içinde

NOT: log_daily_mape() (kümülatif MAPE, roadmap borç #2) emekliye ayrıldı —
yerini src/scorecard.py'nin günlük/pencereli daily_scorecard'ı aldı.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import config_live as _C
from pathlib import Path
from datetime import timedelta
from typing import Optional


def _is_chronos_fallback(day_forecast: pd.DataFrame) -> bool:
    """Eski (pre-fix) Chronos sessiz-fallback bug'ının imzası: Chronos çökünce
    CHRONOS_Pred = XGB_Pred birebir kopyalanıyordu (bkz. 04_predict_48h.py run()
    içindeki chronos_ok=False dalı) — bu günlerde "4-model" aslında XGB'yi iki kez
    sayıyor ve T+2 recursive'de LGBM'in düzleşmesiyle birleşince ensemble'ı
    kolinear/bozuk verilerle kirletiyordu (2026-07-06 oturum teşhisi, arşivlenmiş
    2026-07-03/07-04 run'larında gözlemlendi). get_rolling_ridge bu günleri
    eğitim setinden dışlamalı — yoksa Rolling Ridge aktifleştiği an bu bozuk
    günlerle zehirlenir."""
    if "XGB_Pred" not in day_forecast.columns or "CHRONOS_Pred" not in day_forecast.columns:
        return False
    x = day_forecast["XGB_Pred"].to_numpy(dtype=float)
    c = day_forecast["CHRONOS_Pred"].to_numpy(dtype=float)
    if len(x) == 0 or np.isnan(x).all() or np.isnan(c).all():
        return False
    return bool(np.allclose(x, c, rtol=1e-6, atol=1e-6, equal_nan=True))


def _find_archive_for_date(archive_dir: Path, target_date: str) -> Optional[Path]:
    """Arşiv dizininde belirli bir tarih için tahmin içeren dosya bul."""
    if not archive_dir.is_dir():
        return None
    # Naming: {run_date}_run_{target_date}_full48h.parquet
    # target_date = T+1 date, file contains T+1 + T+2
    for f in sorted(archive_dir.glob("*_full48h.parquet"), reverse=True):
        try:
            df = pd.read_parquet(f)
            if "Datetime" in df.columns:
                dts = pd.to_datetime(df["Datetime"])
                if target_date in dts.dt.strftime("%Y-%m-%d").values:
                    return f
        except Exception:
            continue
    return None


def _records_for_day(
    day_forecast: pd.DataFrame,
    day_actuals: pd.DataFrame,
    target_date,
    raw_target_col: str,
    raw_hour_col: str,
) -> list[dict]:
    """`update_oof_history` ve backfill (Faz 2 2c-2 pilot, 2026-07-13) ORTAK çekirdeği --
    tek bir günün tahmin+actual eşleştirmesinden OOF kayıt listesi üretir. `day_forecast`
    'Datetime' ve model tahmin kolonlarını içermeli (canlı arşiv `*_full48h.parquet` ya
    da asof_regen `*_models_REGEN.parquet` — ikisi de aynı şemayı paylaşır)."""
    day_forecast = day_forecast.copy()
    day_forecast["hour"] = day_forecast["Datetime"].dt.hour
    day_actuals_dict = {int(r[raw_hour_col]): r[raw_target_col] for _, r in day_actuals.iterrows()}

    chronos_fallback = _is_chronos_fallback(day_forecast)

    records = []
    for _, row in day_forecast.iterrows():
        h = int(row["hour"])
        actual = day_actuals_dict.get(h)
        if actual is None or np.isnan(actual):
            continue
        rec = {"date": str(target_date), "hour": h, "actual": float(actual),
               "chronos_fallback": chronos_fallback}
        for col in ["XGB_Pred", "LGBM_Pred", "CAT_Pred", "CHRONOS_Pred", "Ensemble_Pred"]:
            if col in row and pd.notna(row[col]):
                rec[col] = float(row[col])
        records.append(rec)
    return records


def update_oof_history(
    master_path: Path,
    archive_dir: Path,
    oof_path: Path,
    raw_target_col: str = "ADM_Dağıtılan_Enerji_(MWh)",
    raw_date_col: str = "Tarih",
    raw_hour_col: str = "Saat",
) -> dict:
    """
    Step 01'den sonra çağrılır. Yeni ingest edilen günün actual'ını
    arşivlenmiş tahminle karşılaştırır, OOF history'e append eder.

    Returns: {"status": "ok", "date": "...", "mape": ...} or {"status": "no_data"}
    """
    if not master_path.exists():
        return {"status": "no_master"}

    master = pd.read_parquet(master_path)
    master[raw_date_col] = pd.to_datetime(master[raw_date_col])
    last_actual_date = master[raw_date_col].max().date()

    # Arşivde bu tarih için tahmin var mı?
    archive_file = _find_archive_for_date(archive_dir, str(last_actual_date))
    if archive_file is None:
        return {"status": "no_archive", "date": str(last_actual_date)}

    # Tahminleri yükle
    forecast = pd.read_parquet(archive_file)
    if "Datetime" not in forecast.columns:
        return {"status": "no_datetime_col"}

    forecast["Datetime"] = pd.to_datetime(forecast["Datetime"])
    day_forecast = forecast[forecast["Datetime"].dt.strftime("%Y-%m-%d") == str(last_actual_date)].copy()

    if len(day_forecast) == 0:
        return {"status": "no_match"}

    # Actual'ları al
    day_actuals = master[master[raw_date_col] == pd.Timestamp(last_actual_date)].sort_values(raw_hour_col)
    if len(day_actuals) == 0:
        return {"status": "no_actuals"}

    records = _records_for_day(day_forecast, day_actuals, last_actual_date, raw_target_col, raw_hour_col)

    if not records:
        return {"status": "no_records"}

    new_oof = pd.DataFrame(records)

    # OOF history'e append (idempotent: aynı tarih+saat varsa overwrite)
    if oof_path.exists():
        old = pd.read_parquet(oof_path)
        combined = pd.concat([old, new_oof], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "hour"], keep="last")
    else:
        combined = new_oof

    oof_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(oof_path, index=False)

    # MAPE hesapla
    if "Ensemble_Pred" in new_oof.columns:
        mape = np.mean(np.abs(
            (new_oof["actual"] - new_oof["Ensemble_Pred"]) / (new_oof["actual"] + 1e-10)
        )) * 100
    else:
        mape = None

    print(f"     [OOF] {len(new_oof)} satır eklendi | Toplam: {len(combined)} | MAPE: {mape:.2f}%" if mape else f"     [OOF] {len(new_oof)} satır eklendi")

    return {"status": "ok", "date": str(last_actual_date), "rows": len(new_oof), "total": len(combined), "mape": mape}


def get_rolling_ridge(
    oof_path: Path,
    pred_cols: list,
    lookback_days: int = 60,
    min_samples: int | None = None,
) -> Optional[object]:
    """
    OOF history üzerinde rolling Ridge meta-model eğit.
    Step 04'te stack_predictions tarafından çağrılır.

    NEDEN eşik düşürüldü (varsayılan 168 -> config ROLLING_RIDGE_MIN_SAMPLES,
    şu an 48): 168 saat (7 gün) OOF birikene kadar donmuş 3-model (XGB/LGBM/CAT,
    Chronos'suz — Boray'ın kendi CV artefaktı) stacker kullanılıyordu. Bu statik
    ağırlık (XGB=0.50, LGBM=0.50, CAT=0) LGBM'in belirli günlerde düzleşmesini
    (recursive T+2'de gözlemlendi) telafi edemiyor, çünkü Chronos formülde hiç
    yok. 48 saat (2 gün) sonra adaptif Ridge devreye girip 4 modeli de güncel
    hataya göre ağırlıklandırır — alpha=100 güçlü regularizasyon az veriyle aşırı
    uydurmayı sınırlar.

    Returns: (trained Ridge model, eğitimde kullanılan kolon listesi) veya None
    (yeterli veri yoksa). Kolon listesi ZORUNLU dönüyor — çağıran taraf kendi
    pred_cols'undan farklı bir sırayla/sette `.predict()` çağırırsa (ör. OOF'ta
    olmayan CAT_Pred'i dahil ederse) sklearn "X has N features, Ridge expects M"
    hatası fırlatır (canlıda yaşandı, teşhis edildi).
    """
    if min_samples is None:
        min_samples = getattr(_C, "ROLLING_RIDGE_MIN_SAMPLES", 168)
    if not oof_path.exists():
        return None

    oof = pd.read_parquet(oof_path)

    # Kirli gün karantinası: eski Chronos-fallback bug'ının imzasını taşıyan
    # satırlar (bkz. _is_chronos_fallback) eğitim setine ASLA girmemeli — yoksa
    # Rolling Ridge aktifleştiği an kolinear/bozuk veriyle zehirlenir (2026-07-06
    # oturum teşhisi: 48 satırlık ilk OOF'un yarısı böyle kirliydi). Eski
    # kayıtlarda kolon yoksa (bu düzeltmeden önce yazılmış) temiz kabul edilir —
    # kirlilik olsa bile ondan sonraki adımda `min_samples` eşiği zaten korur.
    if "chronos_fallback" in oof.columns:
        oof = oof[~oof["chronos_fallback"].fillna(False)].copy()

    if len(oof) < min_samples:
        return None

    # Lookback window
    oof["date"] = pd.to_datetime(oof["date"])
    cutoff = oof["date"].max() - timedelta(days=lookback_days)
    recent = oof[oof["date"] >= cutoff].copy()

    if len(recent) < min_samples:
        recent = oof.copy()

    # Sadece mevcut pred kolonlarını kullan
    available = [c for c in pred_cols if c in recent.columns]
    if len(available) < 2:
        return None

    X = recent[available].values
    y = recent["actual"].values

    # NaN olan satırları at
    mask = ~np.isnan(y)
    for c in available:
        mask &= ~np.isnan(recent[c].values)
    X = X[mask]
    y = y[mask]

    if len(y) < min_samples:
        return None

    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=100.0, fit_intercept=True)
    ridge.fit(X, y)

    print(f"     [Stacker] Rolling Ridge OOF ({len(y)} samples, {lookback_days}d) — cols: {available}")
    return ridge, available


def get_inverse_mape_weights(
    oof_path: Path,
    pred_cols: list,
    lookback_days: int = 60,
    min_days: int = 14,
) -> Optional[dict]:
    """
    OOF history üzerinden inverse-MAPE adaptive weights hesapla.
    Rolling Ridge için yeterli OOF verisi birikmeden önce (henuz < min_samples)
    statik weight'ler yerine guncel model performansina dayali agirlik verir.

    Faz 2 2c-1 (2026-07-13) — karantina gevşetildi: eskiden chronos_fallback
    günün TÜM satırları (XGB/LGBM/CAT dahil) eğitimden düşürülüyordu. Ama sadece
    CHRONOS_Pred kirli (XGB kopyası) — diğer 3 modelin o günkü OOF'u geçerli ve
    zaten kıt olan örnek sayısını gereksiz azaltıyordu. Artık SADECE
    CHRONOS_Pred'in kendi MAPE hesabından o satırlar NaN'lanarak çıkarılıyor,
    diğer kolonlar (Ensemble_Pred dahil) chronos_fallback günlerini de kullanır
    (get_rolling_ridge'te DURUM FARKLI: çok değişkenli fit tam satır ister, o
    yüzden orada whole-row drop korunuyor).

    Returns: {col: weight} normalize edilmis, veya None (yetersiz veri).
    """
    if not oof_path.exists():
        return None
    oof = pd.read_parquet(oof_path)
    oof["date"] = pd.to_datetime(oof["date"])
    unique_dates = oof["date"].nunique()
    if unique_dates < min_days:
        return None

    cutoff = oof["date"].max() - timedelta(days=lookback_days)
    recent = oof[oof["date"] >= cutoff].copy()
    if len(recent) < min_days * 12:
        recent = oof.copy()

    available = [c for c in pred_cols if c in recent.columns]
    if len(available) < 2:
        return None

    fallback_mask = (
        recent["chronos_fallback"].fillna(False) if "chronos_fallback" in recent.columns
        else pd.Series(False, index=recent.index)
    )

    mape_by_col = {}
    for col in available:
        col_series = recent[col]
        if col == "CHRONOS_Pred":
            # Sadece bu kolonun kendi MAPE'i icin fallback gunleri NaN'lanir --
            # diger kolonlar (ve satirlar) etkilenmez (bkz. fonksiyon docstring'i).
            col_series = col_series.where(~fallback_mask)
        d = pd.DataFrame({"actual": recent["actual"], col: col_series}).dropna()
        if len(d) < 12:
            continue
        ape = np.abs((d["actual"] - d[col]) / (d["actual"] + 1e-10)) * 100
        mape_by_col[col] = float(ape.mean())

    if not mape_by_col:
        return None

    inv = {m: 1.0 / v for m, v in mape_by_col.items() if v > 0 and np.isfinite(v)}
    if not inv:
        return None
    total = sum(inv.values())
    weights = {m: w / total for m, w in inv.items()}
    print(f"     [Stacker] Inverse-MAPE weights ({len(recent)} samples, {unique_dates}d): "
          f"{ {k: round(v, 3) for k, v in weights.items()} }")
    return weights


def _day_type_group_for_dates(dates: pd.Series) -> pd.Series:
    """monitoring/forecast_logger.py:_calendar_columns ile aynı day_type mantığı
    (dow + resmi/dini tatil), monitoring/scorecard.py:DAY_TYPE_GROUPS ile aynı
    4 gruba indirilir (hafta_ici/cumartesi/pazar/ozel_gun) — get_segment_weights
    ve scorecard'ın segment kırılımı AYNI gruplamayı kullanmalı."""
    from holiday_calendar import build_holiday_calendar
    from monitoring.scorecard import DAY_TYPE_GROUPS

    years = list(range(int(dates.dt.year.min()) - 1, int(dates.dt.year.max()) + 2))
    cal = build_holiday_calendar(years)
    dow = dates.dt.dayofweek

    def _group(d, wd):
        meta = cal.get(d.date())
        is_holiday = meta is not None and meta["holiday_type"] in ("religious", "official")
        if is_holiday:
            return "ozel_gun"
        if wd == 5:
            return "cumartesi"
        if wd == 6:
            return "pazar"
        return "hafta_ici"

    return pd.Series([_group(d, wd) for d, wd in zip(dates, dow)], index=dates.index)


def get_segment_weights(
    oof_path: Path,
    pred_cols: list,
    lookback_days: int = 30,
    min_samples_per_segment: int = 30,
) -> Optional[dict]:
    """Faz 2 2c-2 (2026-07-13): `(hour_block, day_type_group)` segmenti başına
    rolling-lookback_days inverse-MAPE ağırlığı — MASTER_PLAN.md §2c-2
    "segment-bazlı adaptif ensemble ağırlıklandırma". `get_inverse_mape_weights`
    ile aynı mantık, TEK farkla: global değil, HOUR_BLOCKS × DAY_TYPE_GROUPS
    kesişiminde ayrı ayrı hesaplanır (GDZ'nin akşam/gece/sabah'ta farklı model
    performansı göstermesi — bkz. 2b-3 segment breakdown bulgusu — global tek
    ağırlığın gizlediği bir sinyal).

    ÖNEMLİ (2026-07-13 durumu): bu fonksiyon SCAFFOLDING'dir — canlı `04_predict_48h.py`
    stacking cascade'ine HENÜZ bağlanmadı. Neden: `min_samples_per_segment` her
    segment için ayrı ayrı sağlanmalı (8 segment × ~30 gün = çok daha fazla OOF
    tarihi gerekir), mevcut canlı `oof_history.parquet` sadece ~4 gün — segment
    başına anlamlı bir walkforward A/B doğrulaması için yetersiz. Yeterli OOF
    birikince (doğal olarak veya ayrı bir backfill oturumunda) walkforward A/B
    raporu üretilmeden canlı cascade'e eklenmeyecek (governance, MASTER_PLAN.md).

    Returns: {(hour_block, day_type_group): {col: weight}} — segment için yeterli
    örnek yoksa o segment sözlükte YOKTUR (çağıran taraf global/statik ağırlığa
    düşmeli). Hiç segment hesaplanamazsa None.
    """
    if not oof_path.exists():
        return None
    oof = pd.read_parquet(oof_path)
    oof["date"] = pd.to_datetime(oof["date"])

    unique_dates = oof["date"].nunique()
    cutoff = oof["date"].max() - timedelta(days=lookback_days)
    recent = oof[oof["date"] >= cutoff].copy()
    if recent.empty:
        return None

    from monitoring.scorecard import HOUR_BLOCKS

    recent["hour_block"] = None
    for label, hrs in HOUR_BLOCKS.items():
        recent.loc[recent["hour"].isin(hrs), "hour_block"] = label
    recent["day_type_group"] = _day_type_group_for_dates(recent["date"])

    fallback_mask = (
        recent["chronos_fallback"].fillna(False) if "chronos_fallback" in recent.columns
        else pd.Series(False, index=recent.index)
    )
    available = [c for c in pred_cols if c in recent.columns]
    if len(available) < 2:
        return None

    segment_weights: dict[tuple[str, str], dict[str, float]] = {}
    for (hour_block, day_type_group), g in recent.groupby(["hour_block", "day_type_group"]):
        if hour_block is None or len(g) < min_samples_per_segment:
            continue
        mape_by_col = {}
        for col in available:
            col_series = g[col].where(~fallback_mask.loc[g.index]) if col == "CHRONOS_Pred" else g[col]
            d = pd.DataFrame({"actual": g["actual"], col: col_series}).dropna()
            if len(d) < 8:
                continue
            ape = np.abs((d["actual"] - d[col]) / (d["actual"] + 1e-10)) * 100
            mape_by_col[col] = float(ape.mean())
        inv = {m: 1.0 / v for m, v in mape_by_col.items() if v > 0 and np.isfinite(v)}
        if not inv:
            continue
        total = sum(inv.values())
        segment_weights[(hour_block, day_type_group)] = {m: w / total for m, w in inv.items()}

    if not segment_weights:
        return None
    print(f"     [Stacker] Segment-adaptive weights: {len(segment_weights)} segment "
          f"({unique_dates}g OOF, lookback={lookback_days}g)")
    return segment_weights
