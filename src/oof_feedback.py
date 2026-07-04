"""
oof_feedback.py — OOF (Out-of-Fold) Feedback Loop
====================================================
Dün tahmini ile bugünün actual'ını karşılaştırır.
Rolling Ridge stacker'ı OOF üzerinde retrains.
PV bias / holiday alpha refit için residual kaydeder.

Çağrı sırası:
  1. update_oof_history() — step 01'den sonra (actual ingest sonrası)
  2. get_rolling_ridge()  — step 04'te stack_predictions içinde
  3. log_daily_mape()     — step 06'dan sonra
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from typing import Optional


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

    # Eşleştir (Saat bazında)
    day_forecast["hour"] = day_forecast["Datetime"].dt.hour
    day_actuals_dict = {int(r[raw_hour_col]): r[raw_target_col] for _, r in day_actuals.iterrows()}

    records = []
    for _, row in day_forecast.iterrows():
        h = int(row["hour"])
        actual = day_actuals_dict.get(h)
        if actual is None or np.isnan(actual):
            continue
        rec = {"date": str(last_actual_date), "hour": h, "actual": float(actual)}
        for col in ["XGB_Pred", "LGBM_Pred", "CAT_Pred", "CHRONOS_Pred", "Ensemble_Pred"]:
            if col in row and pd.notna(row[col]):
                rec[col] = float(row[col])
        records.append(rec)

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
    min_samples: int = 168,
) -> Optional[object]:
    """
    OOF history üzerinde rolling Ridge meta-model eğit.
    Step 04'te stack_predictions tarafından çağrılır.

    Returns: trained Ridge model or None (yeterli veri yoksa)
    """
    if not oof_path.exists():
        return None

    oof = pd.read_parquet(oof_path)
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
    return ridge


def log_daily_mape(oof_path: Path, logs_dir: Path) -> dict:
    """Günlük MAPE'yi logs/mape_history.json'a kaydet."""
    if not oof_path.exists():
        return {"status": "no_oof"}

    oof = pd.read_parquet(oof_path)
    logs_dir.mkdir(parents=True, exist_ok=True)
    mape_log_path = logs_dir / "mape_history.json"

    history = []
    if mape_log_path.exists():
        try:
            history = json.loads(mape_log_path.read_text(encoding="utf-8"))
        except Exception:
            history = []

    # Her model için günlük MAPE hesapla
    oof["date"] = pd.to_datetime(oof["date"]).dt.strftime("%Y-%m-%d")
    daily = oof.groupby("date").agg({
        "actual": "mean",
        "XGB_Pred": "mean" if "XGB_Pred" in oof.columns else lambda x: np.nan,
        "LGBM_Pred": "mean" if "LGBM_Pred" in oof.columns else lambda x: np.nan,
        "Ensemble_Pred": "mean" if "Ensemble_Pred" in oof.columns else lambda x: np.nan,
    }).reset_index()

    results = {}
    for col in ["XGB_Pred", "LGBM_Pred", "CAT_Pred", "CHRONOS_Pred", "Ensemble_Pred"]:
        if col in oof.columns:
            valid = oof.dropna(subset=[col, "actual"])
            if len(valid) > 0:
                mape = np.mean(np.abs((valid["actual"] - valid[col]) / (valid["actual"] + 1e-10))) * 100
                results[col] = round(float(mape), 2)

    entry = {
        "date": str(date.today()),
        "oof_rows": len(oof),
        "mape_by_model": results,
    }
    history.append(entry)
    # Son 365 günü tut
    history = history[-365:]
    mape_log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"     [MAPE Log] {results}")
    return entry
