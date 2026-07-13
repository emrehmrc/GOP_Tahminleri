"""
monitoring/data_quality.py — ADM + GDZ ortak ingest veri-kalite kapısı (Faz 1, 2026-07-13).

Neden: 07-12 Pazar post-mortem incelemesinde ADM master'ında ingest sırasında
gözden kaçabilecek bir bozukluk şüphesi çıktı (aynı gün için tutarsız satır
sayısı) — ingest'te sadece negatif değer + <24 satır kontrolü vardı (bkz.
pipeline/01_ingest_actual.py:validate), sıfır-değer ve tarihsel-outlier
(robust-z) kontrolü hiç yoktu.

Bu modül SESSİZCE geçmez ama koşuyu da DURDURMAZ — ihlal bulunursa
logs/alerts/<date>_data_quality.json'a yazılır + çağıranın summary'sine
`data_quality` bloğu eklenir. Master'a zaten yazılmış bozuk bir gün varsa
OTOMATİK SİLİNMEZ; karar insanda (bkz. docs/MASTER_PLAN.md Faz 1 §3).

`evaluate_ingest_quality` saf/test edilebilir; `check_and_alert` onun
üzerine dosya yazma + loglama ekler (monitoring/scorecard.py:check_alerts
ile aynı desen).
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from monitoring.tenant_config import TenantConfig

_MAD_CONST = 1.4826


def evaluate_ingest_quality(
    new_day: pd.DataFrame,
    master_before: pd.DataFrame | None,
    target_col: str,
    date_col: str,
    hour_col: str,
    z_threshold: float = 4.0,
    lookback_days: int = 30,
) -> dict:
    """new_day: bugün ingest edilen (validate() SONRASI, deduplike/24-satır)
    günün satırları. master_before: bu gün eklenmeden ÖNCEKİ master (tarihsel
    karşılaştırma referansı — outlier kontrolü kendi kendiyle kıyaslamasın)."""
    issues: list[dict] = []
    day = new_day[date_col].iloc[0].date() if len(new_day) else None

    # 1. Saat tamlığı — validate() zaten <24'te raise ediyor, burada
    #    defense-in-depth (0-23 hepsi TAM bir kez var mı).
    hours = sorted(new_day[hour_col].unique().tolist())
    if hours != list(range(24)):
        missing = sorted(set(range(24)) - set(hours))
        extra = sorted(set(hours) - set(range(24)))
        issues.append({
            "type": "missing_or_extra_hours", "severity": "error",
            "detail": f"eksik saatler={missing} beklenmeyen saatler={extra}",
        })

    # 2. Duplicate (date, hour) — validate() zaten düşürüyor, burada
    #    upstream davranışı değişirse sessiz kalmasın diye tekrar bakılır.
    dupe_mask = new_day.duplicated(subset=[date_col, hour_col], keep=False)
    if dupe_mask.any():
        issues.append({
            "type": "duplicate_timestamp", "severity": "warning",
            "detail": f"{int(dupe_mask.sum())} duplicate satır",
        })

    # 3. Negatif / sıfır değer
    neg = int((new_day[target_col] < 0).sum())
    if neg:
        issues.append({"type": "negative_value", "severity": "error", "detail": f"{neg} negatif değer"})
    zero = int((new_day[target_col] == 0).sum())
    if zero:
        issues.append({"type": "zero_value", "severity": "warning", "detail": f"{zero} sıfır değer"})

    # 4. Robust-z outlier: her saat için son `lookback_days` günün aynı-saat
    #    medyanı/MAD'ına karşı (monitoring/scorecard.py:_add_robust_z ile aynı istatistik).
    if master_before is not None and not master_before.empty and day is not None:
        hist = master_before.copy()
        hist[date_col] = pd.to_datetime(hist[date_col])
        cutoff = pd.Timestamp(day) - pd.Timedelta(days=lookback_days)
        hist = hist[(hist[date_col] >= cutoff) & (hist[date_col] < pd.Timestamp(day))]

        outlier_hours = []
        for h in range(24):
            hist_h = hist.loc[hist[hour_col] == h, target_col].dropna()
            if len(hist_h) < 5:
                continue
            med = hist_h.median()
            mad = (hist_h - med).abs().median()
            if mad == 0:
                continue
            row = new_day.loc[new_day[hour_col] == h, target_col]
            if row.empty or pd.isna(row.iloc[0]):
                continue
            z = abs(row.iloc[0] - med) / (_MAD_CONST * mad)
            if z > z_threshold:
                outlier_hours.append({
                    "hour": int(h), "value": float(row.iloc[0]),
                    "median_30d": float(med), "robust_z": round(float(z), 2),
                })
        if outlier_hours:
            issues.append({
                "type": "outlier_vs_history", "severity": "warning",
                "detail": f"{len(outlier_hours)} saat robust-z>{z_threshold}",
                "hours": outlier_hours,
            })

    if any(i["severity"] == "error" for i in issues):
        status = "error"
    elif issues:
        status = "warning"
    else:
        status = "ok"

    return {"status": status, "date": str(day) if day else None, "issues": issues}


def check_and_alert(
    config: TenantConfig,
    new_day: pd.DataFrame,
    master_before: pd.DataFrame | None,
    target_col: str,
    date_col: str,
    hour_col: str,
    z_threshold: float = 4.0,
    lookback_days: int = 30,
) -> dict:
    """evaluate_ingest_quality + ihlal varsa logs/alerts/<date>_data_quality.json'a yaz."""
    result = evaluate_ingest_quality(
        new_day, master_before, target_col, date_col, hour_col,
        z_threshold=z_threshold, lookback_days=lookback_days,
    )
    if result["issues"]:
        config.alerts_dir.mkdir(parents=True, exist_ok=True)
        path = config.alerts_dir / f"{result['date']}_data_quality.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.getLogger(config.logger_name).warning(
            f"[DataQuality] {len(result['issues'])} sorun ({result['status']}) -> {path}"
        )
    return result
