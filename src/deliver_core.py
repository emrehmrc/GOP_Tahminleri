"""
deliver_core.py — 06_deliver.py ortak çekirdeği (Faz 3, 2026-07-13)

ADM ve GDZ'nin 06_deliver.py'si artık ince kabuk: kendi T+2 seçim
stratejisini (ADM: date.today()+1'e göre tarih filtresi + boşsa 48h
fallback; GDZ: horizon_day kolonundan T2_yarin/T+2 filtresi + boşsa
raise) ve kolon adlarını buraya parametre olarak geçirip run_delivery()'yi
çağırır. İki tenant'ın T+2 seçim mantığı KASITLI OLARAK ortaklaştırılmadı
(davranış farkı var, walkforward A/B kanıtı olmadan birleştirilmez —
bkz. docs/MASTER_PLAN.md governance) — sadece boilerplate (sanity check,
excel yazımı, arşivleme) tek kaynağa indirildi. diagnostic_core.py'nin
izlediği desenin aynısı.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

SANITY_BAND_MARGIN = 0.25   # 7-gün aynı-saat bandının ±%25 dışı -> flag
SANITY_FLAT_RATIO = 0.15    # tahmin 24h std'si son 7g günlük std ortalamasının altında bu oranın -> flag


def sanity_check(
    output_df: pd.DataFrame,
    master_path: Path,
    date_col: str,
    target_col: str,
    hour_col: str | None = None,
) -> list:
    """Teslim öncesi son kontrol: 7-gün aynı-saat bandı, düzlük, NaN/negatif.

    Engellemez (pipeline'ı durdurmaz) — sadece loglara ve summary.json'a
    GÖRÜNÜR flag düşer. `hour_col=None` ise (GDZ şeması) saat date_col'dan
    `.dt.hour` ile türetilir; `hour_col` verilirse (ADM şeması) doğrudan
    okunur — iki orijinal implementasyonla birebir aynı davranış.
    """
    flags = []

    neg = output_df["Tahmin_MWh"] < 0
    if neg.any():
        flags.append({"type": "negative", "n": int(neg.sum()),
                       "hours": output_df.loc[neg, "Saat"].tolist()})
    nan = output_df["Tahmin_MWh"].isna()
    if nan.any():
        flags.append({"type": "nan", "n": int(nan.sum()),
                       "hours": output_df.loc[nan, "Saat"].tolist()})

    if not master_path.exists():
        return flags

    if hour_col:
        master = pd.read_parquet(master_path, columns=[date_col, hour_col, target_col])
        master[date_col] = pd.to_datetime(master[date_col])
        hour_series_col = hour_col
    else:
        master = pd.read_parquet(master_path)
        master[date_col] = pd.to_datetime(master[date_col])
        master["Saat"] = master[date_col].dt.hour
        hour_series_col = "Saat"

    last_date = master[date_col].max()
    recent = master[master[date_col] > last_date - pd.Timedelta(days=7)]
    if recent.empty:
        return flags

    by_hour = recent.groupby(hour_series_col)[target_col].agg(["min", "max"])
    out_of_band = []
    for _, row in output_df.iterrows():
        h = int(row["Saat"])
        if h not in by_hour.index or pd.isna(row["Tahmin_MWh"]):
            continue
        lo, hi = by_hour.loc[h, "min"], by_hour.loc[h, "max"]
        margin = (hi - lo) * SANITY_BAND_MARGIN if hi > lo else hi * SANITY_BAND_MARGIN
        if not (lo - margin <= row["Tahmin_MWh"] <= hi + margin):
            out_of_band.append(h)
    if out_of_band:
        flags.append({"type": "out_of_band", "n": len(out_of_band), "hours": out_of_band,
                       "margin_pct": SANITY_BAND_MARGIN * 100})

    pred_std = output_df["Tahmin_MWh"].std()
    recent_daily_std = recent.groupby(recent[date_col].dt.date)[target_col].std().mean()
    if pd.notna(recent_daily_std) and recent_daily_std > 0 and pred_std < SANITY_FLAT_RATIO * recent_daily_std:
        flags.append({"type": "flat", "pred_std": round(float(pred_std), 1),
                       "recent_daily_std_avg": round(float(recent_daily_std), 1)})

    return flags


def run_delivery(
    *,
    postproc_path: Path,
    select_t2: Callable[[pd.DataFrame], tuple[pd.DataFrame, str]],
    master_path: Path,
    date_col: str,
    target_col: str,
    delivery_root: Path,
    archive_dir: Path,
    output_filename_template: str,
    dated_output_path_fn: Callable,
    log: logging.Logger,
    hour_col: str | None = None,
    label: str = "",
    issue_date: str | date | None = None,
) -> dict:
    """Adım 06 — teslim çıktısı yaz (ADM+GDZ ortak çekirdek).

    Args:
        select_t2: `df -> (t2_df, target_str)` — tenant'a özel T+2 seçim
            stratejisi (ADM: tarih filtresi + 48h fallback; GDZ: horizon_day
            filtresi + raise). Kasıtlı olarak tenant tarafında kalır.
        label: konsol print önekinde tenant adı ("" ADM, "GDZ " GDZ gibi).
        issue_date: bkz. orijinal 06_deliver.py docstring'i — asof_regen
            gibi geçmişteymişiz gibi çalışan script'ler için.

    Returns:
        {"status": "ok", "output_file": "...", "target_date": "...", "n_hours": N,
         "sanity_flags": [...]}
    """
    print(f"\n[06] {label}müşteri çıktısı hazırlanıyor...")
    issue_dt = pd.Timestamp(issue_date).date() if issue_date is not None else date.today()

    df = pd.read_parquet(postproc_path)
    if "Datetime" not in df.columns:
        raise ValueError("Datetime kolonu bulunamadı (postprocessed_predictions.parquet)")
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    t2_df, target_str = select_t2(df)
    print(f"     Teslim günü: {target_str}  ({len(t2_df)} saat)")

    output_df = pd.DataFrame({
        "Tarih": t2_df["Datetime"].dt.date,
        "Saat": t2_df["Datetime"].dt.hour,
        "Tahmin_MWh": t2_df["Final_Pred"].round(2),
    })
    output_df = output_df.sort_values(["Tarih", "Saat"]).reset_index(drop=True)

    sanity_flags = sanity_check(output_df, master_path, date_col, target_col, hour_col=hour_col)
    if sanity_flags:
        log.warning(f"[06] SANITY UYARI — teslim şüpheli olabilir: {sanity_flags}")
        print(f"     [Sanity] UYARI: {sanity_flags}")
    else:
        print("     [Sanity] OK")

    filename = output_filename_template.format(date=target_str)
    output_path = dated_output_path_fn(delivery_root, target_str, filename, create=True)
    output_df.to_excel(output_path, index=False, sheet_name="Tahmin")
    print(f"     Teslim dosyası: {output_path.name}")

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{issue_dt}_run_{target_str}_full48h.parquet"
    archive_path = archive_dir / archive_name
    df_archive = df.copy()
    df_archive["issue_date"] = pd.Timestamp(issue_dt)
    df_archive.to_parquet(archive_path, index=False)
    print(f"     Arşiv:         {archive_path.name}")

    return {
        "status": "ok",
        "output_file": str(output_path),
        "target_date": target_str,
        "n_hours": len(t2_df),
        "sanity_flags": sanity_flags,
    }
