"""
01_ingest_actual.py — Müşteri Gerçekleşme Verisi İngest (ADM/Aydem)
=====================================================================
LIVE_DATA_DIR / DD.MM / DemandaBereket_Aydem_Daily.csv dosyasını okur,
doğrular, master.parquet'e upsert eder.

Format:
    Asset Id;Starts dd.mm.YYYY HH:MM;Time zone;Energy MWh
    DemandaBereket_Aydem;29.06.2026 00:00;Europe/Istanbul;1406,877983
    ';' ayraç, ondalık ',', 24 satır = 1 gün.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from config_live import (
    LIVE_DATA_DIR, MASTER_PARQUET, ARCHIVE_DIR, OOF_HISTORY_PATH,
    RAW_TARGET_COL, RAW_DATE_COL, RAW_HOUR_COL,
)
from src.data_scanner import find_csv_for_date, find_latest_csv
from src.oof_feedback import update_oof_history
from src.forecast_logger import update_actuals_log


def load_source_csv(path: Path) -> pd.DataFrame:
    """CSV'yi format toleransı ile oku, Tarih+Saat+Hedef kolonlarına çevir.

    Desteklenen format:
      - ';' ayraç (birincil)
      - ondalık ',' (birincil) veya '.' (otomatik tanı)
      - BOM'lu veya BOM'suz UTF-8
      - Whitespace-tolerant header matching
    """
    if not path.exists():
        raise FileNotFoundError(f"Kaynak dosya bulunamadı: {path}")

    raw = path.read_bytes()
    has_bom = raw[:3] == b"\xef\xbb\xbf"

    # Ayraç tespiti: önce ';' dene, olmazsa ',' dene
    sample = raw.decode("utf-8-sig" if has_bom else "utf-8")[:2000]
    sep = ";"
    if sep not in sample and ";" not in sample:
        sep = ","

    df = pd.read_csv(
        path, sep=sep, decimal=",",
        encoding="utf-8-sig" if has_bom else "utf-8",
        dtype=str,
    )

    # Header normalization
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    header_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if "starts" in cl and ":" in cl:
            header_map[col] = "Starts dd.mm.YYYY HH:MM"
        elif "energy" in cl or "mwh" in cl:
            header_map[col] = "Energy MWh"
    df = df.rename(columns=header_map)

    required = ["Starts dd.mm.YYYY HH:MM", "Energy MWh"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Eksik kolonlar {path.name}: {missing}\n"
            f"Mevcut: {list(df.columns)}\n"
            f"Beklenen: Asset Id;Starts dd.mm.YYYY HH:MM;Time zone;Energy MWh"
        )

    # Decimal auto-detect
    energy_str = df["Energy MWh"].astype(str).str.strip()
    has_comma = bool(energy_str.str.contains(",").any())
    has_dot = bool(energy_str.str.contains(r"^\d+\.?\d*$").any())
    if has_comma and not has_dot:
        decimal_sep = ","
    elif has_dot and not has_comma:
        decimal_sep = "."
    else:
        decimal_sep = ","
    energy_vals = energy_str.str.replace(",", ".", regex=False).astype(float)

    date_str = df["Starts dd.mm.YYYY HH:MM"].astype(str).str.replace(",", ".", regex=False)
    dt = pd.to_datetime(date_str, format="%d.%m.%Y %H:%M")

    out = pd.DataFrame({
        RAW_DATE_COL: dt.dt.normalize(),
        RAW_HOUR_COL: dt.dt.hour,
        RAW_TARGET_COL: energy_vals,
    })
    return out


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Temel doğrulama: 24 saat var mı, negatif var mı, duplikasyon var mı."""
    dupes = df.duplicated(subset=[RAW_DATE_COL, RAW_HOUR_COL])
    if dupes.any():
        print(f"[Uyarı] {int(dupes.sum())} duplikasyon var, ilki alınıyor.")
        df = df[~dupes].copy()

    if not df[RAW_HOUR_COL].between(0, 23).all():
        bad = df[~df[RAW_HOUR_COL].between(0, 23)]
        raise ValueError(f"Saat aralığı dışı değer: {bad}")

    n = len(df)
    if n < 24:
        raise ValueError(f"Beklenen 24 saat, gelen: {n}. Veri eksik.")
    if n > 24:
        print(f"[Uyarı] 24'ten fazla satır ({n}), birden fazla gün olabilir.")

    neg = (df[RAW_TARGET_COL] < 0).sum()
    if neg > 0:
        raise ValueError(f"{neg} adet negatif tüketim var. Veriyi kontrol et.")

    return df


def check_gap(new_date, master: pd.DataFrame):
    """master'daki son tarih ile yeni gelen tarih arasında boşluk var mı uyar."""
    if master.empty:
        return
    last_date = master[RAW_DATE_COL].max().date()
    expected_next = last_date + pd.Timedelta(days=1)
    if new_date > expected_next:
        gap_days = (new_date - last_date).days - 1
        print(
            f"[UYARI] Veri boşluğu tespit edildi! Son master tarihi: {last_date}, "
            f"gelen: {new_date} ({gap_days} gün eksik olabilir)."
        )
    elif new_date <= last_date:
        print(f"[Bilgi] {new_date} zaten master'da var, üzerine yazılacak (idempotent).")


def append_to_master(new_df: pd.DataFrame) -> pd.DataFrame:
    """Yeni satırları master.parquet'e upsert et (Tarih+Saat anahtarıyla).
    Atomic write: önce .tmp, sonra rename → crash'e karşı korumalı."""
    if MASTER_PARQUET.exists():
        master = pd.read_parquet(MASTER_PARQUET)
        master[RAW_DATE_COL] = pd.to_datetime(master[RAW_DATE_COL])
    else:
        print(f"[Ingest] master.parquet yok, yeni oluşturulacak: {MASTER_PARQUET}")
        master = pd.DataFrame(columns=new_df.columns)

    check_gap(new_df[RAW_DATE_COL].iloc[0].date(), master)

    combined = pd.concat([master, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=[RAW_DATE_COL, RAW_HOUR_COL], keep="last")
    combined = combined.sort_values([RAW_DATE_COL, RAW_HOUR_COL]).reset_index(drop=True)

    MASTER_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MASTER_PARQUET.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp_path, index=False)
    tmp_path.rename(MASTER_PARQUET)
    return combined


def run(target_date: Optional[date] = None, source_name: str = "aydem") -> dict:
    """
    Adım 01 — ingest.

    Args:
        target_date: Verinin ait olduğu tarih. None = en son mevcut veri.
        source_name: "aydem" (GDZ ayrı bir repo).

    Returns:
        {"status": "ok", "source": "aydem", "date": "...", "rows_added": 24}
    """
    print(f"\n[01] Gerçekleşme verisi ingest ediliyor ({source_name})...")

    if target_date is None:
        csv_path = find_latest_csv(LIVE_DATA_DIR, source_name)
        if csv_path is None:
            raise FileNotFoundError(f"{LIVE_DATA_DIR} içinde {source_name} CSV bulunamadı.")
    else:
        csv_path = find_csv_for_date(LIVE_DATA_DIR, target_date, source_name)
        if csv_path is None:
            raise FileNotFoundError(f"{target_date} için {source_name} CSV bulunamadı.")

    print(f"     Kaynak: {csv_path}")

    raw = load_source_csv(csv_path)
    validated = validate(raw)

    added_date = validated[RAW_DATE_COL].iloc[0].date()
    master = append_to_master(validated)

    print(f"     Tarih: {added_date}  |  Satır: {len(validated)}  |  Master toplam: {len(master)}")

    # OOF feedback: dün tahmini ile bugünün actual'ını karşılaştır
    try:
        oof_result = update_oof_history(MASTER_PARQUET, ARCHIVE_DIR, OOF_HISTORY_PATH,
                                         RAW_TARGET_COL, RAW_DATE_COL, RAW_HOUR_COL)
        if oof_result.get("status") == "ok":
            print(f"     [OOF] MAPE: {oof_result.get('mape', '?')}")
    except Exception as e:
        print(f"     [OOF] Uyarı: {e}")

    # Faz 0: actuals_log D+1 yük dalgası (y_actual + data_quality_flag)
    try:
        al_result = update_actuals_log(validated)
        print(f"     [ActualsLog] {al_result}")
    except Exception as e:
        print(f"     [ActualsLog] Uyarı: {e}")

    return {
        "status": "ok",
        "source": source_name,
        "date": str(added_date),
        "rows_added": len(validated),
        "master_total": len(master),
    }


if __name__ == "__main__":
    import sys as _sys
    target = None
    if len(_sys.argv) > 1:
        target = date.fromisoformat(_sys.argv[1])
    result = run(target_date=target)
    print(result)
