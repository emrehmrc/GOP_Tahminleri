"""
data_scanner.py — DD.MM subfolder veri kaynağı tarayıcı
=========================================================
OneDrive'daki `02_Alınan Veriler/gdz-adm live/talep/DD.MM/` yapısını tarar.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

_DD_MM_RE = re.compile(r"^(\d{2})\.(\d{2})$")

_FILES = {
    "aydem": "DemandaBereket_Aydem_Daily.csv",
    "gediz": "DemandaBereket_Gediz_Daily.csv",
}


def _parse_folder(folder: Path) -> Optional[date]:
    """DD.MM klasör adını bu yıl içinde bir `date`'e çevir."""
    m = _DD_MM_RE.match(folder.name)
    if not m:
        return None
    day, month = int(m.group(1)), int(m.group(2))
    today = date.today()
    try:
        return date(today.year, month, day)
    except ValueError:
        return None


def _data_date_from_folder(folder_date: date) -> date:
    """Klasör tarihinin bir gün öncesi = verinin ait olduğu tarih."""
    return folder_date - timedelta(days=1)


def scan_available_days(data_dir: Path, region: str = "aydem") -> list[date]:
    """LIVE_DATA_DIR'daki tüm DD.MM klasörlerini tara, veri tarihlerini döndür."""
    results: list[date] = []
    if not data_dir.is_dir():
        return results
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        folder_d = _parse_folder(entry)
        if folder_d is None:
            continue
        data_d = _data_date_from_folder(folder_d)
        csv_file = entry / _FILES[region]
        if csv_file.is_file():
            results.append(data_d)
    return sorted(results)


def find_csv_for_date(data_dir: Path, target_date: date, region: str = "aydem") -> Optional[Path]:
    """Verilen tarih için doğru subfolder ve CSV'yi bul."""
    folder_date = target_date + timedelta(days=1)
    folder_name = folder_date.strftime("%d.%m")
    csv_path = data_dir / folder_name / _FILES[region]
    return csv_path if csv_path.is_file() else None


def find_latest_csv(data_dir: Path, region: str = "aydem") -> Optional[Path]:
    """En son mevcut veriyi bul."""
    days = scan_available_days(data_dir, region)
    if not days:
        return None
    return find_csv_for_date(data_dir, max(days), region)


def get_pending_days(data_dir: Path, master_max_date: Optional[date], region: str = "aydem") -> list[date]:
    """Ingest edilmemiş günleri tespit et: data'da var, master'da yok."""
    available = scan_available_days(data_dir, region)
    if not available:
        return []
    if master_max_date is None:
        return available
    return [d for d in available if d > master_max_date]


def get_ingestion_candidates(
    data_dir: Path,
    master_path: Path,
    raw_date_col: str = "Tarih",
    region: str = "aydem",
) -> list[date]:
    """Master parquet'ten max tarihi oku, pending günleri döndür."""
    try:
        master = pd.read_parquet(master_path)
        master_max = pd.to_datetime(master[raw_date_col]).max().date()
    except (FileNotFoundError, KeyError, ValueError):
        master_max = None
    return get_pending_days(data_dir, master_max, region)
