"""
data_scanner.py — tarih klasörlü günlük veri kaynağı tarayıcı
================================================================
Birincil yapı:
`02_Alınan Veriler/gdz-adm live/talep/YYYY.MM/DD/`

Gün klasörü hem `01` hem `1` biçiminde olabilir. Geçiş döneminde mevcut
`DD.MM/` klasörleri de geri uyumluluk amacıyla taranır.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

_YYYY_MM_RE = re.compile(r"^(20\d{2})\.(0[1-9]|1[0-2])$")
_DAY_RE = re.compile(r"^(0?[1-9]|[12]\d|3[01])$")
_LEGACY_DD_MM_RE = re.compile(r"^(0[1-9]|[12]\d|3[01])\.(0[1-9]|1[0-2])$")

_FILES = {
    "aydem": "DemandaBereket_Aydem_Daily.csv",
    "gediz": "DemandaBereket_Gediz_Daily.csv",
}


def _region_filename(region: str) -> str:
    try:
        return _FILES[region.lower()]
    except KeyError as exc:
        raise ValueError(f"Bilinmeyen bölge: {region!r}. Beklenen: {sorted(_FILES)}") from exc


def _parse_new_folder(month_folder: Path, day_folder: Path) -> Optional[date]:
    """`YYYY.MM/DD` (veya `YYYY.MM/D`) yolunu tarihe çevir."""
    month_match = _YYYY_MM_RE.fullmatch(month_folder.name)
    day_match = _DAY_RE.fullmatch(day_folder.name)
    if not month_match or not day_match:
        return None
    try:
        return date(
            int(month_match.group(1)),
            int(month_match.group(2)),
            int(day_match.group(1)),
        )
    except ValueError:
        return None


def _parse_legacy_folder(folder: Path) -> Optional[date]:
    """Eski `DD.MM` klasörünü, yıl geçişini de gözeterek tarihe çevir."""
    match = _LEGACY_DD_MM_RE.fullmatch(folder.name)
    if not match:
        return None
    day, month = int(match.group(1)), int(match.group(2))
    today = date.today()
    try:
        parsed = date(today.year, month, day)
    except ValueError:
        return None
    if parsed > today + timedelta(days=45):
        try:
            parsed = date(today.year - 1, month, day)
        except ValueError:
            return None
    return parsed


def _iter_source_folders(data_dir: Path) -> Iterator[tuple[date, Path]]:
    """Yeni yapıyı önce, eski tek-seviye yapıyı sonra üret."""
    if not data_dir.is_dir():
        return

    entries = sorted(data_dir.iterdir(), key=lambda path: path.name)
    for month_folder in entries:
        if not month_folder.is_dir() or not _YYYY_MM_RE.fullmatch(month_folder.name):
            continue
        for day_folder in sorted(month_folder.iterdir(), key=lambda path: path.name):
            if not day_folder.is_dir():
                continue
            folder_date = _parse_new_folder(month_folder, day_folder)
            if folder_date is not None:
                yield folder_date, day_folder

    for folder in entries:
        if not folder.is_dir():
            continue
        folder_date = _parse_legacy_folder(folder)
        if folder_date is not None:
            yield folder_date, folder


def _data_date_from_folder(folder_date: date) -> date:
    """Klasör tarihinin bir gün öncesi = verinin ait olduğu tarih."""
    return folder_date - timedelta(days=1)


def _candidate_folders(data_dir: Path, folder_date: date) -> list[Path]:
    """Bir tarih için yeni ve eski olası klasörleri tercih sırasıyla döndür."""
    month_folder = data_dir / folder_date.strftime("%Y.%m")
    candidates = [month_folder / folder_date.strftime("%d")]
    unpadded = month_folder / str(folder_date.day)
    if unpadded != candidates[0]:
        candidates.append(unpadded)
    candidates.append(data_dir / folder_date.strftime("%d.%m"))
    return candidates


def scan_available_days(data_dir: Path, region: str = "aydem") -> list[date]:
    """Tarih klasörlerini tara ve CSV içeren veri günlerini döndür."""
    filename = _region_filename(region)
    results: set[date] = set()
    for folder_date, folder in _iter_source_folders(Path(data_dir)):
        if (folder / filename).is_file():
            results.add(_data_date_from_folder(folder_date))
    return sorted(results)


def find_csv_for_date(data_dir: Path, target_date: date, region: str = "aydem") -> Optional[Path]:
    """Veri tarihi için `target + 1 gün` klasöründeki doğru CSV'yi bul."""
    filename = _region_filename(region)
    folder_date = target_date + timedelta(days=1)
    for folder in _candidate_folders(Path(data_dir), folder_date):
        csv_path = folder / filename
        if csv_path.is_file():
            return csv_path
    return None


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
    return [day for day in available if day > master_max_date]


def get_ingestion_candidates(
    data_dir: Path,
    master_path: Path,
    raw_date_col: str = "Tarih",
    region: str = "aydem",
) -> list[date]:
    """Master parquet'ten maksimum tarihi oku ve bekleyen günleri döndür."""
    try:
        master = pd.read_parquet(master_path)
        master_max = pd.to_datetime(master[raw_date_col]).max().date()
    except (FileNotFoundError, KeyError, ValueError):
        master_max = None
    return get_pending_days(data_dir, master_max, region)
