"""Input klasör şeması için ADM tarayıcı regresyon testleri."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.data_scanner import find_csv_for_date, find_latest_csv, scan_available_days


AYDEM_FILE = "DemandaBereket_Aydem_Daily.csv"
GEDIZ_FILE = "DemandaBereket_Gediz_Daily.csv"


def _create_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test", encoding="utf-8")
    return path


def test_scans_year_month_and_padded_day_folders(tmp_path: Path):
    expected = _create_csv(tmp_path / "2026.07" / "12" / AYDEM_FILE)

    assert scan_available_days(tmp_path, "aydem") == [date(2026, 7, 11)]
    assert find_csv_for_date(tmp_path, date(2026, 7, 11), "aydem") == expected
    assert find_latest_csv(tmp_path, "aydem") == expected


def test_supports_unpadded_day_and_multiple_months(tmp_path: Path):
    _create_csv(tmp_path / "2026.07" / "1" / GEDIZ_FILE)
    latest = _create_csv(tmp_path / "2026.08" / "02" / GEDIZ_FILE)

    assert scan_available_days(tmp_path, "gediz") == [
        date(2026, 6, 30),
        date(2026, 8, 1),
    ]
    assert find_latest_csv(tmp_path, "gediz") == latest


def test_month_and_year_boundary_uses_folder_date(tmp_path: Path):
    expected = _create_csv(tmp_path / "2027.01" / "01" / AYDEM_FILE)

    assert find_csv_for_date(tmp_path, date(2026, 12, 31), "aydem") == expected
    assert scan_available_days(tmp_path, "aydem") == [date(2026, 12, 31)]


def test_new_structure_is_preferred_over_legacy(tmp_path: Path):
    expected = _create_csv(tmp_path / "2026.07" / "12" / AYDEM_FILE)
    _create_csv(tmp_path / "12.07" / AYDEM_FILE)

    assert find_csv_for_date(tmp_path, date(2026, 7, 11), "aydem") == expected
    assert scan_available_days(tmp_path, "aydem") == [date(2026, 7, 11)]


def test_unknown_region_is_reported(tmp_path: Path):
    with pytest.raises(ValueError, match="Bilinmeyen bölge"):
        scan_available_days(tmp_path, "unknown")
