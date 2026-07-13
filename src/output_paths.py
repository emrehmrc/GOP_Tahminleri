"""Tarihli output dosyalari icin YYYY.MM/D klasor duzeni ve geriye uyum."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

DATE_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")

# Musteriye/gonderilen teslim ciktilarinin (forecast xlsx, diagnostic html/json,
# STLF_LIVE_RAPOR.xlsx) yazildigi paylasilan harici kok. ADM ve GDZ ayni klasoru
# paylasir (dosya adlari zaten EDAS'a gore ayrisir, bkz. OUTPUT_FILENAME_TEMPLATE).
DELIVERY_ROOT = Path(
    r"C:\Users\Emre Hangul\MRC\MRC - 1.1.3_T&SI\_Economic_Dispatch_Modeling"
    r"\Dagıtılan Enerji Tahmini Internal Projesi\04_Proje Çıktıları\gönderilen veriler"
)


def date_from_name(name: str) -> str | None:
    match = DATE_RE.search(name)
    return match.group(0) if match else None


def date_folder(root: Path, value: str | date, create: bool = False) -> Path:
    d = date.fromisoformat(value) if isinstance(value, str) else value
    folder = Path(root) / d.strftime("%Y.%m") / str(d.day)
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def dated_output_path(root: Path, target_date: str, filename: str, create: bool = False) -> Path:
    return date_folder(root, target_date, create=create) / filename


def resolve_output_file(root: Path, filename: str) -> Path:
    """Yeni YYYY.MM/D yolu; yoksa gecis donemi flat/legacy yolu."""
    root = Path(root)
    ds = date_from_name(filename)
    if ds:
        candidate = dated_output_path(root, ds, filename)
        if candidate.exists():
            return candidate
    flat = root / filename
    if flat.exists():
        return flat
    matches = [p for p in root.glob(f"*/*/*{filename}") if p.parent.parent.name != "archive"]
    return matches[0] if matches else (dated_output_path(root, ds, filename) if ds else flat)


def glob_output_files(root: Path, pattern: str) -> list[Path]:
    root = Path(root)
    files = list(root.glob(pattern))
    files.extend(p for p in root.glob(f"*/*/{pattern}") if p.parent.parent.name != "archive")
    return sorted(set(files))
