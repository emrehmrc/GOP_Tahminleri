"""ADM ve GDZ icin tarih eksenli ortak STLF Excel raporu.

Rapor forecast olustuktan hemen sonra EDAS bazinda guncellenebilir. Gerceklesenler
gunluk kaynak CSV'lerden, D+2 tahminler paylasilan teslim klasorundeki forecast
Excel'lerinden okunur. Ayni tarih her uc tabloda da ayni Excel sutunundadir.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config_live as C  # noqa: E402
from src.data_scanner import find_csv_for_date, scan_available_days  # noqa: E402
from src.output_paths import DELIVERY_ROOT, dated_output_path  # noqa: E402

HOURS = tuple(range(24))
REPORT_NAME = "STLF_LIVE_RAPOR.xlsx"
REGIONS = {
    "ADM": {"source_region": "aydem", "filename_token": "adm"},
    "GDZ": {"source_region": "gediz", "filename_token": "gdz"},
}

HEADER_FILL = PatternFill("solid", fgColor="4472C4")
HEADER_FONT = Font(color="FFFFFF", bold=True)
DESCRIPTION_FILL = PatternFill("solid", fgColor="D9EAF7")
AVERAGE_FILL = PatternFill("solid", fgColor="D9EAD3")
GREEN_FILL = PatternFill("solid", fgColor="C6EFCE")
YELLOW_FILL = PatternFill("solid", fgColor="FFEB9C")
RED_FILL = PatternFill("solid", fgColor="FFC7CE")
THIN = Border(
    left=Side(style="thin", color="B7B7B7"),
    right=Side(style="thin", color="B7B7B7"),
    top=Side(style="thin", color="B7B7B7"),
    bottom=Side(style="thin", color="B7B7B7"),
)

# Kullanici tarafindan istenen sabit yerlesim. A aciklama karti, B saat,
# C ve sonrasi tarih sutunlaridir.
TABLES = {
    "actual": {"header": 1, "first_hour": 2, "average": 26, "description": "A8:A10"},
    "forecast": {"header": 27, "first_hour": 28, "average": 52, "description": "A28:A30"},
    "deviation": {"header": 55, "first_hour": 56, "average": 80, "description": "A54:A56"},
}
DESCRIPTIONS = {
    "actual": (
        "1. GERÇEKLEŞEN (MWh)\n\nGünlük kaynak CSV'lerde bulunan saatlik "
        "gerçekleşen enerji değerleridir. Klasör tarihi veri tarihinden bir gün sonradır."
    ),
    "forecast": (
        "2. D+2 TAHMİN (MWh)\n\nPaylaşılan Çıktılar klasöründe saklanan, ilgili "
        "teslim gününe ait saatlik tahmin değerleridir."
    ),
    "deviation": (
        "3. MUTLAK SAPMA\n\nGerçekleşen ile D+2 tahmini arasındaki saatlik mutlak "
        "orandır: ABS(Gerçekleşen / Tahmin - 1)."
    ),
}


def _as_number(value) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip().replace(" ", "").replace(",", ".")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _series_from_hour_values(hours, values) -> pd.Series | None:
    frame = pd.DataFrame({"hour": hours, "value": values})
    frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce")
    frame["value"] = frame["value"].map(_as_number)
    frame = frame.dropna(subset=["hour", "value"])
    frame["hour"] = frame["hour"].astype(int)
    frame = frame[frame["hour"].isin(HOURS)].drop_duplicates("hour", keep="last")
    if set(frame["hour"]) != set(HOURS):
        return None
    return frame.set_index("hour")["value"].sort_index().astype(float)


def read_actual_csv(path: Path, expected_date: date | None = None) -> tuple[date, pd.Series]:
    """Gunluk gerceklesen CSV'sini tarih ve 24 saatlik seri olarak oku."""
    frame = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    date_col = next((col for col in frame.columns if str(col).strip().lower().startswith("starts")), None)
    value_col = next((col for col in frame.columns if "energy" in str(col).strip().lower()), None)
    if not date_col or not value_col:
        raise ValueError(f"Gerceklesen CSV semasi gecersiz: {path}")
    # Kaynak servis bazı günlerde `04.07.2026`, bazı günlerde
    # `4,07,2026` yazabiliyor. Tarih ayıracını normalize ederek ikisini de oku.
    normalized_timestamps = frame[date_col].astype(str).str.strip().str.replace(",", ".", regex=False)
    timestamps = pd.to_datetime(normalized_timestamps, format="%d.%m.%Y %H:%M", errors="coerce")
    valid_dates = timestamps.dropna().dt.date.unique().tolist()
    if len(valid_dates) != 1:
        raise ValueError(f"CSV tek veri gunu icermiyor: {path}")
    data_date = valid_dates[0]
    if expected_date is not None and data_date != expected_date:
        raise ValueError(f"CSV tarihi {data_date}, beklenen {expected_date}: {path}")
    series = _series_from_hour_values(timestamps.dt.hour, frame[value_col])
    if series is None:
        raise ValueError(f"CSV 0..23 saatlerini tam icermiyor: {path}")
    return data_date, series


def collect_actuals(input_root: Path, region: str, through: date) -> dict[date, pd.Series]:
    actuals: dict[date, pd.Series] = {}
    for data_date in scan_available_days(Path(input_root), region=region):
        if data_date > through:
            continue
        path = find_csv_for_date(Path(input_root), data_date, region=region)
        if path is None:
            continue
        try:
            parsed_date, series = read_actual_csv(path, expected_date=data_date)
            actuals[parsed_date] = series
        except (OSError, ValueError, UnicodeError) as exc:
            print(f"     [Uyari] Gerceklesen okunamadi: {path.name}: {exc}")
    return dict(sorted(actuals.items()))


def _forecast_candidates(delivery_root: Path, edas: str) -> list[Path]:
    token = REGIONS[edas]["filename_token"]
    candidates = []
    for path in Path(delivery_root).rglob("*.xlsx"):
        name = path.name.lower()
        if "forecast" in name and token in name and "regen" not in name:
            candidates.append(path)
    return sorted(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def read_forecast_excel(path: Path) -> tuple[date, pd.Series]:
    """Eski/yeni adlandirmadan bagimsiz olarak forecast Excel'ini oku."""
    try:
        frame = pd.read_excel(path, sheet_name="Tahmin")
    except ValueError:
        frame = pd.read_excel(path)
    frame.columns = [str(col).strip() for col in frame.columns]
    required = {"Saat", "Tahmin_MWh"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Forecast semasi gecersiz: eksik={sorted(required - set(frame.columns))}")

    target_date = None
    if "Tarih" in frame.columns:
        dates = pd.to_datetime(frame["Tarih"], errors="coerce").dropna().dt.date.unique().tolist()
        if dates:
            target_date = dates[0]
    if target_date is None:
        match = re.search(r"20\d{2}-\d{2}-\d{2}", path.name)
        if match:
            target_date = date.fromisoformat(match.group(0))
    if target_date is None:
        raise ValueError("Forecast hedef tarihi bulunamadi")

    series = _series_from_hour_values(frame["Saat"], frame["Tahmin_MWh"])
    if series is None:
        raise ValueError("Forecast 0..23 saatlerini tam icermiyor")
    return target_date, series


def collect_forecasts(delivery_root: Path, edas: str, through: date) -> dict[date, pd.Series]:
    """Ayni hedef gun birden cok kez varsa son kaydedilen dosyayi kullan."""
    forecasts: dict[date, pd.Series] = {}
    for path in _forecast_candidates(delivery_root, edas):
        try:
            target_date, series = read_forecast_excel(path)
        except (OSError, ValueError, KeyError) as exc:
            print(f"     [Uyari] Forecast okunamadi: {path.name}: {exc}")
            continue
        if target_date <= through:
            forecasts[target_date] = series
    return dict(sorted(forecasts.items()))


def build_report_data(
    input_root: Path,
    delivery_root: Path,
    edas: str,
    target_date: date,
) -> dict:
    edas = edas.upper()
    if edas not in REGIONS:
        raise ValueError(f"Bilinmeyen EDAS: {edas}")
    actuals = collect_actuals(input_root, REGIONS[edas]["source_region"], target_date)
    forecasts = collect_forecasts(delivery_root, edas, target_date)
    dates = sorted(set(actuals) | set(forecasts))
    if not dates:
        raise FileNotFoundError(f"{edas} icin rapor verisi bulunamadi")
    return {"edas": edas, "target_date": target_date, "dates": dates,
            "actuals": actuals, "forecasts": forecasts}


def _clear_sheet(ws) -> None:
    for merged in list(ws.merged_cells.ranges):
        ws.unmerge_cells(str(merged))
    if ws.max_row:
        ws.delete_rows(1, ws.max_row)
    if ws.max_column:
        ws.delete_cols(1, ws.max_column)


def _value_for(data: dict, table: str, day: date, hour: int) -> float | None:
    actual = data["actuals"].get(day)
    forecast = data["forecasts"].get(day)
    if table == "actual":
        return None if actual is None else float(actual.get(hour, np.nan))
    if table == "forecast":
        return None if forecast is None else float(forecast.get(hour, np.nan))
    if actual is None or forecast is None:
        return None
    actual_value = _as_number(actual.get(hour))
    forecast_value = _as_number(forecast.get(hour))
    if actual_value is None or forecast_value in (None, 0):
        return None
    return abs(actual_value / forecast_value - 1.0)


def _style_description(ws, cell_range: str, text: str) -> None:
    ws.merge_cells(cell_range)
    cell = ws[cell_range.split(":", 1)[0]]
    cell.value = text
    cell.fill = DESCRIPTION_FILL
    cell.font = Font(bold=True, color="1F1F1F", size=10)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = THIN
    for row in ws[cell_range]:
        for ranged_cell in row:
            ranged_cell.border = THIN


def write_sheet(ws, data: dict) -> None:
    _clear_sheet(ws)
    dates = data["dates"]
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "C2"
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 12
    for col in range(3, 3 + len(dates)):
        ws.column_dimensions[get_column_letter(col)].width = 14

    for table_name, layout in TABLES.items():
        header_row = layout["header"]
        first_hour_row = layout["first_hour"]
        average_row = layout["average"]
        _style_description(ws, layout["description"], DESCRIPTIONS[table_name])

        hour_header = ws.cell(header_row, 2, "Saat")
        hour_header.fill = HEADER_FILL
        hour_header.font = HEADER_FONT
        hour_header.alignment = Alignment(horizontal="center")
        hour_header.border = THIN
        for offset, day in enumerate(dates, start=3):
            cell = ws.cell(header_row, offset, day)
            cell.number_format = "dd.mm.yyyy"
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = Alignment(horizontal="center")
            cell.border = THIN

        column_values: dict[int, list[float]] = {col: [] for col in range(3, 3 + len(dates))}
        for hour in HOURS:
            row = first_hour_row + hour
            label = ws.cell(row, 2, f"{hour:02d}:00")
            label.font = Font(bold=True)
            label.alignment = Alignment(horizontal="center")
            label.border = THIN
            for col, day in enumerate(dates, start=3):
                value = _value_for(data, table_name, day, hour)
                cell = ws.cell(row, col)
                cell.border = THIN
                cell.alignment = Alignment(horizontal="right")
                if value is None or not np.isfinite(value):
                    continue
                cell.value = float(value)
                column_values[col].append(float(value))
                if table_name == "deviation":
                    cell.number_format = "0.00%"
                    cell.fill = GREEN_FILL if value < 0.03 else YELLOW_FILL if value < 0.06 else RED_FILL
                else:
                    cell.number_format = "#,##0.00"

        avg_label = ws.cell(average_row, 2, "ORTALAMA")
        avg_label.font = Font(bold=True)
        avg_label.fill = AVERAGE_FILL
        avg_label.border = THIN
        avg_label.alignment = Alignment(horizontal="center")
        for col in range(3, 3 + len(dates)):
            cell = ws.cell(average_row, col)
            values = column_values[col]
            cell.value = float(np.mean(values)) if values else None
            cell.number_format = "0.00%" if table_name == "deviation" else "#,##0.00"
            cell.font = Font(bold=True)
            cell.fill = AVERAGE_FILL
            cell.border = THIN

    ws.auto_filter.ref = f"B1:{get_column_letter(2 + len(dates))}26"
    ws.print_title_rows = "1:1"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.sheet_properties.pageSetUpPr.fitToPage = True


def update_workbook(report_path: Path, sheet_data: dict[str, dict]) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        wb = load_workbook(report_path)
    else:
        wb = Workbook()

    for edas in ("ADM", "GDZ"):
        if edas not in sheet_data:
            continue
        if edas in wb.sheetnames:
            ws = wb[edas]
        else:
            ws = wb.create_sheet(edas)
        write_sheet(ws, sheet_data[edas])

    if "Sheet" in wb.sheetnames and len(wb.sheetnames) > 1:
        default = wb["Sheet"]
        if default.max_row == 1 and default.max_column == 1 and default["A1"].value is None:
            wb.remove(default)
    ordered = [wb[name] for name in ("ADM", "GDZ") if name in wb.sheetnames]
    ordered.extend(ws for ws in wb.worksheets if ws.title not in {"ADM", "GDZ"})
    wb._sheets = ordered

    tmp = report_path.with_name(f"{report_path.stem}.tmp.xlsx")
    try:
        wb.save(tmp)
        wb.close()
        os.replace(tmp, report_path)
    except Exception:
        wb.close()
        tmp.unlink(missing_ok=True)
        raise
    return report_path


def _latest_forecast_date(edas: str | None = None) -> date:
    edases = [edas.upper()] if edas else ["ADM", "GDZ"]
    latest = []
    for name in edases:
        for path in _forecast_candidates(DELIVERY_ROOT, name):
            try:
                target, _ = read_forecast_excel(path)
                latest.append(target)
            except (OSError, ValueError, KeyError):
                continue
    if not latest:
        raise FileNotFoundError("Teslim klasorunde forecast Excel'i bulunamadi")
    return max(latest)


def run(target_date: str | date | None = None, edas: str | None = None) -> dict:
    """Hedef raporda bir EDAS'i veya mevcut iki EDAS'i deterministik yenile."""
    target = (
        date.fromisoformat(target_date) if isinstance(target_date, str)
        else target_date if isinstance(target_date, date)
        else _latest_forecast_date(edas)
    )
    requested = [edas.upper()] if edas else ["ADM", "GDZ"]
    invalid = set(requested) - set(REGIONS)
    if invalid:
        raise ValueError(f"Bilinmeyen EDAS: {sorted(invalid)}")

    sheet_data = {}
    skipped = {}
    for name in requested:
        data = build_report_data(C.LIVE_DATA_DIR, DELIVERY_ROOT, name, target)
        if target not in data["forecasts"]:
            skipped[name] = f"{target} tarihli forecast yok"
            continue
        sheet_data[name] = data
    if not sheet_data:
        raise FileNotFoundError(f"{target} icin guncellenecek forecast bulunamadi: {skipped}")

    report = dated_output_path(DELIVERY_ROOT, str(target), REPORT_NAME, create=True)
    update_workbook(report, sheet_data)
    print(f"     Rapor guncellendi: {report}")
    return {"status": "ok", "file": str(report), "target_date": str(target),
            "sheets_updated": list(sheet_data), "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="STLF LIVE ortak Excel raporunu guncelle")
    parser.add_argument("--target", default=None, help="Hedef tarih YYYY-MM-DD")
    parser.add_argument("--edas", choices=("ADM", "GDZ"), default=None)
    args = parser.parse_args()
    print(run(target_date=args.target, edas=args.edas))


if __name__ == "__main__":
    main()
