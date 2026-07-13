import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui"))

import forecast_adjustment as fa


def _entry(hour, fc=1000.0, exp=1100.0, lo=900.0, hi=1080.0, n=60):
    return {"h": hour, "fc": fc, "exp": exp, "lo": lo, "hi": hi, "n": n}


def test_ai_recommendations_consider_all_hours_and_honor_limit():
    data = {"REC": [_entry(h) for h in range(24)]}
    all_hours = fa.build_recommendations(data, max_hours=24)
    limited = fa.build_recommendations(data, max_hours=9)
    assert set(all_hours["Saat"]) == set(range(24))
    assert len(limited) == 9


def test_ai_delta_is_bounded_and_requires_history():
    result = fa.build_recommendations({"REC": [
        _entry(17, exp=1400.0, n=60),
        _entry(18, exp=1400.0, n=10),
    ]})
    assert list(result["Saat"]) == [17]
    assert float(result.iloc[0]["Değişim (MWh)"]) == fa.DEFAULT_MAX_AI_DELTA_MWH


def test_small_in_band_gap_is_not_changed():
    result = fa.build_recommendations({"REC": [
        _entry(21, fc=1000.0, exp=1039.0, lo=900.0, hi=1100.0, n=60),
    ]})
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_preview_contains_confidence_explanation_and_old_new_values():
    result = fa.build_recommendations({"REC": [_entry(12)]})
    assert {"Eskisi (MWh)", "Yenisi (MWh)", "Güven", "Açıklama", "P95 Alt", "P95 Üst"}.issubset(result.columns)


def test_forecast_excel_update_is_hour_scoped(tmp_path):
    path = tmp_path / "forecast.xlsx"
    pd.DataFrame({
        "Tarih": ["2026-07-13"] * 24,
        "Saat": list(range(24)),
        "Tahmin_MWh": [1000.0] * 24,
    }).to_excel(path, index=False, sheet_name="Tahmin")
    fa._write_forecast_atomic(path, {17: 1040.0, 21: 1030.0})
    out = pd.read_excel(path, sheet_name="Tahmin").set_index("Saat")["Tahmin_MWh"]
    assert out.loc[17] == 1040.0
    assert out.loc[21] == 1030.0
    assert out.drop(index=[17, 21]).eq(1000.0).all()


def test_customer_date_is_forecast_target_not_send_date():
    import importlib.util
    spec = importlib.util.spec_from_file_location("email_report_test", ROOT / "pipeline" / "09_email_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.forecast_date_display("2026-07-13") == "13.07.2026"
