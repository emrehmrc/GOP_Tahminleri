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
    all_hours = fa.build_recommendations(data)
    limited = fa.build_recommendations(data, max_hours=9)
    assert fa.DEFAULT_MAX_AI_HOURS == 24
    assert set(all_hours["Saat"]) == set(range(24))
    assert len(limited) == 9


def test_ai_delta_is_bounded_at_300_and_low_history_is_visible():
    result = fa.build_recommendations({"REC": [
        _entry(17, exp=2000.0, n=60),
        _entry(18, exp=2000.0, n=10),
    ]})
    assert fa.DEFAULT_MAX_AI_DELTA_MWH == 300.0
    assert list(result["Saat"]) == [17, 18]
    assert result["Değişim (MWh)"].eq(300.0).all()
    assert result.set_index("Saat").loc[18, "Güven"] == "Düşük"


def test_small_in_band_gap_still_gets_a_recommendation():
    result = fa.build_recommendations({"REC": [
        _entry(21, fc=1000.0, exp=1039.0, lo=900.0, hi=1100.0, n=60),
    ]})
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert float(result.iloc[0]["Değişim (MWh)"]) == 25.35


def test_delta_never_exceeds_300_mwh_in_either_direction():
    result = fa.build_recommendations({"REC": [
        _entry(5, fc=1000.0, exp=2000.0),
        _entry(6, fc=1000.0, exp=0.0),
    ]})
    changes = result.set_index("Saat")["Değişim (MWh)"]
    assert changes.loc[5] == 300.0
    assert changes.loc[6] == -300.0


def test_preview_contains_confidence_explanation_and_old_new_values():
    result = fa.build_recommendations({"REC": [_entry(12)]})
    assert {"Eskisi (MWh)", "Yenisi (MWh)", "Güven", "Açıklama", "P95 Alt", "P95 Üst"}.issubset(result.columns)


def test_diagnostic_recommendations_start_from_latest_excel_values():
    data = {"REC": [
        _entry(0, fc=1000.0, exp=1100.0),
        _entry(1, fc=1000.0, exp=1100.0),
    ]}
    forecast = pd.DataFrame({
        "Tarih": ["2026-07-14", "2026-07-14"],
        "Saat": [0, 1],
        "Tahmin_MWh": [1200.0, 900.0],
    })

    synced = fa.sync_diagnostic_with_forecast(data, forecast)
    result = fa.build_recommendations(synced).set_index("Saat")

    assert data["REC"][0]["fc"] == 1000.0  # Girdi diagnostic'i mutate edilmez.
    assert result.loc[0, "Eskisi (MWh)"] == 1200.0
    assert result.loc[1, "Eskisi (MWh)"] == 900.0
    assert result.loc[0, "Değişim (MWh)"] < 0
    assert result.loc[1, "Değişim (MWh)"] > 0


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


def test_customer_email_template_uses_requested_subject_sender_card_and_test_recipient():
    import importlib.util
    spec = importlib.util.spec_from_file_location("email_report_template_test", ROOT / "pipeline" / "09_email_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    msg, _ = module._build_message("customer", "2026-07-15", module.CUSTOMER_TO, thread_state={})
    html = next(part for part in msg.walk() if part.get_content_type() == "text/html")
    images = [part for part in msg.walk() if part.get_content_maintype() == "image"]

    assert msg["Subject"] == "MRC-AYDEM Demo GÖP Tahmin Sonuçları"
    assert msg["From"] == "talep.tahmin@mrc-tr.com"
    assert module.CUSTOMER_TO == ["emre.hangul@mrc-tr.com"]
    assert "15.07.2026 tarihine dair tahminlerimiz ektedir" in html.get_payload(decode=True).decode("utf-8")
    assert "cid:kartvizit" in html.get_payload(decode=True).decode("utf-8")
    assert len(images) == 1
    assert images[0]["Content-ID"] == "<kartvizit>"


def test_customer_email_continues_reply_all_conversation_headers():
    import importlib.util
    spec = importlib.util.spec_from_file_location("email_report_thread_test", ROOT / "pipeline" / "09_email_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    previous = {
        "root_message_id": "<root@mrc-tr.com>",
        "last_message_id": "<previous@mrc-tr.com>",
        "references": ["<root@mrc-tr.com>"],
    }
    msg, next_state = module._build_message(
        "customer", "2026-07-16", ["emre.hangul@mrc-tr.com"], thread_state=previous,
    )

    assert msg["In-Reply-To"] == "<previous@mrc-tr.com>"
    assert msg["References"] == "<root@mrc-tr.com> <previous@mrc-tr.com>"
    assert next_state["root_message_id"] == "<root@mrc-tr.com>"
    assert next_state["last_message_id"] == msg["Message-ID"]


def test_unconfirmed_send_as_configuration_is_blocked(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("email_report_send_as_test", ROOT / "pipeline" / "09_email_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(module, "SMTP_USER", "login@example.com")
    monkeypatch.setattr(module, "FROM", "group@example.com")
    monkeypatch.setattr(module, "SMTP_SEND_AS_CONFIRMED", False)
    monkeypatch.setattr(module, "check_readiness", lambda: {
        "ready": True,
        "adm": {"target_date": "2026-07-15"},
        "gdz": {"target_date": "2026-07-15"},
    })

    result = module.run("customer")
    assert result["status"] == "skipped"
    assert "Send As" in result["reason"]


def test_customer_recipient_change_starts_a_fresh_conversation():
    import importlib.util
    spec = importlib.util.spec_from_file_location("email_report_recipient_thread_test", ROOT / "pipeline" / "09_email_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    test_thread = {
        "root_message_id": "<test-root@mrc-tr.com>",
        "last_message_id": "<test-last@mrc-tr.com>",
        "references": ["<test-root@mrc-tr.com>"],
        "to": ["emre.hangul@mrc-tr.com"],
    }
    live_recipients = ["emre.hangul@mrc-tr.com", "talep.tahmin@aydemenerji.com.tr"]
    msg, next_state = module._build_message(
        "customer", "2026-07-17", live_recipients, thread_state=test_thread,
    )

    assert msg["In-Reply-To"] is None
    assert msg["References"] is None
    assert next_state["root_message_id"] == msg["Message-ID"]
