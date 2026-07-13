"""Dashboard hot-reload/config cache regression tests."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "ui"

for directory in (ROOT, ROOT / "src", UI_DIR):
    value = str(directory)
    if value not in sys.path:
        sys.path.insert(0, value)


def test_dashboard_recovers_from_stale_config_live_module():
    """Streamlit rerun'u eski config nesnesini tutsa bile yeni sabitler yuklenir."""
    import config_live
    import dashboard_common

    new_names = (
        "INGEST_OUTLIER_Z_THRESHOLD",
        "INGEST_OUTLIER_LOOKBACK_DAYS",
        "GDZ_LIVE_ROOT",
    )
    for name in new_names:
        config_live.__dict__.pop(name, None)

    refreshed = dashboard_common.refresh_adm_config()
    assert refreshed.INGEST_OUTLIER_Z_THRESHOLD == 4.0
    assert refreshed.INGEST_OUTLIER_LOOKBACK_DAYS == 30
    assert Path(refreshed.GDZ_LIVE_ROOT).resolve() == (ROOT.parent / "gdz talep" / "live").resolve()

    ingest = dashboard_common.import_pipeline_step("01_ingest_actual")
    email_report = dashboard_common.import_pipeline_step("09_email_report")

    assert ingest.INGEST_OUTLIER_Z_THRESHOLD == 4.0
    assert ingest.INGEST_OUTLIER_LOOKBACK_DAYS == 30
    assert Path(email_report.GDZ_LIVE_ROOT).resolve() == Path(refreshed.GDZ_LIVE_ROOT).resolve()
