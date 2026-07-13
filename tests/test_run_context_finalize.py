"""Faz 1 (2026-07-13) — finalize_run hard-fail davranışı + write_summary(extra=...)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config_live as C  # noqa: E402
import run_context  # noqa: E402
import forecast_logger  # noqa: E402
import scorecard  # noqa: E402


def _ctx():
    return {
        "edas_id": "ADM", "run_id": "2026-07-13_testhash", "config_hash": "testhash",
        "issue_date": "2026-07-13", "target_date": "2026-07-14",
        "started_at": "2026-07-13T06:00:00",
    }


def test_write_summary_extra_merges_top_level(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "LOGS_DIR", tmp_path)
    path = run_context.write_summary(
        _ctx(), {"01_ingest": {"status": "ok"}}, "awaiting_approval",
        extra={"forecast_logged": True},
    )
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["status"] == "awaiting_approval"
    assert data["forecast_logged"] is True


def test_write_summary_without_extra_is_backward_compatible(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "LOGS_DIR", tmp_path)
    path = run_context.write_summary(_ctx(), {}, "dry_run_ok")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["status"] == "dry_run_ok"
    assert "forecast_logged" not in data


def test_finalize_run_hard_fail_marks_forecast_not_logged(monkeypatch):
    """write_forecast_log HATA fırlatırsa: forecast_logged=False, steps.forecast_log.status='error',
    ama reconcile/scorecard yine de denenir (soft-fail bağımsız kalmalı)."""
    def _boom(ctx):
        raise RuntimeError("simulated disk write failure")

    monkeypatch.setattr(forecast_logger, "write_forecast_log", _boom)
    monkeypatch.setattr(forecast_logger, "reconcile", lambda: {"status": "ok", "gaps": []})
    monkeypatch.setattr(scorecard, "build_daily_scorecard", lambda: {"status": "ok"})
    monkeypatch.setattr(scorecard, "check_alerts", lambda: [])

    steps = {}
    result = run_context.finalize_run(_ctx(), steps)

    assert result["forecast_logged"] is False
    assert steps["forecast_log"]["status"] == "error"
    assert "simulated disk write failure" in steps["forecast_log"]["error"]
    # soft-fail adımları forecast_log'un çökmesinden ETKİLENMEMİŞ olmalı
    assert steps["reconcile"]["status"] == "ok"
    assert steps["scorecard"]["status"] == "ok"


def test_finalize_run_happy_path_marks_forecast_logged_true(monkeypatch):
    monkeypatch.setattr(forecast_logger, "write_forecast_log", lambda ctx: {"status": "ok", "rows": 48})
    monkeypatch.setattr(forecast_logger, "rebuild_duckdb_views", lambda: {"status": "ok"})
    monkeypatch.setattr(forecast_logger, "backup_logs_zip", lambda: Path("dummy.zip"))
    monkeypatch.setattr(forecast_logger, "reconcile", lambda: {"status": "ok", "gaps": []})
    monkeypatch.setattr(scorecard, "build_daily_scorecard", lambda: {"status": "ok"})
    monkeypatch.setattr(scorecard, "check_alerts", lambda: [])

    steps = {}
    result = run_context.finalize_run(_ctx(), steps)

    assert result["forecast_logged"] is True
    assert steps["forecast_log"]["status"] == "ok"


def test_finalize_run_reconcile_failure_does_not_flip_forecast_logged(monkeypatch):
    """reconcile/scorecard patlarsa bile forecast_log başarılıysa forecast_logged True kalmalı
    (soft-fail adımlar hard-fail'i kirletmemeli)."""
    monkeypatch.setattr(forecast_logger, "write_forecast_log", lambda ctx: {"status": "ok", "rows": 48})
    monkeypatch.setattr(forecast_logger, "rebuild_duckdb_views", lambda: {"status": "ok"})
    monkeypatch.setattr(forecast_logger, "backup_logs_zip", lambda: Path("dummy.zip"))

    def _reconcile_boom():
        raise RuntimeError("reconcile patladi")

    monkeypatch.setattr(forecast_logger, "reconcile", _reconcile_boom)
    monkeypatch.setattr(scorecard, "build_daily_scorecard", lambda: {"status": "ok"})
    monkeypatch.setattr(scorecard, "check_alerts", lambda: [])

    steps = {}
    result = run_context.finalize_run(_ctx(), steps)

    assert result["forecast_logged"] is True
    assert steps["reconcile"]["status"] == "error"
