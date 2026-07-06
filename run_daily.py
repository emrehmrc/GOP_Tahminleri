"""
run_daily.py — ADM Günlük Tahmin Orkestratörü
==============================================
Her sabah çalıştırılır. 6 adımı sırayla çağırır.
Her adım idempotent: hata olursa o adımda durur, önceki adım zarar görmez.

Kullanım:
    python run_daily.py                    # normal — dün ingest, yarın tahmin
    python run_daily.py --target 2026-07-01  # teslim gününü manuel belirt
    python run_daily.py --skip-ingest      # ingest atla (test / tekrar çalıştırma)
    python run_daily.py --skip-weather     # hava çekme atla (cache var)
    python run_daily.py --dry-run          # 04+05+06 atla, sadece 01-03 kontrol

Log: logs/<YYYY-MM-DD>_run.log
"""

import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from datetime import date, datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "src"))
from run_context import start_run, archive_models, prune_archive, write_summary
from forecast_logger import write_forecast_log, rebuild_duckdb_views, backup_logs_zip
from scorecard import build_daily_scorecard, check_alerts

# ── Loglama ───────────────────────────────────────────────────────────────────
# Handler'ları start_run() kurar (UI ile ORTAK yol). Burada sadece logger referansı.
log = logging.getLogger("adm_live")


# ── Yardımcı ─────────────────────────────────────────────────────────────────
def run_step(name: str, fn, *args, **kwargs) -> dict:
    log.info(f"══ {name} BAŞLIYOR ══")
    t0 = datetime.now()
    try:
        result = fn(*args, **kwargs)
        elapsed = (datetime.now() - t0).total_seconds()
        log.info(f"══ {name} TAMAM ({elapsed:.0f}s) | {result}")
        return result
    except Exception:
        elapsed = (datetime.now() - t0).total_seconds()
        log.error(f"══ {name} HATA ({elapsed:.0f}s)\n{traceback.format_exc()}")
        raise


# ── Pipeline adımları ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ADM Günlük Tahmin Pipeline")
    parser.add_argument("--target", default=None,
                        help="Teslim günü YYYY-MM-DD (varsayılan: yarın = T+2)")
    parser.add_argument("--skip-ingest",  action="store_true", help="Adım 01 atla")
    parser.add_argument("--skip-weather", action="store_true", help="Adım 02 atla")
    parser.add_argument("--dry-run", action="store_true",
                        help="Sadece 01-03 çalıştır (tahmin üretme)")
    args = parser.parse_args()

    # Run kimliği + paylaşılan dosya loglaması (UI ile aynı primitif).
    ctx = start_run(target_date=args.target)

    summary = {"run_date": str(date.today()), "steps": {}}

    try:
        # Adım 01 — Ingest
        if not args.skip_ingest:
            result01 = run_step("01_INGEST", _step_import("01_ingest_actual").run)
        else:
            log.info("01_INGEST atlandı (--skip-ingest)")
            result01 = {"status": "skipped"}
        summary["steps"]["01_ingest"] = result01

        # Adım 02 — Weather
        if not args.skip_weather:
            result02 = run_step("02_WEATHER", _step_import("02_fetch_weather").run)
        else:
            log.info("02_WEATHER atlandı (--skip-weather)")
            result02 = {"status": "skipped"}
        summary["steps"]["02_weather"] = result02

        # Adım 03 — Features
        result03 = run_step("03_FEATURES", _step_import("03_build_features").run)
        summary["steps"]["03_features"] = result03

        if args.dry_run:
            log.info("DRY-RUN: 04-06 atlandı.")
            summary["status"] = "dry_run_ok"
            write_summary(ctx, summary["steps"], "dry_run_ok")
            return

        # Adım 04 — Predict
        result04 = run_step("04_PREDICT", _step_import("04_predict_48h").run)
        summary["steps"]["04_predict"] = result04

        # 04 başarılı → o run'ın modellerini arşivle (UI ile aynı davranış)
        try:
            archive_models(ctx)
            prune_archive()
        except Exception as e:
            log.warning(f"Model arşivleme hatası (tahmin etkilenmez): {e}")

        # Adım 05 — Post-process
        result05 = run_step("05_POSTPROCESS", _step_import("05_postprocess").run)
        summary["steps"]["05_postprocess"] = result05

        # Adım 06 — Deliver
        result06 = run_step("06_DELIVER", _step_import("06_deliver").run,
                            target_date=args.target)
        summary["steps"]["06_deliver"] = result06

        # Faz 0: forecast_log yaz + türetilmiş DuckDB view'ları + günlük yedek.
        try:
            summary["steps"]["forecast_log"] = write_forecast_log(ctx)
            rebuild_duckdb_views()
            backup_logs_zip()
        except Exception as e:
            log.warning(f"Forecast log yazımı hatası (teslimi etkilemez): {e}")

        # Faz 1: daily_scorecard + alarm kontrolü. (Eski `log_daily_mape` kümülatif
        # MAPE'si — teknik borç #2, roadmap §1.4 — kaldırıldı; yerini scorecard aldı.)
        try:
            summary["steps"]["scorecard"] = build_daily_scorecard()
            alerts = check_alerts()
            summary["steps"]["alerts"] = {"count": len(alerts)}
        except Exception as e:
            log.warning(f"Scorecard/alarm hatası (teslimi etkilemez): {e}")

        summary["status"] = "ok"
        log.info(f"\n✓ Pipeline tamamlandı. Teslim: {result06.get('output_file', '?')}")

    except Exception:
        summary["status"] = "error"
        log.error("Pipeline HATA ile durdu.")
    finally:
        write_summary(ctx, summary["steps"], summary.get("status", "error"))


# ── Adım import yardımcısı ────────────────────────────────────────────────────
import importlib.util


def _step_import(module_filename: str):
    """pipeline/<module_filename>.py'yi isim çakışması olmadan import et."""
    path = ROOT / "pipeline" / f"{module_filename}.py"
    spec = importlib.util.spec_from_file_location(module_filename, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    main()
