"""backtest_walkforward.py — Walk-forward backtest + forecast_log backfill
=======================================================================
Her gün için 'o gün elinizde olan veriyle' (as-of, lookahead yok) 03-04-05-06
pipeline'ını yeniden koşturup forecst_log'a yazar. Böylece Faz 3 dashboard
ilk açılışta 7+ günlük geçmişle karşılaşır. DuckDB dedup view `backfill`
etiketini tanıyıp gerçek canlı run'lara öncelik verir.

Kullanim:
    python backtest_walkforward.py                          # 2026-07-01 .. 2026-07-07
    python backtest_walkforward.py --start 2026-06-01 --end 2026-07-07
    python backtest_walkforward.py --force                  # mevcut günleri yeniden uret
"""
from __future__ import annotations
import sys, json, shutil, importlib
from pathlib import Path
from datetime import datetime
import pandas as pd, numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C
from run_context import compute_config_hash
from forecast_logger import write_forecast_log, rebuild_duckdb_views
import asof_regen as AR


def walkforward_backtest(start_date: str = "2026-07-01", end_date: str = "2026-07-07",
                         force: bool = False) -> list[dict]:
    dates = pd.date_range(start_date, end_date)
    targets = [d.strftime("%Y-%m-%d") for d in dates]

    print(f"{'='*60}")
    print(f"WALK-FORWARD BACKTEST")
    print(f"Aralik: {start_date} -> {end_date}  ({len(targets)} gun)")
    print(f"EDAS: {C.EDAS_ID}")
    print(f"{'='*60}")

    existing_runs: set[str] = set()
    fc_dir = C.FORECAST_LOG_DIR / f"edas_id={C.EDAS_ID}"
    if fc_dir.exists():
        for td_dir in fc_dir.iterdir():
            if td_dir.is_dir() and td_dir.name.startswith("target_date="):
                existing_runs.add(td_dir.name.split("=", 1)[1])

    todo = [t for t in targets if t not in existing_runs or force]
    skip = [t for t in targets if t not in todo]
    if skip:
        print(f"Zaten mevcut: {len(skip)} gun  ({skip[0]}..{skip[-1]})")
        for s in skip:
            print(f"  - {s}")
    if not todo:
        print("Tum gunler mevcut. --force ile yeniden uretebilirsiniz.")
        return []

    print(f"Uretilecek: {len(todo)} gun  ({todo[0]}..{todo[-1]})")

    # AR.regen_one() artik tamamen sandbox'li (bkz. asof_regen.py docstring) —
    # canli dosyalara hic dokunmadigi icin ayrica backup/restore gerekmiyor.
    # Run context yedegi (04_predict cagiriyor — LIVE_FILES icinde degil)
    ctx_bak = None
    if C.RUN_CONTEXT_PATH.exists():
        import json as _json
        ctx_bak = _json.loads(C.RUN_CONTEXT_PATH.read_text(encoding="utf-8"))

    config_hash = compute_config_hash()
    results: list[dict] = []

    try:
        for i, t in enumerate(todo):
            stamp = f"[{i+1}/{len(todo)}]"
            print(f"\n{'='*60}")
            print(f"{stamp} {t}")
            print(f"{'='*60}")

            # Synthetic ctx: 04_predict get_run_context()->ctx["started_at"] kullanir
            run_id = f"{t}_backfill_{config_hash}"
            started_at = datetime.now().isoformat(timespec="seconds")
            ctx = {
                "edas_id": C.EDAS_ID,
                "run_id": run_id,
                "config_hash": config_hash,
                "issue_date": t,
                "target_date": t,
                "started_at": started_at,
            }
            C.RUN_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
            C.RUN_CONTEXT_PATH.write_text(
                json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            try:
                result = AR.regen_one(t)
                # regen_one exception firlatmazsa basarilidir (return dict'indeki
                # "status" anahtarina guvenilmez — hata durumunda exception atar)
                # ONEMLI: postproc_path/meta_path acikca verilmezse write_forecast_log()
                # GERCEK (bu regen'le ilgisiz, bayat) canli postproc/meta'yi okur —
                # regen_one() sandbox'tan bu run'a ait *_REGEN dosyalarini kopyaliyor,
                # onlari kullanmaliyiz (bkz. asof_regen.py:regen_one).
                fc_result = write_forecast_log(
                    ctx,
                    postproc_path=Path(result["models_path"]) if result.get("models_path") else None,
                    meta_path=Path(result["meta_path"]) if result.get("meta_path") else None,
                    source="backfill",
                )
                n_rows = fc_result.get("rows", 0)
                print(f"       forecast_log: {fc_result.get('status')} ({n_rows} satir)")
                results.append({"target": t, "status": "ok", "forecast_log_rows": n_rows})
            except Exception as e:
                import traceback
                print(f"       HATA: {e}")
                traceback.print_exc()
                results.append({"target": t, "status": "error", "error": str(e)[:200]})

    finally:
        # Restore orijinal run_context.json (bu script'in dogrudan yazdigi tek
        # canli dosya — 04_predict'in kendi get_run_context()'i icin gerekli).
        if ctx_bak is not None:
            C.RUN_CONTEXT_PATH.write_text(
                json.dumps(ctx_bak, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        elif C.RUN_CONTEXT_PATH.exists():
            C.RUN_CONTEXT_PATH.unlink()
        print(f"\n[geri yukle] run_context.json restore edildi")

    # Ozet
    ok_cnt = sum(1 for r in results if r["status"] == "ok")
    err_cnt = sum(1 for r in results if r["status"] != "ok")

    print(f"\n{'='*60}")
    print(f"BACKTEST SONUCU")
    print(f"{'='*60}")
    print(f"Basarili: {ok_cnt}  |  Hata: {err_cnt}")
    if ok_cnt:
        total_rows = sum(r.get("forecast_log_rows", 0) for r in results if r["status"] == "ok")
        print(f"Toplam forecast_log satiri: {total_rows}")

    # DuckDB view'larini yeniden kur (backfill run'lar dahil)
    print(f"\nDuckDB view'lari yeniden kuruluyor...")
    try:
        r = rebuild_duckdb_views()
        print(f"DuckDB: {r}")
    except Exception as e:
        print(f"DuckDB view hatasi: {e}")

    print(f"\nDone.")
    return results


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Walk-forward backtest")
    p.add_argument("--start", default="2026-07-01")
    p.add_argument("--end", default="2026-07-07")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    walkforward_backtest(args.start, args.end, args.force)
