"""generate_postmortem.py — Faz 2 2a-3 (2026-07-13): günlük post-mortem CLI.

ADM + GDZ icin bir target_date'in post-mortem'ini uretir
(output/daily/<target_date>/postmortem_<edas>.{md,json}). Actual henuz
gelmemisse status="no_actuals" ile sessizce atlar (crash etmez).

Kullanim:
    python generate_postmortem.py                # dun (target_date=bugun-1)
    python generate_postmortem.py 2026-07-12      # belirli bir gun
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C
from monitoring.postmortem import write_postmortem


def _load_gdz_tenant():
    sys.path.insert(0, str(C.GDZ_LIVE_ROOT))
    import config_live_gdz as CG  # noqa: PLC0415
    return CG


def run_for_tenant(edas_id: str, tenant_config, output_dir: Path, target_date: str) -> None:
    out_dir = output_dir / "daily" / target_date
    result = write_postmortem(tenant_config, target_date, out_dir)
    status = result["status"]
    if status == "ok":
        print(f"  {edas_id}: post-mortem yazildi -> {out_dir / f'postmortem_{edas_id}.md'}")
    else:
        print(f"  {edas_id}: atlandi (status={status})")


if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else (date.today() - timedelta(days=1)).isoformat()
    print(f"Post-mortem uretiliyor: target_date={target_date}")

    run_for_tenant(C.EDAS_ID, C.TENANT, C.OUTPUT_DIR, target_date)

    try:
        CG = _load_gdz_tenant()
    except ImportError as e:
        print(f"GDZ config yuklenemedi ({e}) -- GDZ atlandi.")
    else:
        run_for_tenant(CG.EDAS_ID, CG.TENANT, CG.OUTPUT_DIR, target_date)
