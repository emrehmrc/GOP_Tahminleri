#!/usr/bin/env python3
"""
diagnostic.py — ADM ve GDZ icin interaktif HTML diagnostic (Chart.js)
Kullanim:
    python diagnostic.py                    # ADM (varsayilan)
    python diagnostic.py --edas GDZ         # GDZ
"""
import sys, subprocess, argparse
from pathlib import Path

ROOT = Path(__file__).parent

def run_adm():
    script = ROOT / "pipeline" / "08_diagnostic_html.py"
    return subprocess.run([sys.executable, str(script)], cwd=str(ROOT)).returncode

def run_gdz():
    """GDZ diagnostic scriptini subprocess ile calistir"""
    GDZ_DIR = ROOT.parent / "gdz talep" / "live"
    if not GDZ_DIR.exists():
        print(f"HATA: GDZ dizini bulunamadi: {GDZ_DIR}")
        return 1
    gdz_script = GDZ_DIR / "pipeline" / "08_diagnostic_html.py"
    if not gdz_script.exists():
        print(f"HATA: GDZ diagnostic scripti bulunamadi: {gdz_script}")
        return 1
    print(f"  GDZ diagnostic: {gdz_script}")
    return subprocess.run([sys.executable, str(gdz_script)], cwd=str(GDZ_DIR)).returncode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STLF Diagnostic HTML generator")
    parser.add_argument("--edas", choices=["ADM", "GDZ"], default="ADM", help="EDAS (ADM/GDZ)")
    args = parser.parse_args()
    if args.edas == "ADM":
        sys.exit(run_adm())
    else:
        sys.exit(run_gdz())
