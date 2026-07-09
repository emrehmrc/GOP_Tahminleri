"""
src/common.py — Ortak yardimci fonksiyonlar
==============================================
Tum pipeline step'lerinin ortak kullandigi yardimcilar.
"""
import sys
from pathlib import Path


def add_local_src_path(root: Path = None) -> None:
    if root is None:
        import inspect
        f = inspect.currentframe().f_back
        root = Path(f.f_globals.get("__file__", ".")).resolve().parent.parent
    for p in [str(root), str(root / "src")]:
        if p not in sys.path:
            sys.path.insert(0, p)
