"""
src/common.py — Ortak yardimci fonksiyonlar
==============================================
Tum pipeline step'lerinin ortak kullandigi yardimcilar.
"""
import sys
from pathlib import Path


def add_local_src_path(root: Path) -> None:
    """ROOT/src'yi sys.path'e ekle (tekrar eklerse yok sayar)."""
    for p in [str(root), str(root / "src")]:
        if p not in sys.path:
            sys.path.insert(0, p)
