"""Temporary launcher for the selected reference implementation.

The reference script is kept unchanged during the first organization pass.
Use this launcher only after supplying a compatible local dataset.
"""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "src" / "plma" / "plma_legacy_reference.py"), run_name="__main__")
