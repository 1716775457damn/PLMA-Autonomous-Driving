from pathlib import Path

def test_release_layout():
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").exists()
    assert (root / "LICENSE").exists()
    assert (root / "src" / "plma" / "plma_legacy_reference.py").exists()
