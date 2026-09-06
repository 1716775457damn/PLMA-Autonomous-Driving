from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ABSOLUTE = re.compile(r"(?:[A-Za-z]:\\\\|/home/|/Users/)")


def main():
    required = ["README.md", "LICENSE", "CITATION.cff", "pyproject.toml", "requirements.txt", "src/plma/plma_legacy_reference.py"]
    failures = [f"missing required file: {f}" for f in required if not (ROOT / f).exists()]
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".yaml", ".yml", ".toml", ".cff", ".md"}:
            continue
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if EMAIL.search(text) and path.name != "CITATION.cff":
            failures.append(f"possible email address: {path.relative_to(ROOT)}")
        if ABSOLUTE.search(text) and path.name != "check_release.py":
            failures.append(f"possible absolute path: {path.relative_to(ROOT)}")
    if failures:
        print("Release validation failed:")
        print("\n".join(sorted(set(failures))))
        return 1
    print("Release validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
