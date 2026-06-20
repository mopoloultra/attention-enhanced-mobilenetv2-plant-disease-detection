"""Verify model files against models/SHA256SUMS."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures = []
    for line in (MODELS / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = MODELS / relative
        actual = sha256(path)
        if actual != expected:
            failures.append(relative)
        else:
            print(f"OK  {relative}")
    if failures:
        raise SystemExit("Checksum mismatch: " + ", ".join(failures))


if __name__ == "__main__":
    main()

