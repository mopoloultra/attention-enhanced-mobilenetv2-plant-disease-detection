"""Strip notebook outputs, normalize paths, and write model checksums."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = os.environ.get("LEGACY_SOURCE_ROOT", "")


def portable_path(value: str) -> str:
    normalized = value.replace("/", "\\")
    if not SOURCE_ROOT:
        return value
    source = SOURCE_ROOT.replace("/", "\\")
    if not normalized.casefold().startswith(source.casefold()):
        return value
    relative = normalized[len(source):].lstrip("\\")
    if relative.casefold().startswith("result\\lite_models\\"):
        relative = "models\\tflite\\" + relative[len("result\\lite_models\\"):]
    elif relative.casefold().startswith("result\\"):
        relative = "models\\h5\\" + relative[len("result\\"):]
    elif relative.casefold().startswith("plantvillage\\"):
        relative = "data\\PlantVillage\\" + relative[len("plantvillage\\"):]
    return relative.replace("\\", "/")


def sanitize_json_value(value):
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, str):
        return portable_path(value)
    return value


def clean_notebooks() -> None:
    replacements = {
        "PlantVillage/train": "data/PlantVillage/train",
        "PlantVillage/test": "data/PlantVillage/test",
        "PlantVillage/val": "data/PlantVillage/val",
        "original_dataset_dir = 'PlantVillage'": "original_dataset_dir = 'data/PlantVillage_raw'",
        "base_dir = 'PlantVillage_Split'": "base_dir = 'data/PlantVillage'",
        "result/best_": "models/h5/best_",
        "./result\\\\lite_models": "./models/tflite",
        "model_dir = \"./result\"": "model_dir = \"./models/h5\"",
        "output_dir = os.path.join(model_dir, \"lite_models\")": "output_dir = \"./models/tflite\"",
    }
    for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            cell["metadata"] = {}
            source = "".join(cell.get("source", []))
            for old, new in replacements.items():
                source = source.replace(old, new)
            cell["source"] = source.splitlines(keepends=True)
        notebook["metadata"] = {
            key: value for key, value in notebook.get("metadata", {}).items()
            if key in {"kernelspec", "language_info"}
        }
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def sanitize_outputs() -> None:
    candidates = [ROOT / "analysis" / "model_inspection.json"]
    candidates.extend((ROOT / "analysis" / "outputs").glob("*.json"))
    for path in candidates:
        value = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(
            json.dumps(sanitize_json_value(value), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    for path in (ROOT / "analysis" / "outputs").glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        if "paths" in values:
            values["paths"] = np.asarray(
                [portable_path(str(item)) for item in values["paths"]],
                dtype=values["paths"].dtype,
            )
        np.savez_compressed(path, **values)


def write_checksums() -> None:
    rows = []
    models_root = ROOT / "models"
    for path in sorted(models_root.rglob("*")):
        if path.is_file() and path.name not in {"README.md", "SHA256SUMS"}:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            rows.append(f"{digest.hexdigest()}  {path.relative_to(models_root).as_posix()}")
    (models_root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="ascii")


if __name__ == "__main__":
    clean_notebooks()
    sanitize_outputs()
    write_checksums()
    print("Repository assets prepared.")
