"""Portable repository paths with optional environment-variable overrides."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(os.environ.get("PLANTVILLAGE_DIR", REPO_ROOT / "data" / "PlantVillage"))
H5_ROOT = Path(os.environ.get("H5_MODEL_DIR", REPO_ROOT / "models" / "h5"))
TFLITE_ROOT = Path(os.environ.get("TFLITE_MODEL_DIR", REPO_ROOT / "models" / "tflite"))
OUTPUT_DIR = REPO_ROOT / "analysis" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

