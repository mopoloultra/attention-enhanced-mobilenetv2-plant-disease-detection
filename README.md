# Attention-Enhanced MobileNetV2 for Plant Disease Detection

Reproducibility package for **“An Efficient Attention-Enhanced MobileNetV2 Framework for Plant Disease Detection on Resource-Constrained Devices.”**

The study evaluates a single CBAM block placed after the final MobileNetV2 convolutional feature map and before global average pooling. The package includes the archived training notebooks, saved H5 and TensorFlow Lite models, dataset-split notebook, validation scripts, prediction arrays, statistical analysis, FLOP profiles, Grad-CAM evidence, and manuscript source.

## Main reported results

- Dataset: 54,306 PlantVillage images in 38 classes.
- Directory split: 70% train, 15% validation, 15% test, using `random_state=42` per class.
- MobileNetV2 H5 accuracy: 96.78%.
- MobileNetV2 + CBAM H5 accuracy: 97.17%.
- Paired TFLite accuracy difference: +0.64 percentage points (95% CI 0.31-0.96; exact McNemar `p = 1.42e-4`).
- MobileNetV2 + CBAM complexity: 4.02 million parameters, 0.604 GFLOPs (approximately 0.302 GMACs).
- Quantized MobileNetV2 + CBAM: 4.07 MiB and 96.70% accuracy.
- Laptop CPU median latency: 45.42 ms using TensorFlow Lite/XNNPACK with eight threads and batch size one.

These are controlled-background, laptop-specific results. They do not establish field generalization, energy efficiency, smartphone performance, or microcontroller performance.

## Repository layout

```text
analysis/
  scripts/               Re-evaluation, audit, FLOP, statistics, and Grad-CAM code
  outputs/               Derived metrics, predictions, audits, and visual evidence
data/
  README.md               Dataset acquisition and split instructions
docs/
  VALIDATION_SUMMARY.md   Concise validation summary
manuscript/               Revised LaTeX source, figures, references, and MDPI template files
models/
  h5/                     Saved floating-point models (Git LFS)
  tflite/                 Dynamic-range-quantized models (Git LFS)
notebooks/
  data/                   Dataset split notebook
  training/               CONE, PRISM, RHOMBUS, and CUBE training notebooks
  deployment/             Conversion, comparison, and device-test notebooks
tools/                    Repository preparation and integrity utilities
```

## Model and notebook mapping

| Architecture | Training notebook | H5 model | TFLite model |
|---|---|---|---|
| InceptionV3 | `CONE.ipynb` | `best_cone_model.h5` | `best_cone_model.tflite` |
| Xception | `PRISM.ipynb` | `best_prism_model.h5` | `best_prism_model.tflite` |
| MobileNetV2 | `RHOMBUS.ipynb` | `best_rhombus_model.h5` | `best_rhombus_model.tflite` |
| MobileNetV2 + CBAM | `CUBE.ipynb` | `best_cube_model.h5` | `best_cube_model.tflite` |

## Clone and install

The model files are stored with Git LFS.

```bash
git lfs install
git clone https://github.com/mopoloultra/attention-enhanced-mobilenetv2-plant-disease-detection.git
cd attention-enhanced-mobilenetv2-plant-disease-detection
git lfs pull
```

For saved-model evaluation and deployment analysis, use Python 3.10:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-validation.txt
```

The archived training environment is recorded separately in `requirements-training.txt`.

## Dataset preparation

The PlantVillage image corpus is not redistributed in this repository because of its size. Obtain the dataset from the [PlantVillage dataset repository](https://github.com/spMohanty/PlantVillage-Dataset), then follow [data/README.md](data/README.md).

The analysis scripts use `data/PlantVillage` by default. A different location can be supplied with the `PLANTVILLAGE_DIR` environment variable.

Expected structure:

```text
data/PlantVillage/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

## Reproduce the derived analyses

From the repository root:

```bash
python analysis/scripts/inspect_models.py
python analysis/scripts/recompute_tflite_metrics.py
python analysis/scripts/paired_significance.py
```

The full audit and validation entry point is:

```bash
python analysis/scripts/validate_experiments.py --help
```

The stored outputs in `analysis/outputs` allow the reported metrics and paired significance test to be inspected without retraining. Re-running model inference requires the PlantVillage split.

## Reproducibility notes

- Training notebooks initially froze the backbones and then unfroze the final 30 layers.
- Accuracy, precision, recall, and F1 use the definitions documented in the manuscript; precision, recall, and F1 are support-weighted, while AUC is macro one-vs-rest.
- TensorFlow Lite conversion used `Optimize.DEFAULT` without a representative calibration dataset, producing dynamic-range weight quantization with float32 inputs and outputs.
- A SHA-256 audit found 10 exact cross-split duplicate groups, seven involving test images. Excluding those seven test images changed every reported TFLite accuracy by less than 0.01 percentage points.
- Only one archived training run exists for each architecture. The paired confidence interval and McNemar test do not replace independent-seed reruns.

## Hardware used in the study

- Windows 11 Pro
- Intel Core i7-11800H, 8 cores / 16 logical processors
- 16 GB DDR4 RAM
- NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM

## Integrity

SHA-256 hashes for every distributed model are provided in `models/SHA256SUMS`.

```bash
python tools/verify_checksums.py
```

## Citation

[![DOI](https://zenodo.org/badge/1275518776.svg)](https://doi.org/10.5281/zenodo.20778518)

## License

No software license has yet been specified. Copyright remains with the authors.
