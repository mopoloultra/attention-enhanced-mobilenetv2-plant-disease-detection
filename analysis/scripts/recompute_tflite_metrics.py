import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from paths import DATASET, OUTPUT_DIR

OUT = OUTPUT_DIR
NAMES = {
    "InceptionV3": "inceptionv3_tflite_predictions.npz",
    "MobileNetV2": "mobilenetv2_tflite_predictions.npz",
    "MobileNetV2+CBAM": "mobilenetv2_cbam_tflite_predictions.npz",
    "Xception": "xception_tflite_predictions.npz",
}

classes = sorted(path.name for path in (DATASET / "test").iterdir() if path.is_dir())
class_index = {name: index for index, name in enumerate(classes)}
audit = json.loads((OUT / "dataset_audit.json").read_text(encoding="utf-8"))
excluded = {
    row["path"]
    for rows in audit["cross_split_duplicates"].values()
    for row in rows
    if row["split"] == "test"
}
old = json.loads((OUT / "tflite_validation.json").read_text(encoding="utf-8"))

def metrics(y_true, y_pred, probabilities):
    binary = label_binarize(y_true, classes=np.arange(len(classes)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_macro_ovr": roc_auc_score(binary, probabilities, average="macro", multi_class="ovr"),
        "images": len(y_true),
    }

report = {
    "class_order": classes,
    "quantization": old["quantization"],
    "runtime": old["runtime"],
    "models": {},
}
for name, filename in NAMES.items():
    data = np.load(OUT / filename)
    paths = [str(path) for path in data["paths"]]
    probabilities = data["probabilities"]
    y_true = np.asarray([class_index[Path(path).parent.name] for path in paths], dtype=np.int64)
    y_pred = probabilities.argmax(axis=1)
    clean = np.asarray([path not in excluded for path in paths])
    block = metrics(y_true, y_pred, probabilities)
    block["duplicate_excluded"] = {
        **metrics(y_true[clean], y_pred[clean], probabilities[clean]),
        "excluded_test_images": int((~clean).sum()),
    }
    block["cpu_latency"] = old["models"][name]["cpu_latency"]
    block["file_bytes"] = old["models"][name]["file_bytes"]
    report["models"][name] = block

(OUT / "tflite_corrected_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
