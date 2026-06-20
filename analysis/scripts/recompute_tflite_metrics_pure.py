import ast
import json
import math
import struct
import zipfile
from pathlib import Path
from paths import DATASET, OUTPUT_DIR

OUT = OUTPUT_DIR
NAMES = {
    "InceptionV3": "inceptionv3_tflite_predictions.npz",
    "MobileNetV2": "mobilenetv2_tflite_predictions.npz",
    "MobileNetV2+CBAM": "mobilenetv2_cbam_tflite_predictions.npz",
    "Xception": "xception_tflite_predictions.npz",
}


def read_npy_from_npz(path, member):
    with zipfile.ZipFile(path) as archive:
        data = archive.read(member)
    if data[:6] != b"\x93NUMPY":
        raise ValueError("Not an NPY payload")
    major = data[6]
    if major == 1:
        header_length = struct.unpack("<H", data[8:10])[0]
        offset = 10
    else:
        header_length = struct.unpack("<I", data[8:12])[0]
        offset = 12
    header = ast.literal_eval(data[offset:offset + header_length].decode("latin1").strip())
    payload = data[offset + header_length:]
    shape = header["shape"]
    descr = header["descr"]
    if descr in ("<i8", "|i8"):
        values = list(struct.unpack("<" + "q" * (len(payload) // 8), payload))
    elif descr in ("<f4", "|f4"):
        values = list(struct.unpack("<" + "f" * (len(payload) // 4), payload))
    else:
        raise ValueError(f"Unsupported dtype {descr}")
    return shape, values


def test_rows():
    keras_classes = sorted(path.name for path in (DATASET / "test").iterdir() if path.is_dir())
    class_index = {name: index for index, name in enumerate(keras_classes)}
    paths = []
    labels = []
    for class_dir in sorted((DATASET / "test").iterdir(), key=lambda p: p.name.lower()):
        if class_dir.is_dir():
            for path in sorted(class_dir.iterdir(), key=lambda p: p.name.lower()):
                if path.is_file():
                    paths.append(str(path))
                    labels.append(class_index[class_dir.name])
    return keras_classes, paths, labels


def auc_binary(y_true_binary, scores):
    ordered = sorted(zip(scores, y_true_binary), key=lambda item: item[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum_positive += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    positives = sum(y_true_binary)
    negatives = len(y_true_binary) - positives
    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def metric_block(y_true, y_pred, probabilities, classes_count):
    confusion = [[0] * classes_count for _ in range(classes_count)]
    for true, pred in zip(y_true, y_pred):
        confusion[true][pred] += 1
    supports = [sum(row) for row in confusion]
    precisions, recalls, f1s = [], [], []
    for class_id in range(classes_count):
        tp = confusion[class_id][class_id]
        fp = sum(confusion[row][class_id] for row in range(classes_count)) - tp
        fn = sum(confusion[class_id]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    total = len(y_true)
    aucs = []
    for class_id in range(classes_count):
        binary = [1 if true == class_id else 0 for true in y_true]
        scores = [row[class_id] for row in probabilities]
        aucs.append(auc_binary(binary, scores))
    return {
        "accuracy": sum(confusion[i][i] for i in range(classes_count)) / total,
        "precision_weighted": sum(value * weight for value, weight in zip(precisions, supports)) / total,
        "recall_weighted": sum(value * weight for value, weight in zip(recalls, supports)) / total,
        "f1_weighted": sum(value * weight for value, weight in zip(f1s, supports)) / total,
        "precision_macro": sum(precisions) / classes_count,
        "recall_macro": sum(recalls) / classes_count,
        "f1_macro": sum(f1s) / classes_count,
        "auc_macro_ovr": sum(aucs) / classes_count,
        "images": total,
    }


classes, paths, y_true = test_rows()
audit = json.loads((OUT / "dataset_audit.json").read_text(encoding="utf-8"))
excluded = {
    row["path"]
    for rows in audit["cross_split_duplicates"].values()
    for row in rows
    if row["split"] == "test"
}
old = json.loads((OUT / "tflite_validation.json").read_text(encoding="utf-8"))
report = {"class_order": classes, "quantization": old["quantization"], "runtime": old["runtime"], "models": {}}

for name, filename in NAMES.items():
    shape_pred, y_pred = read_npy_from_npz(OUT / filename, "y_pred.npy")
    shape_prob, flat_prob = read_npy_from_npz(OUT / filename, "probabilities.npy")
    rows, columns = shape_prob
    probabilities = [flat_prob[index * columns:(index + 1) * columns] for index in range(rows)]
    keep = [index for index, path in enumerate(paths) if path not in excluded]
    block = metric_block(y_true, y_pred, probabilities, len(classes))
    block["duplicate_excluded"] = {
        **metric_block([y_true[i] for i in keep], [y_pred[i] for i in keep], [probabilities[i] for i in keep], len(classes)),
        "excluded_test_images": len(paths) - len(keep),
    }
    block["cpu_latency"] = old["models"][name]["cpu_latency"]
    block["file_bytes"] = old["models"][name]["file_bytes"]
    report["models"][name] = block

(OUT / "tflite_corrected_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
