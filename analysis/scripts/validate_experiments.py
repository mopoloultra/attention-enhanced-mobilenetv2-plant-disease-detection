import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
import tf_keras
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from paths import DATASET, H5_ROOT, OUTPUT_DIR, TFLITE_ROOT

MODEL_ROOT = H5_ROOT
OUT = OUTPUT_DIR

MODELS = {
    "InceptionV3": {"h5": "best_cone_model.h5", "size": 299, "last_conv": "mixed10"},
    "MobileNetV2": {"h5": "best_rhombus_model.h5", "size": 224, "last_conv": "out_relu"},
    "MobileNetV2+CBAM": {"h5": "best_cube_model.h5", "size": 224, "last_conv": "multiply_5"},
    "Xception": {"h5": "best_prism_model.h5", "size": 224, "last_conv": "block14_sepconv2_act"},
}


def json_dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def image_files(split):
    rows = []
    for class_dir in sorted((DATASET / split).iterdir(), key=lambda p: p.name.lower()):
        if class_dir.is_dir():
            for path in sorted(class_dir.iterdir(), key=lambda p: p.name.lower()):
                if path.is_file():
                    rows.append((path, class_dir.name))
    return rows


def audit_dataset():
    split_rows = {split: image_files(split) for split in ("train", "val", "test")}
    classes = sorted({label for rows in split_rows.values() for _, label in rows})
    counts = {
        split: {
            "total": len(rows),
            "per_class": dict(Counter(label for _, label in rows)),
        }
        for split, rows in split_rows.items()
    }
    total = sum(v["total"] for v in counts.values())
    for split in counts:
        counts[split]["percentage"] = 100.0 * counts[split]["total"] / total

    named = defaultdict(list)
    for split, rows in split_rows.items():
        for path, label in rows:
            named[(label, path.name)].append(split)
    filename_overlap = {f"{k[0]}/{k[1]}": v for k, v in named.items() if len(set(v)) > 1}

    all_rows = [(split, path, label) for split, rows in split_rows.items() for path, label in rows]
    def hash_row(row):
        split, path, label = row
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return split, str(path), label, digest
    with ThreadPoolExecutor(max_workers=12) as pool:
        hashed = list(pool.map(hash_row, all_rows))
    by_hash = defaultdict(list)
    for split, path, label, digest in hashed:
        by_hash[digest].append({"split": split, "path": path, "class": label})
    cross_split_duplicates = {
        digest: rows for digest, rows in by_hash.items()
        if len({row["split"] for row in rows}) > 1
    }
    report = {
        "classes": len(classes),
        "class_order": classes,
        "total_images": total,
        "splits": counts,
        "filename_overlap_count": len(filename_overlap),
        "cross_split_sha256_duplicate_groups": len(cross_split_duplicates),
        "filename_overlap": filename_overlap,
        "cross_split_duplicates": cross_split_duplicates,
        "split_method": "Per-class train_test_split: 70% train; remaining 30% divided equally into validation and test; random_state=42 for both operations.",
    }
    json_dump(OUT / "dataset_audit.json", report)
    return report


def load_image(path, size):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.NEAREST)
        return np.asarray(image, dtype=np.float32) / 255.0


def class_and_test_rows():
    # ImageDataGenerator.flow_from_directory uses Python's default case-sensitive
    # lexical sort when assigning class indices; preserve that exact ordering.
    classes = sorted([p.name for p in (DATASET / "test").iterdir() if p.is_dir()])
    class_index = {name: index for index, name in enumerate(classes)}
    rows = image_files("test")
    return classes, [(path, class_index[label], label) for path, label in rows]


def evaluate_tflite_model(name, config, classes, rows):
    path = TFLITE_ROOT / config["h5"].replace(".h5", ".tflite")
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=8)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    size = config["size"]
    batch_size = 32
    cache_path = OUT / f"{name.lower().replace('+', '_').replace(' ', '_')}_tflite_predictions.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        probabilities = cached["probabilities"]
    else:
        predictions = []
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            batch = np.stack([load_image(path, size) for path, _, _ in batch_rows])
            interpreter.resize_tensor_input(input_detail["index"], batch.shape, strict=False)
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_detail["index"], batch)
            interpreter.invoke()
            predictions.append(interpreter.get_tensor(output_detail["index"]).copy())
        probabilities = np.concatenate(predictions, axis=0)
    y_true = np.asarray([index for _, index, _ in rows], dtype=np.int64)
    y_pred = probabilities.argmax(axis=1)
    y_bin = label_binarize(y_true, classes=np.arange(len(classes)))
    def metric_block(true_values, predicted_values, probability_values):
        binary = label_binarize(true_values, classes=np.arange(len(classes)))
        return {
            "accuracy": accuracy_score(true_values, predicted_values),
            "precision_weighted": precision_score(true_values, predicted_values, average="weighted", zero_division=0),
            "recall_weighted": recall_score(true_values, predicted_values, average="weighted", zero_division=0),
            "f1_weighted": f1_score(true_values, predicted_values, average="weighted", zero_division=0),
            "precision_macro": precision_score(true_values, predicted_values, average="macro", zero_division=0),
            "recall_macro": recall_score(true_values, predicted_values, average="macro", zero_division=0),
            "f1_macro": f1_score(true_values, predicted_values, average="macro", zero_division=0),
            "auc_macro_ovr": roc_auc_score(binary, probability_values, average="macro", multi_class="ovr"),
            "images": len(true_values),
        }
    metrics = {
        **metric_block(y_true, y_pred, probabilities),
        "test_images": len(rows),
        "input_size": size,
        "input_dtype": str(input_detail["dtype"]),
        "output_dtype": str(output_detail["dtype"]),
        "file_bytes": path.stat().st_size,
    }
    audit_path = OUT / "dataset_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        excluded = {
            row["path"]
            for duplicate_rows in audit["cross_split_duplicates"].values()
            for row in duplicate_rows
            if row["split"] == "test"
        }
        clean_mask = np.asarray([str(path) not in excluded for path, _, _ in rows])
        metrics["duplicate_excluded"] = {
            **metric_block(y_true[clean_mask], y_pred[clean_mask], probabilities[clean_mask]),
            "excluded_test_images": int((~clean_mask).sum()),
        }

    # Reproducible single-image CPU latency; decoding/resizing is excluded.
    rng = random.Random(42)
    sampled = [rows[i] for i in rng.sample(range(len(rows)), 250)]
    tensors = [load_image(path, size)[None, ...] for path, _, _ in sampled]
    interpreter.resize_tensor_input(input_detail["index"], [1, size, size, 3], strict=False)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    for tensor in tensors[:50]:
        interpreter.set_tensor(input_detail["index"], tensor)
        interpreter.invoke()
        interpreter.get_tensor(output_detail["index"])
    elapsed_ms = []
    for tensor in tensors[50:]:
        interpreter.set_tensor(input_detail["index"], tensor)
        start = time.perf_counter_ns()
        interpreter.invoke()
        interpreter.get_tensor(output_detail["index"])
        elapsed_ms.append((time.perf_counter_ns() - start) / 1_000_000)
    q1, q3 = np.percentile(elapsed_ms, [25, 75])
    metrics["cpu_latency"] = {
        "threads": 8,
        "warmup_runs": 50,
        "timed_runs": 200,
        "batch_size": 1,
        "preprocessing_included": False,
        "mean_ms": statistics.fmean(elapsed_ms),
        "median_ms": statistics.median(elapsed_ms),
        "std_ms": statistics.pstdev(elapsed_ms),
        "iqr_ms": q3 - q1,
        "p95_ms": float(np.percentile(elapsed_ms, 95)),
    }
    np.savez_compressed(
        cache_path,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        paths=np.asarray([str(path) for path, _, _ in rows]),
    )
    return metrics


def evaluate_tflite():
    classes, rows = class_and_test_rows()
    report = {
        "runtime": {
            "tensorflow": tf.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "devices": [str(x) for x in tf.config.list_physical_devices()],
        },
        "quantization": "TensorFlow Lite Optimize.DEFAULT without a representative dataset: dynamic-range weight quantization with float32 model inputs and outputs.",
        "models": {},
    }
    for name, config in MODELS.items():
        print(f"Evaluating {name}", flush=True)
        report["models"][name] = evaluate_tflite_model(name, config, classes, rows)
        json_dump(OUT / "tflite_validation.json", report)
    with (OUT / "tflite_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Model", "Accuracy", "Precision_weighted", "Recall_weighted", "F1_weighted", "Precision_macro", "Recall_macro", "F1_macro", "AUC_macro_OVR", "Median_latency_ms", "P95_latency_ms"])
        for name, m in report["models"].items():
            writer.writerow([name, m["accuracy"], m["precision_weighted"], m["recall_weighted"], m["f1_weighted"], m["precision_macro"], m["recall_macro"], m["f1_macro"], m["auc_macro_ovr"], m["cpu_latency"]["median_ms"], m["cpu_latency"]["p95_ms"]])
    return report


def load_legacy_model(path):
    return tf_keras.models.load_model(path, compile=False)


def model_flops(model, size):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    fn = tf.function(lambda x: model(x, training=False))
    concrete = fn.get_concrete_function(tf.TensorSpec([1, size, size, 3], tf.float32))
    frozen, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        options["output"] = "none"
        profile = tf.compat.v1.profiler.profile(graph=graph, options=options)
    return int(profile.total_float_ops)


def inspect_h5_and_flops():
    report = {}
    for name, config in MODELS.items():
        print(f"Loading {name}", flush=True)
        model = load_legacy_model(MODEL_ROOT / config["h5"])
        flops = model_flops(model, config["size"])
        report[name] = {
            "input_shape": model.input_shape,
            "parameters": model.count_params(),
            "trainable_parameters": int(sum(tf.size(w).numpy() for w in model.trainable_weights)),
            "nontrainable_parameters": int(sum(tf.size(w).numpy() for w in model.non_trainable_weights)),
            "flops_batch1": flops,
            "gflops_batch1": flops / 1e9,
            "approx_gmacs": flops / 2e9,
            "last_conv_layer": config["last_conv"],
        }
    json_dump(OUT / "h5_flops.json", report)
    return report


def gradcam_heatmap(model, image_array, layer_name, class_index):
    grad_model = tf_keras.models.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    tensor = tf.convert_to_tensor(image_array[None, ...])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(tensor, training=False)
        target = predictions[:, class_index]
    gradients = tape.gradient(target, conv_output)
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_output * weights[:, None, None, :], axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    if maximum > 0:
        heatmap = heatmap / maximum
    return heatmap.numpy()


def overlay(image_array, heatmap):
    base = np.uint8(np.clip(image_array * 255, 0, 255))
    heatmap = cv2.resize(heatmap, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 0.55, colored, 0.45, 0)


def generate_gradcam_panel():
    classes, rows = class_and_test_rows()
    base_npz = np.load(OUT / "mobilenetv2_tflite_predictions.npz")
    cbam_npz = np.load(OUT / "mobilenetv2_cbam_tflite_predictions.npz")
    y_true = base_npz["y_true"]
    base_pred = base_npz["y_pred"]
    cbam_pred = cbam_npz["y_pred"]
    candidates = {
        "CBAM corrects baseline": np.where((base_pred != y_true) & (cbam_pred == y_true))[0],
        "Both correct": np.where((base_pred == y_true) & (cbam_pred == y_true))[0],
        "Both incorrect": np.where((base_pred != y_true) & (cbam_pred != y_true))[0],
    }
    selected = []
    for label, indices in candidates.items():
        if len(indices):
            selected.append((label, int(indices[0])))
    selected = selected[:3]

    baseline = load_legacy_model(MODEL_ROOT / MODELS["MobileNetV2"]["h5"])
    cbam = load_legacy_model(MODEL_ROOT / MODELS["MobileNetV2+CBAM"]["h5"])
    tile_w, tile_h = 420, 380
    panel = Image.new("RGB", (tile_w * 3, tile_h * len(selected)), "white")
    draw = ImageDraw.Draw(panel)
    metadata = []
    for row_number, (case, index) in enumerate(selected):
        path, true_index, true_label = rows[index]
        array = load_image(path, 224)
        h_base = gradcam_heatmap(baseline, array, MODELS["MobileNetV2"]["last_conv"], int(base_pred[index]))
        h_cbam = gradcam_heatmap(cbam, array, MODELS["MobileNetV2+CBAM"]["last_conv"], int(cbam_pred[index]))
        images = [np.uint8(array * 255), overlay(array, h_base), overlay(array, h_cbam)]
        headings = ["Original", f"MobileNetV2: {classes[int(base_pred[index])]}", f"MobileNetV2+CBAM: {classes[int(cbam_pred[index])]}"]
        for column, (pixels, heading) in enumerate(zip(images, headings)):
            image = Image.fromarray(pixels).resize((300, 300), Image.Resampling.LANCZOS)
            x = column * tile_w + (tile_w - 300) // 2
            y = row_number * tile_h + 35
            panel.paste(image, (x, y))
            draw.text((column * tile_w + 10, row_number * tile_h + 8), heading.replace("___", ": ").replace("_", " ")[:58], fill="black")
        draw.text((10, row_number * tile_h + 342), f"{case}; true: {true_label.replace('___', ': ').replace('_', ' ')}", fill="black")
        metadata.append({"case": case, "path": str(path), "true": true_label, "baseline": classes[int(base_pred[index])], "cbam": classes[int(cbam_pred[index])]})
    panel.save(OUT / "gradcam_baseline_vs_cbam.png")
    json_dump(OUT / "gradcam_samples.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["dataset", "tflite", "flops", "gradcam", "all"])
    args = parser.parse_args()
    if args.stage in ("dataset", "all"):
        print(json.dumps(audit_dataset(), indent=2)[:4000])
    if args.stage in ("tflite", "all"):
        print(json.dumps(evaluate_tflite(), indent=2))
    if args.stage in ("flops", "all"):
        print(json.dumps(inspect_h5_and_flops(), indent=2))
    if args.stage in ("gradcam", "all"):
        print(json.dumps(generate_gradcam_panel(), indent=2))


if __name__ == "__main__":
    main()
