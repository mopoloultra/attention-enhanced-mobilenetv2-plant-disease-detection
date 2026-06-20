import ast
import json
import math
import struct
import zipfile
from pathlib import Path
from paths import DATASET, OUTPUT_DIR

OUT = OUTPUT_DIR


def read_ints(npz_path, member="y_pred.npy"):
    with zipfile.ZipFile(npz_path) as archive:
        data = archive.read(member)
    major = data[6]
    if major == 1:
        length = struct.unpack("<H", data[8:10])[0]
        offset = 10
    else:
        length = struct.unpack("<I", data[8:12])[0]
        offset = 12
    header = ast.literal_eval(data[offset:offset + length].decode("latin1").strip())
    payload = data[offset + length:]
    return list(struct.unpack("<" + "q" * (len(payload) // 8), payload))


classes = sorted(path.name for path in (DATASET / "test").iterdir() if path.is_dir())
class_index = {name: index for index, name in enumerate(classes)}
y_true = []
for class_dir in sorted((DATASET / "test").iterdir(), key=lambda p: p.name.lower()):
    if class_dir.is_dir():
        y_true.extend([class_index[class_dir.name]] * len([path for path in class_dir.iterdir() if path.is_file()]))

baseline = read_ints(OUT / "mobilenetv2_tflite_predictions.npz")
cbam = read_ints(OUT / "mobilenetv2_cbam_tflite_predictions.npz")
baseline_correct = [pred == true for pred, true in zip(baseline, y_true)]
cbam_correct = [pred == true for pred, true in zip(cbam, y_true)]
b = sum(a and not c for a, c in zip(baseline_correct, cbam_correct))
c = sum((not a) and c for a, c in zip(baseline_correct, cbam_correct))
n = b + c
tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
p_exact = min(1.0, 2 * tail)
differences = [int(c) - int(a) for a, c in zip(baseline_correct, cbam_correct)]
mean = sum(differences) / len(differences)
variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
standard_error = math.sqrt(variance / len(differences))
report = {
    "test_images": len(y_true),
    "baseline_accuracy": sum(baseline_correct) / len(y_true),
    "cbam_accuracy": sum(cbam_correct) / len(y_true),
    "paired_difference_percentage_points": 100 * mean,
    "paired_difference_95_ci_percentage_points": [100 * (mean - 1.96 * standard_error), 100 * (mean + 1.96 * standard_error)],
    "baseline_only_correct": b,
    "cbam_only_correct": c,
    "mcnemar_exact_two_sided_p": p_exact,
}
(OUT / "paired_significance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
