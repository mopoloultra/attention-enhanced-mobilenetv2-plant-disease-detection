import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tf_keras
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from paths import DATASET, H5_ROOT, OUTPUT_DIR

MODELS = {
    "MobileNetV2": "best_rhombus_model.h5",
    "MobileNetV2+CBAM": "best_cube_model.h5",
    "Xception": "best_prism_model.h5",
}
generator = tf_keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    DATASET / "test", target_size=(224, 224), batch_size=32,
    class_mode="categorical", shuffle=False,
)
y_true = generator.classes
binary = label_binarize(y_true, classes=np.arange(38))
report = {}
for name, filename in MODELS.items():
    print(f"Evaluating {name}", flush=True)
    model = tf_keras.models.load_model(H5_ROOT / filename, compile=False)
    probabilities = model.predict(generator, verbose=1)
    y_pred = probabilities.argmax(axis=1)
    report[name] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_macro_ovr": roc_auc_score(binary, probabilities, average="macro", multi_class="ovr"),
    }
    (OUTPUT_DIR / "remaining_h5_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
