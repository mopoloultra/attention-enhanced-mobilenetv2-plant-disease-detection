import json
from pathlib import Path

import tensorflow as tf
from paths import H5_ROOT, REPO_ROOT, TFLITE_ROOT

MODELS = {
    "InceptionV3": "best_cone_model.h5",
    "MobileNetV2": "best_rhombus_model.h5",
    "MobileNetV2+CBAM": "best_cube_model.h5",
    "Xception": "best_prism_model.h5",
}

report = {"tensorflow": tf.__version__, "devices": [str(x) for x in tf.config.list_physical_devices()], "models": {}}
for name, filename in MODELS.items():
    path = H5_ROOT / filename
    entry = {"path": str(path), "bytes": path.stat().st_size}
    try:
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        entry.update(
            loaded=True,
            input_shape=model.input_shape,
            output_shape=model.output_shape,
            parameters=model.count_params(),
            trainable_parameters=sum(int(tf.size(w)) for w in model.trainable_weights),
            nontrainable_parameters=sum(int(tf.size(w)) for w in model.non_trainable_weights),
            layers=len(model.layers),
            layer_names=[layer.name for layer in model.layers[-15:]],
        )
    except Exception as exc:
        entry.update(loaded=False, error=repr(exc))
    report["models"][name] = entry

for name, filename in MODELS.items():
    tflite_path = TFLITE_ROOT / filename.replace(".h5", ".tflite")
    entry = report["models"][name]
    try:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        def tensor_detail(detail):
            return {
                "name": detail["name"],
                "index": int(detail["index"]),
                "shape": detail["shape"].tolist(),
                "shape_signature": detail["shape_signature"].tolist(),
                "dtype": str(detail["dtype"]),
                "quantization": list(detail["quantization"]),
            }
        entry["tflite"] = {
            "bytes": tflite_path.stat().st_size,
            "inputs": [tensor_detail(x) for x in interpreter.get_input_details()],
            "outputs": [tensor_detail(x) for x in interpreter.get_output_details()],
        }
    except Exception as exc:
        entry["tflite_error"] = repr(exc)

def simplify(obj):
    if hasattr(obj, "tolist") and not isinstance(obj, type):
        return obj.tolist()
    if isinstance(obj, type):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(type(obj).__name__)

(REPO_ROOT / "analysis" / "model_inspection.json").write_text(json.dumps(report, indent=2, default=simplify), encoding="utf-8")
print(json.dumps(report, indent=2, default=simplify))
