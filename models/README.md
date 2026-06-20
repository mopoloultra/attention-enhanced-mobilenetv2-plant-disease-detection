# Saved models

The H5 and TensorFlow Lite files are managed through Git LFS. Run `git lfs pull` after cloning.

TensorFlow Lite models were converted with `tf.lite.Optimize.DEFAULT` and no representative calibration dataset. They therefore use dynamic-range weight quantization while retaining float32 inputs and outputs.

Use `SHA256SUMS` to verify downloaded files.

