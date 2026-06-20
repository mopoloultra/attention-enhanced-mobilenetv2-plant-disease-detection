# Independent asset-validation summary

- Dataset: 54,306 images, 38 classes; 37,998 train, 8,146 validation, 8,162 test (approximately 70/15/15).
- Effective legacy generators: 30,416 training and 1,614 validation images because `validation_split=0.2` was applied to already separated directories.
- SHA-256 audit: 10 exact cross-split duplicate groups; seven test images duplicated content in train/validation. Excluding them changed accuracy by less than 0.01 percentage points.
- Transfer learning: every notebook unfreezes the final 30 backbone layers; the models were not fully frozen.
- Quantization: `Optimize.DEFAULT` without representative calibration data—dynamic-range weight quantization with float32 inputs/outputs.
- Saved-model re-evaluation accuracy: InceptionV3 96.89%, MobileNetV2 96.78%, MobileNetV2+CBAM 97.17%, Xception 98.30%.
- TFLite accuracy: InceptionV3 96.80%, MobileNetV2 96.07%, MobileNetV2+CBAM 96.70%, Xception 98.22%.
- Complexity (GFLOPs): InceptionV3 11.451, MobileNetV2 0.602, MobileNetV2+CBAM 0.604, Xception 9.115.
- Paired TFLite comparison: CBAM +0.64 percentage points (95% CI 0.31–0.96); 117 CBAM-only versus 65 baseline-only correct; exact McNemar p=0.000142.
- CPU median/P95 latency (ms): InceptionV3 10.71/12.27, MobileNetV2 45.38/106.96, MobileNetV2+CBAM 45.42/102.54, Xception 200.15/304.13. Protocol: TFLite 2.18/XNNPACK, i7-11800H, eight threads, batch one, 50 warm-ups, 200 timed runs, preprocessing excluded.
- Grad-CAM: matched baseline/CBAM correct, corrected, and shared-error cases generated and added as Figure 5.

