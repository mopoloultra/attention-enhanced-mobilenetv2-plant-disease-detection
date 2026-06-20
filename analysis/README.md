# Analysis assets

`scripts/` contains the dataset audit, H5/TFLite evaluation, FLOP profiling, paired significance, and Grad-CAM workflows used during manuscript revision.

`outputs/` contains the derived metrics, prediction arrays, duplicate audit, statistical result, FLOP profiles, and matched Grad-CAM panel used in the revised manuscript.

Scripts resolve resources relative to the repository root:

- Dataset: `data/PlantVillage` or `PLANTVILLAGE_DIR`
- H5 models: `models/h5` or `H5_MODEL_DIR`
- TFLite models: `models/tflite` or `TFLITE_MODEL_DIR`
- Outputs: `analysis/outputs`

