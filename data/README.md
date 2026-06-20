# PlantVillage dataset

The image corpus is not committed to GitHub. Download PlantVillage from the [dataset repository](https://github.com/spMohanty/PlantVillage-Dataset).

## Recreate the study split

1. Place the unsplit class directories under `data/PlantVillage_raw/`.
2. Open `notebooks/data/Train_test_split.ipynb` from the repository root.
3. Run the notebook to create `data/PlantVillage/{train,val,test}`.

The archived split procedure partitions each class independently with `random_state=42`: 70% training and 30% temporary data, followed by an equal 15%/15% validation/test division.

Resulting directory counts in the archived experiment:

- Train: 37,998 images (69.97%)
- Validation: 8,146 images (15.00%)
- Test: 8,162 images (15.03%)

The archived training generators also used `validation_split=0.2`, yielding effective generator counts of 30,416 training and 1,614 validation images. This legacy behavior is retained in the notebooks and documented in the manuscript.

To keep the dataset elsewhere, set `PLANTVILLAGE_DIR` to the split directory before running the scripts in `analysis/scripts`.

