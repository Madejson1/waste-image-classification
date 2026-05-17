# waste-image-classification
Waste image classification pipeline using PyTorch, OpenCV and transfer learning, with Power BI dashboard for model evaluation and error analysis.

Computer vision pipeline for classifying waste images into four categories:

- Hazardous
- Non-Recyclable
- Organic
- Recyclable

The project uses image validation, metadata generation, stratified train/validation/test split, transfer learning and model evaluation.

## Tech stack

- Python
- PyTorch
- torchvision
- OpenCV
- scikit-learn
- pandas
- NumPy
- matplotlib
- seaborn

## Dataset

Download the dataset from Kaggle (https://www.kaggle.com/datasets/phenomsg/waste-classification) and place it inside:

```text
data/raw/
```

Recommended structure:

```text
data/raw/waste-classification/
├── Hazardous/
├── Non-Recyclable/
├── Organic/
└── Recyclable/
```

The scanner is recursive and also supports nested folders, as long as one of the parent folders contains one of the expected class names.

Expected classes:

```text
hazardous
non-recyclable
organic
recyclable
```

## Setup

Windows CMD / PowerShell:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick run

For a fast local test:

```bash
python -m src.train --data-root data/raw --max-images-per-class 300 --epochs 3 --batch-size 32
```

For a larger run:

```bash
python -m src.train --data-root data/raw --epochs 8 --batch-size 32
```

## Outputs

```text
outputs/metadata/metadata.csv
outputs/splits/train.csv
outputs/splits/val.csv
outputs/splits/test.csv
outputs/metrics/model_metrics.csv
outputs/predictions/test_predictions.csv
outputs/figures/confusion_matrix.png
outputs/figures/training_curves.png
outputs/model/best_model.pth
outputs/model/class_to_idx.json
```

## Power BI Dashboard

This project includes a Power BI dashboard for model performance analysis.

The dashboard visualizes:
- confusion matrix
- model accuracy
- prediction confidence distribution
- error analysis

### How to use

1. Run the training pipeline:

```bash
python -m src.train --data-root data/raw --epochs 8 --batch-size 32

run power_bi_dashboard.pbit and specify the project folder location, e.g. C:\path\to\waste-image-classification\

the dashboard loads data from outputs/predictions/test_predictions.csv

## Notes

The model uses MobileNetV2 transfer learning. By default, the feature extractor is frozen and only the classifier head is trained, which makes training faster on CPU.
