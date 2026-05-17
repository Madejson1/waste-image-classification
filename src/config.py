from pathlib import Path


RANDOM_STATE = 42

OUTPUT_DIR = Path("outputs")
METADATA_DIR = OUTPUT_DIR / "metadata"
SPLITS_DIR = OUTPUT_DIR / "splits"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "model"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_ALIASES = {
    "hazardous": "hazardous",
    "hazardous waste": "hazardous",
    "non-recyclable": "non-recyclable",
    "non recyclable": "non-recyclable",
    "non_recyclable": "non-recyclable",
    "nonrecyclable": "non-recyclable",
    "organic": "organic",
    "recyclable": "recyclable",
}

CLASS_ORDER = ["hazardous", "non-recyclable", "organic", "recyclable"]