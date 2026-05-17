from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import IMAGE_EXTENSIONS, CLASS_ORDER, RANDOM_STATE


def normalize_part(value: str) -> str:
    return value.lower().strip().replace("_", " ").replace("-", " ")


def infer_label_from_path(image_path: Path, data_root: Path) -> Optional[str]:
    relative_parts = image_path.relative_to(data_root).parts[:-1]

    for part in relative_parts:
        normalized = normalize_part(part)

        if normalized == "hazardous":
            return "hazardous"

        if normalized == "non recyclable":
            return "non-recyclable"

        if normalized == "organic":
            return "organic"

        if normalized == "recyclable":
            return "recyclable"

    return None


def validate_image(image_path: Path) -> dict:
    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "is_valid": False,
            "width": None,
            "height": None,
            "channels": None,
        }

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    return {
        "is_valid": True,
        "width": width,
        "height": height,
        "channels": channels,
    }


def build_metadata(data_root: str, max_images_per_class: Optional[int] = None) -> pd.DataFrame:
    data_root_path = Path(data_root)

    if not data_root_path.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    rows = []

    for image_path in data_root_path.rglob("*"):
        if image_path.is_dir():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label = infer_label_from_path(image_path, data_root_path)

        if label is None:
            continue

        image_info = validate_image(image_path)

        rows.append(
            {
                "image_path": str(image_path),
                "label": label,
                "file_name": image_path.name,
                "file_size_kb": round(image_path.stat().st_size / 1024, 2),
                **image_info,
            }
        )

    metadata = pd.DataFrame(
        rows,
        columns=[
            "image_path",
            "label",
            "file_name",
            "file_size_kb",
            "is_valid",
            "width",
            "height",
            "channels",
        ],
    )

    if metadata.empty:
        discovered_dirs = sorted(
            {
                str(path.relative_to(data_root_path))
                for path in data_root_path.rglob("*")
                if path.is_dir()
            }
        )[:50]

        raise ValueError(
            "No labeled images found. Check folder names and --data-root.\n"
            "Expected top-level folders: Hazardous, Non-Recyclable, Organic, Recyclable.\n"
            f"First discovered folders: {discovered_dirs}"
        )

    metadata = metadata[metadata["is_valid"]].copy()

    if metadata.empty:
        raise ValueError("Images were found, but none could be read by OpenCV.")

    metadata = metadata[metadata["label"].isin(CLASS_ORDER)].copy()

    if metadata.empty:
        raise ValueError(f"Labels found, but none match CLASS_ORDER: {CLASS_ORDER}")

    if max_images_per_class is not None:
        sampled_parts = []

        for label, group in metadata.groupby("label"):
            n = min(len(group), max_images_per_class)
            sampled_parts.append(group.sample(n=n, random_state=RANDOM_STATE))

        metadata = pd.concat(sampled_parts, ignore_index=True)

    return metadata


def create_stratified_splits(
    metadata: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if round(train_size + val_size + test_size, 5) != 1:
        raise ValueError("train_size + val_size + test_size must equal 1.")

    min_class_count = metadata["label"].value_counts().min()
    if min_class_count < 3:
        raise ValueError("Each class needs at least 3 valid images for stratified train/val/test split.")

    train_df, temp_df = train_test_split(
        metadata,
        train_size=train_size,
        random_state=RANDOM_STATE,
        stratify=metadata["label"],
    )

    relative_test_size = test_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )