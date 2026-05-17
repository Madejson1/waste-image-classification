import argparse
import json
import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

from src.config import (
    CLASS_ORDER,
    FIGURES_DIR,
    METADATA_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    SPLITS_DIR,
)
from src.data_utils import build_metadata, create_stratified_splits
from src.dataset import WasteImageDataset
from src.evaluation import (
    predict,
    save_classification_report,
    save_confusion_matrix,
    save_metrics,
    save_training_curves,
)
from src.model import build_model
from src.training import train_model
from src.transforms import get_eval_transforms, get_train_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train waste image classification model.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    for directory in [METADATA_DIR, SPLITS_DIR, METRICS_DIR, PREDICTIONS_DIR, FIGURES_DIR, MODEL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    print("Scanning and validating images...")
    metadata = build_metadata(args.data_root, max_images_per_class=args.max_images_per_class)
    metadata.to_csv(METADATA_DIR / "metadata.csv", index=False)

    print("\nClass distribution:")
    print(metadata["label"].value_counts())

    train_df, val_df, test_df = create_stratified_splits(metadata)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    classes = [label for label in CLASS_ORDER if label in metadata["label"].unique()]
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    with open(MODEL_DIR / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    train_dataset = WasteImageDataset(
        train_df,
        class_to_idx=class_to_idx,
        transform=get_train_transforms(args.image_size),
    )
    val_dataset = WasteImageDataset(
        val_df,
        class_to_idx=class_to_idx,
        transform=get_eval_transforms(args.image_size),
    )
    test_dataset = WasteImageDataset(
        test_df,
        class_to_idx=class_to_idx,
        transform=get_eval_transforms(args.image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = build_model(
        num_classes=len(classes),
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.unfreeze_backbone,
    )
    model = model.to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(range(len(classes)))),
        y=train_df["label"].map(class_to_idx).values,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    model_path = MODEL_DIR / "best_model.pth"

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        model_path=model_path,
    )

    pd.DataFrame(history).to_csv(METRICS_DIR / "training_history.csv", index=False)
    save_training_curves(history, FIGURES_DIR / "training_curves.png")

    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(model_path, map_location=device))

    predictions = predict(
        model=model,
        dataloader=test_loader,
        device=device,
        idx_to_class=idx_to_class,
        dataframe=test_df,
    )

    predictions.to_csv(PREDICTIONS_DIR / "test_predictions.csv", index=False)

    metrics = save_metrics(predictions, METRICS_DIR / "model_metrics.csv")
    save_classification_report(predictions, METRICS_DIR / "classification_report.json")
    save_confusion_matrix(predictions, classes, FIGURES_DIR / "confusion_matrix.png")

    print("\nDone.")
    print(f"Test metrics: {metrics}")
    print(f"Predictions saved to: {PREDICTIONS_DIR / 'test_predictions.csv'}")
    print(f"Best model saved to: {model_path}")


if __name__ == "__main__":
    main()