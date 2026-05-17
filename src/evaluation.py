import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm


def predict(model, dataloader, device, idx_to_class: dict, dataframe: pd.DataFrame) -> pd.DataFrame:
    model.eval()

    rows = []
    offset = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, leave=False):
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidence, preds = torch.max(probabilities, dim=1)

            batch_size = images.size(0)
            batch_df = dataframe.iloc[offset: offset + batch_size].reset_index(drop=True)
            offset += batch_size

            for i in range(batch_size):
                actual_idx = int(targets[i].item())
                predicted_idx = int(preds[i].cpu().item())

                rows.append(
                    {
                        "image_path": batch_df.loc[i, "image_path"],
                        "file_name": batch_df.loc[i, "file_name"],
                        "actual_label": idx_to_class[actual_idx],
                        "predicted_label": idx_to_class[predicted_idx],
                        "confidence": float(confidence[i].cpu().item()),
                        "is_correct": actual_idx == predicted_idx,
                    }
                )

    predictions = pd.DataFrame(rows)
    predictions["error_type"] = "correct"
    predictions.loc[
        predictions["actual_label"] != predictions["predicted_label"],
        "error_type",
    ] = "misclassified"

    return predictions


def save_metrics(predictions: pd.DataFrame, output_path):
    y_true = predictions["actual_label"]
    y_pred = predictions["predicted_label"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    pd.DataFrame([metrics]).to_csv(output_path, index=False)
    return metrics


def save_classification_report(predictions: pd.DataFrame, output_path):
    report = classification_report(
        predictions["actual_label"],
        predictions["predicted_label"],
        output_dict=True,
        zero_division=0,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def save_confusion_matrix(predictions: pd.DataFrame, labels: list[str], output_path):
    cm = confusion_matrix(
        predictions["actual_label"],
        predictions["predicted_label"],
        labels=labels,
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_training_curves(history: dict, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.plot(epochs, history["train_macro_f1"], label="train_macro_f1")
    plt.plot(epochs, history["val_macro_f1"], label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()