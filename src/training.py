from collections import defaultdict

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


def run_epoch(model, dataloader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in tqdm(dataloader, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = sum(p == y for p, y in zip(all_preds, all_targets)) / len(all_targets)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs: int, model_path):
    history = defaultdict(list)
    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        for key, value in train_metrics.items():
            history[f"train_{key}"].append(value)

        for key, value in val_metrics.items():
            history[f"val_{key}"].append(value)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), model_path)

    return history