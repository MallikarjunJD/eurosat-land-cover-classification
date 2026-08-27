"""
train.py — Training entrypoint for the EuroSAT dual-mode classifier.

Orchestrates data.py (pipelines) and model.py (architecture) to train
the RGB model, the multispectral model, or both, then freezes the
resulting normalization statistics to normalization_stats.json — the
same file gradio.md's app.py reads at inference time.

Usage:
    python train.py --modality both
    python train.py --modality rgb --epochs 40 --lr 5e-4
    python train.py --modality multispectral \
        --ms-train-root ./data_ms/train --ms-val-root ./data_ms/val --ms-test-root ./data_ms/test
"""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

from data import (
    NUM_CLASSES,
    build_rgb_dataloaders,
    build_multispectral_dataloaders,
    save_normalization_stats,
)
from model import SEResEuroNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model: nn.Module, train_loader, val_loader, model_name: str,
                 num_epochs: int = 60, lr: float = 1e-3, patience: int = 8):
    """
    Trains `model` with AdamW + ReduceLROnPlateau, early-stopping on
    validation loss, and restores the best checkpoint (not the last
    epoch's weights) before returning.

    Four mechanics drive every iteration:
      - forward propagation: input -> logits
      - loss: CrossEntropyLoss(logits, labels) -> scalar
      - backpropagation: loss.backward() computes d(loss)/d(weight) for
        every weight via the chain rule
      - optimizer: applies the weight update using those gradients
    """
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        # ---- Training phase ----
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # ---- Validation phase ----
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[{model_name}] epoch {epoch:02d} | train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | lr={current_lr:.2e}")

        # ---- Early stopping ----
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[{model_name}] early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate(model: nn.Module, test_loader) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train EuroSAT RGB and/or multispectral SE-ResEuroNet models.")
    parser.add_argument("--modality", choices=["rgb", "multispectral", "both"], default="both")

    parser.add_argument("--data-root", default="./data",
                         help="Root dir for the torchvision-managed RGB EuroSAT download.")
    parser.add_argument("--ms-train-root", default="./data_ms/train")
    parser.add_argument("--ms-val-root", default="./data_ms/val")
    parser.add_argument("--ms-test-root", default="./data_ms/test")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output-dir", default="./checkpoints",
                         help="Where rgb_model_best.pt / multispectral_model_best.pt are written.")
    parser.add_argument("--stats-path", default="./normalization_stats.json",
                         help="Frozen normalization contract shared with app.py.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_mean = rgb_std = ms_mean = ms_std = None

    if args.modality in ("rgb", "both"):
        print("=== RGB pipeline ===")
        rgb_train_loader, rgb_val_loader, rgb_test_loader, rgb_mean, rgb_std = build_rgb_dataloaders(
            data_root=args.data_root, batch_size=args.batch_size,
            num_workers=args.num_workers, seed=args.seed,
        )
        rgb_model = SEResEuroNet(in_channels=3, num_classes=NUM_CLASSES)
        rgb_model, _ = train_model(
            rgb_model, rgb_train_loader, rgb_val_loader, "RGB",
            num_epochs=args.epochs, lr=args.lr, patience=args.patience,
        )
        rgb_test_acc = evaluate(rgb_model, rgb_test_loader)
        print(f"RGB test accuracy: {rgb_test_acc:.4f}")
        torch.save(rgb_model.state_dict(), output_dir / "rgb_model_best.pt")
        print(f"Saved {output_dir / 'rgb_model_best.pt'}")

    if args.modality in ("multispectral", "both"):
        print("=== Multispectral pipeline ===")
        ms_train_loader, ms_val_loader, ms_test_loader, ms_mean, ms_std = build_multispectral_dataloaders(
            train_root=args.ms_train_root, val_root=args.ms_val_root, test_root=args.ms_test_root,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        ms_model = SEResEuroNet(in_channels=13, num_classes=NUM_CLASSES)
        ms_model, _ = train_model(
            ms_model, ms_train_loader, ms_val_loader, "Multispectral",
            num_epochs=args.epochs, lr=args.lr, patience=args.patience,
        )
        ms_test_acc = evaluate(ms_model, ms_test_loader)
        print(f"Multispectral test accuracy: {ms_test_acc:.4f}")
        torch.save(ms_model.state_dict(), output_dir / "multispectral_model_best.pt")
        print(f"Saved {output_dir / 'multispectral_model_best.pt'}")

    # ---- Freeze normalization stats ----
    # Preserve whichever modality's stats already exist on disk if this
    # run only trained the other modality, so a partial run never
    # clobbers a previously-frozen contract.
    existing = {}
    stats_path = Path(args.stats_path)
    if stats_path.exists():
        with open(stats_path, "r") as f:
            existing = json.load(f)

    final_rgb_mean = rgb_mean if rgb_mean is not None else existing.get("rgb", {}).get("mean")
    final_rgb_std = rgb_std if rgb_std is not None else existing.get("rgb", {}).get("std")
    final_ms_mean = ms_mean if ms_mean is not None else existing.get("multispectral", {}).get("mean")
    final_ms_std = ms_std if ms_std is not None else existing.get("multispectral", {}).get("std")

    save_normalization_stats(final_rgb_mean, final_rgb_std, final_ms_mean, final_ms_std, path=str(stats_path))
    print(f"Froze normalization stats to {stats_path}")


if __name__ == "__main__":
    main()
