"""
Train ResNet-18 (CIFAR variant) on the 45k train split.

Run this AFTER scripts/make_splits.py has produced data/split_indices.npz.

Schedule: ~35 epochs, OneCycle-style LR schedule, SGD + momentum.
This is the "fast" budget (~30-40 min on a Colab T4) rather than the full
100-200 epoch paper schedule. It will produce a real, usable, overconfident
ResNet-18 -- which is exactly what this project needs (a clean miscalibration
signal), just with slightly lower peak accuracy (~90-92% vs ~94-95% for the
full schedule). That's a fine tradeoff: the calibration *story* doesn't need
the model to be at its absolute best, it needs the model to be normally
trained and then honestly measured.

Usage (from the project root):
    python scripts/train.py

Outputs:
    models/resnet18_cifar_trained.pt  -- model weights (state_dict)
    results/training_log.csv          -- per-epoch train/val loss & accuracy
"""

import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.resnet import resnet18_cifar

# ---- Config ----
DATA_DIR = "./cifar10_data"
SPLIT_PATH = "./data/split_indices.npz"
MODEL_OUT_PATH = "./models/resnet18_cifar_trained.pt"
LOG_OUT_PATH = "./results/training_log.csv"

EPOCHS = 35
BATCH_SIZE = 128
MAX_LR = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
NUM_WORKERS = 2

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders():
    """
    Two separate transform pipelines, both built on the SAME underlying
    CIFAR10 train data but with different indices:
      - train_loader: 45k split, WITH augmentation (random crop + flip)
      - val_loader:   the 5k calibration split, NO augmentation -- this is
                       just to monitor val accuracy during training as a
                       sanity check / checkpoint selector. It is NOT where
                       calibration parameters get fit (that happens later,
                       in a separate script, using saved logits).
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset_full = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    val_dataset_full = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=eval_transform
    )

    split = np.load(SPLIT_PATH)
    train_idx, cal_idx = split["train_idx"], split["cal_idx"]

    train_subset = Subset(train_dataset_full, train_idx)
    val_subset = Subset(val_dataset_full, cal_idx)  # cal split, used read-only here

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print(
            "WARNING: no GPU detected. This script will be extremely slow "
            "on CPU. Make sure you're running this on a Colab/Kaggle GPU runtime."
        )

    train_loader, val_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val (cal-split) batches: {len(val_loader)}")

    model = resnet18_cifar(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=MAX_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
    )

    log_rows = []
    best_val_acc = 0.0

    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_OUT_PATH), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start_time = time.time()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{epoch_time:.1f}s"
        )

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT_PATH)
            print(f"  -> New best val_acc={val_acc:.4f}, checkpoint saved.")

    with open(LOG_OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    print(f"Best checkpoint saved to: {MODEL_OUT_PATH}")
    print(f"Training log saved to: {LOG_OUT_PATH}")


if __name__ == "__main__":
    main()
