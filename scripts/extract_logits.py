"""
Extract logits from the trained model on the calibration set and the
official CIFAR-10 test set.

This is the ONLY script that runs the test set through the model. After
this script finishes, test set logits are cached to disk and every later
script (calibration fitting, metric comparison) reads from that cache.
The model itself is never touched again after this point.

We save raw LOGITS, not softmax probabilities, because temperature scaling
needs to divide logits by T before applying softmax -- if we only saved
post-softmax outputs we could not undo the softmax to apply temperature
scaling correctly.

Usage (from project root):
    python scripts/extract_logits.py

Outputs:
    results/cal_logits.npz   -- logits + labels for the 5k calibration set
    results/test_logits.npz  -- logits + labels for the 10k official test set
"""

import os
import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.resnet import resnet18_cifar

DATA_DIR = "./cifar10_data"
SPLIT_PATH = "./data/split_indices.npz"
MODEL_PATH = "./models/resnet18_cifar_trained.pt"
CAL_LOGITS_OUT = "./results/cal_logits.npz"
TEST_LOGITS_OUT = "./results/test_logits.npz"

BATCH_SIZE = 256
NUM_WORKERS = 2

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_eval_transform():
    # No augmentation for logit extraction -- we want a single deterministic
    # pass per image, not the random-crop/flip distribution used in training.
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


@torch.no_grad()
def extract_logits(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = resnet18_cifar(num_classes=10).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded trained weights from {MODEL_PATH}")

    eval_transform = get_eval_transform()

    # ---- Calibration set (5k held-out, used only for fitting calibrators) ----
    cal_dataset_full = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=eval_transform
    )
    split = np.load(SPLIT_PATH)
    cal_idx = split["cal_idx"]
    cal_subset = Subset(cal_dataset_full, cal_idx)
    cal_loader = DataLoader(
        cal_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    cal_logits, cal_labels = extract_logits(model, cal_loader, device)
    assert cal_logits.shape == (5000, 10), f"Unexpected cal logits shape: {cal_logits.shape}"
    assert cal_labels.shape == (5000,)
    os.makedirs(os.path.dirname(CAL_LOGITS_OUT), exist_ok=True)
    np.savez(CAL_LOGITS_OUT, logits=cal_logits, labels=cal_labels)
    print(f"Saved calibration logits: {cal_logits.shape} -> {CAL_LOGITS_OUT}")

    # ---- Official test set (10k, untouched until now) ----
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=eval_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    test_logits, test_labels = extract_logits(model, test_loader, device)
    assert test_logits.shape == (10000, 10), f"Unexpected test logits shape: {test_logits.shape}"
    assert test_labels.shape == (10000,)
    np.savez(TEST_LOGITS_OUT, logits=test_logits, labels=test_labels)
    print(f"Saved test logits: {test_logits.shape} -> {TEST_LOGITS_OUT}")

    # Quick sanity check: accuracy from these logits should roughly match
    # the val_acc seen during training (cal set) and give us our first look
    # at true test accuracy (never seen before this point).
    cal_acc = (cal_logits.argmax(axis=1) == cal_labels).mean()
    test_acc = (test_logits.argmax(axis=1) == test_labels).mean()
    print(f"\nCalibration set accuracy: {cal_acc:.4f}")
    print(f"Test set accuracy:        {test_acc:.4f}  <-- first look, never seen until now")


if __name__ == "__main__":
    main()
