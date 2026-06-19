"""
Stratified split of CIFAR-10 into train / calibration / test.

This is the single most important script in the whole project from a
methodology standpoint.

Rules enforced here:
1. The official CIFAR-10 test set (10,000 images) is NEVER touched until
   final evaluation. It is not used for training, not used for fitting
   calibration parameters, not used for any hyperparameter decision.
2. The official CIFAR-10 train set (50,000 images) is split into:
   - train:       45,000 images -> used to fit the classifier
   - calibration:  5,000 images -> used ONLY to fit calibration methods
                    (temperature scaling T, isotonic regression curve, etc.)
3. The split is stratified by class, so train/cal both have ~4500/500
   images per class (CIFAR-10 has exactly 5000 images/class in the
   official 50k train set, so this divides evenly).
4. The split is done ONCE and indices are saved to disk, so every later
   script (training, calibration fitting, evaluation) loads the same
   split rather than re-randomizing it.

Output: data/split_indices.npz containing 'train_idx' and 'cal_idx',
arrays of integer indices into the official CIFAR-10 train dataset.
"""

import numpy as np
import torchvision
from collections import defaultdict

SEED = 42
CAL_SIZE_PER_CLASS = 500  # 500 * 10 classes = 5000 total calibration images
DATA_DIR = "./cifar10_data"
OUTPUT_PATH = "./data/split_indices.npz"


def main():
    rng = np.random.default_rng(SEED)

    # Downloads CIFAR-10 train set if not already present. We only need
    # the labels here, not the images, to figure out the stratified split.
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True
    )
    labels = np.array(train_dataset.targets)
    assert len(labels) == 50000, f"Expected 50000 train images, got {len(labels)}"

    # Group indices by class.
    indices_by_class = defaultdict(list)
    for idx, label in enumerate(labels):
        indices_by_class[label].append(idx)

    train_idx = []
    cal_idx = []

    for class_label in sorted(indices_by_class.keys()):
        class_indices = np.array(indices_by_class[class_label])
        assert len(class_indices) == 5000, (
            f"Expected 5000 images for class {class_label}, "
            f"got {len(class_indices)}"
        )

        # Shuffle deterministically, then split.
        rng.shuffle(class_indices)
        cal_part = class_indices[:CAL_SIZE_PER_CLASS]
        train_part = class_indices[CAL_SIZE_PER_CLASS:]

        cal_idx.extend(cal_part.tolist())
        train_idx.extend(train_part.tolist())

    train_idx = np.array(sorted(train_idx))
    cal_idx = np.array(sorted(cal_idx))

    # Sanity checks before saving anything.
    assert len(train_idx) == 45000, f"Expected 45000 train, got {len(train_idx)}"
    assert len(cal_idx) == 5000, f"Expected 5000 cal, got {len(cal_idx)}"
    assert len(set(train_idx.tolist()) & set(cal_idx.tolist())) == 0, (
        "Train and calibration sets overlap! This must never happen."
    )
    assert len(train_idx) + len(cal_idx) == 50000

    # Confirm per-class balance in both splits (should be 4500/500).
    train_labels = labels[train_idx]
    cal_labels = labels[cal_idx]
    for class_label in range(10):
        n_train = (train_labels == class_label).sum()
        n_cal = (cal_labels == class_label).sum()
        assert n_train == 4500, f"Class {class_label} train count: {n_train}"
        assert n_cal == 500, f"Class {class_label} cal count: {n_cal}"

    np.savez(OUTPUT_PATH, train_idx=train_idx, cal_idx=cal_idx)

    print("Split created and verified:")
    print(f"  Train:       {len(train_idx)} images (4500/class)")
    print(f"  Calibration: {len(cal_idx)} images (500/class)")
    print(f"  Test:        10,000 images (official CIFAR-10 test set, "
          f"untouched, loaded separately at evaluation time)")
    print(f"  Saved to: {OUTPUT_PATH}")
    print(f"  No overlap between train and calibration indices: confirmed.")


if __name__ == "__main__":
    main()
