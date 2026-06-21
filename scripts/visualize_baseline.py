"""
Baseline calibration visualizations: reliability diagram + confidence
histogram, computed on the UNCALIBRATED test set logits.

This script is read-only with respect to the model and data pipeline --
it just loads the cached logits from extract_logits.py and plots them.
In Phase 4, the same plotting functions get reused on post-calibration
probabilities for side-by-side before/after comparison.

Usage (from project root):
    python scripts/visualize_baseline.py

Outputs:
    results/reliability_diagram_baseline.png
    results/confidence_histogram_baseline.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for Colab/scripts
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.metrics import softmax, expected_calibration_error, compute_all_metrics

TEST_LOGITS_PATH = "./results/test_logits.npz"
RELIABILITY_OUT = "./results/reliability_diagram_baseline.png"
HISTOGRAM_OUT = "./results/confidence_histogram_baseline.png"
N_BINS = 15


def plot_reliability_diagram(probs, labels, n_bins, title, save_path):
    """
    x-axis: predicted confidence (binned)
    y-axis: empirical accuracy in that bin
    Diagonal: perfect calibration reference (y=x)
    Bars below the diagonal in high-confidence bins = overconfidence,
    the visual signature this whole project is built around.
    """
    ece, bin_stats = expected_calibration_error(probs, labels, n_bins, "equal_width")

    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for (lo, hi, n, conf, acc) in bin_stats:
        bin_centers.append((lo + hi) / 2)
        bin_accuracies.append(acc if n > 0 else 0)
        bin_confidences.append(conf if n > 0 else 0)
        bin_counts.append(n)

    bin_centers = np.array(bin_centers)
    bin_accuracies = np.array(bin_accuracies)
    bin_width = 1.0 / n_bins

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    ax.bar(
        bin_centers,
        bin_accuracies,
        width=bin_width * 0.9,
        edgecolor="black",
        color="steelblue",
        label="Accuracy",
        alpha=0.8,
    )

    for center, acc in zip(bin_centers, bin_accuracies):
        if acc == 0 and center == 0:
            continue
        ax.plot(
            [center, center],
            [acc, center],
            color="red",
            linewidth=1.5,
            alpha=0.6,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved reliability diagram to {save_path}")
    return ece


def plot_confidence_histogram(probs, title, save_path):
    """
    Distribution of max-softmax confidence across the test set.
    Uncalibrated networks typically show a histogram heavily skewed toward
    0.9-1.0, i.e. most predictions claim near-certainty.
    """
    confidences = probs.max(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(confidences, bins=20, range=(0, 1), color="steelblue", edgecolor="black")
    ax.set_xlabel("Confidence (max softmax probability)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.axvline(
        confidences.mean(), color="red", linestyle="--",
        label=f"Mean confidence = {confidences.mean():.3f}",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved confidence histogram to {save_path}")


def main():
    data = np.load(TEST_LOGITS_PATH)
    test_logits, test_labels = data["logits"], data["labels"]
    test_probs = softmax(test_logits)

    metrics = compute_all_metrics(test_probs, test_labels, n_bins=N_BINS)
    print("Baseline (uncalibrated) test set metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    plot_reliability_diagram(
        test_probs, test_labels, N_BINS,
        title="Reliability Diagram (Uncalibrated)",
        save_path=RELIABILITY_OUT,
    )
    plot_confidence_histogram(
        test_probs,
        title="Confidence Histogram (Uncalibrated)",
        save_path=HISTOGRAM_OUT,
    )

    import json
    with open("./results/baseline_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print("Saved baseline_metrics.json")


if __name__ == "__main__":
    main()
