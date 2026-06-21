"""
Calibration metrics: ECE, MCE, Brier Score, NLL.
"""

import numpy as np


def softmax(logits):
    """Numerically stable softmax along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def expected_calibration_error(probs, labels, n_bins=15, binning="equal_width"):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    if binning == "equal_width":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif binning == "equal_mass":
        bin_edges = np.quantile(confidences, np.linspace(0, 1, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_edges = np.unique(bin_edges)
    else:
        raise ValueError(f"Unknown binning scheme: {binning}")

    ece = 0.0
    n_total = len(labels)
    bin_stats = []

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)

        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            bin_stats.append((lo, hi, 0, np.nan, np.nan))
            continue

        bin_confidence = confidences[in_bin].mean()
        bin_accuracy = accuracies[in_bin].mean()
        bin_weight = n_in_bin / n_total

        ece += bin_weight * abs(bin_confidence - bin_accuracy)
        bin_stats.append((lo, hi, n_in_bin, bin_confidence, bin_accuracy))

    return ece, bin_stats


def maximum_calibration_error(probs, labels, n_bins=15, binning="equal_width"):
    _, bin_stats = expected_calibration_error(probs, labels, n_bins, binning)
    gaps = [abs(conf - acc) for (_, _, n, conf, acc) in bin_stats if n > 0]
    return max(gaps) if gaps else np.nan


def brier_score(probs, labels, num_classes=10):
    n = len(labels)
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), labels] = 1.0
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def negative_log_likelihood(probs, labels, eps=1e-12):
    n = len(labels)
    true_class_probs = probs[np.arange(n), labels]
    true_class_probs = np.clip(true_class_probs, eps, 1.0)
    return -np.mean(np.log(true_class_probs))


def accuracy(probs, labels):
    predictions = probs.argmax(axis=1)
    return (predictions == labels).mean()


def compute_all_metrics(probs, labels, n_bins=15):
    ece_equal_width, _ = expected_calibration_error(probs, labels, n_bins, "equal_width")
    ece_equal_mass, _ = expected_calibration_error(probs, labels, n_bins, "equal_mass")
    mce = maximum_calibration_error(probs, labels, n_bins, "equal_width")

    return {
        "accuracy": accuracy(probs, labels),
        "ece_equal_width": ece_equal_width,
        "ece_equal_mass": ece_equal_mass,
        "mce": mce,
        "brier": brier_score(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
    }


if __name__ == "__main__":
    n, num_classes = 1000, 10
    labels = np.zeros(n, dtype=int)
    probs = np.zeros((n, num_classes))
    probs[:, 0] = 1.0

    metrics = compute_all_metrics(probs, labels)
    print("Sanity check (perfect classifier, should all be ~0):")
    for k, v in metrics.items():
        if k != "accuracy":
            assert abs(v) < 1e-9, f"{k} should be 0, got {v}"
    print(metrics)
    print("All metrics correctly near zero for a perfect classifier.")
