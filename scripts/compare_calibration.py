"""
Phase 4: fit all three calibration methods on the calibration set, apply to
the test set, and produce the centerpiece comparison table + before/after
reliability diagrams.

Strict discipline maintained: calibration methods are fit ONLY on
cal_logits.npz (5k held-out set). They are applied to test_logits.npz
(10k official test set) for evaluation ONLY.

Usage (from project root):
    python scripts/compare_calibration.py
"""

import os
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.metrics import softmax, expected_calibration_error, compute_all_metrics
from scripts.calibration_methods import (
    fit_temperature, apply_temperature,
    fit_platt, apply_platt,
    fit_isotonic, apply_isotonic,
)

CAL_LOGITS_PATH = "./results/cal_logits.npz"
TEST_LOGITS_PATH = "./results/test_logits.npz"
TABLE_CSV_OUT = "./results/calibration_comparison.csv"
TABLE_JSON_OUT = "./results/calibration_comparison.json"
DIAGRAMS_OUT = "./results/reliability_diagrams_comparison.png"
TEMPERATURE_OUT = "./results/temperature_value.txt"
N_BINS = 15
NUM_CLASSES = 10


def plot_reliability_subplot(ax, probs, labels, n_bins, title):
    ece, bin_stats = expected_calibration_error(probs, labels, n_bins, "equal_width")

    bin_centers, bin_accuracies = [], []
    for (lo, hi, n, conf, acc) in bin_stats:
        bin_centers.append((lo + hi) / 2)
        bin_accuracies.append(acc if n > 0 else 0)

    bin_centers = np.array(bin_centers)
    bin_accuracies = np.array(bin_accuracies)
    bin_width = 1.0 / n_bins

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.bar(bin_centers, bin_accuracies, width=bin_width * 0.9,
           edgecolor="black", color="steelblue", alpha=0.8)
    for center, acc in zip(bin_centers, bin_accuracies):
        ax.plot([center, center], [acc, center], color="red", linewidth=1, alpha=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}\nECE={ece:.4f}", fontsize=10)
    ax.set_xlabel("Confidence", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)


def main():
    cal_data = np.load(CAL_LOGITS_PATH)
    cal_logits, cal_labels = cal_data["logits"], cal_data["labels"]

    test_data = np.load(TEST_LOGITS_PATH)
    test_logits, test_labels = test_data["logits"], test_data["labels"]

    print(f"Calibration set: {cal_logits.shape[0]} samples")
    print(f"Test set:        {test_logits.shape[0]} samples")
    print("Fitting all calibration methods on the calibration set ONLY...\n")

    results = {}

    baseline_test_probs = softmax(test_logits)
    results["uncalibrated"] = compute_all_metrics(baseline_test_probs, test_labels, N_BINS)

    T = fit_temperature(cal_logits, cal_labels)
    temp_test_probs = apply_temperature(test_logits, T)
    results["temperature_scaling"] = compute_all_metrics(temp_test_probs, test_labels, N_BINS)
    print(f"Temperature scaling: fitted T = {T:.4f}")
    with open(TEMPERATURE_OUT, "w") as f:
        f.write(f"Fitted temperature T = {T:.6f}\n")
        f.write(f"(T > 1 means the model was overconfident; "
                f"T < 1 would mean underconfident)\n")

    platt_cals = fit_platt(cal_logits, cal_labels, NUM_CLASSES)
    platt_test_probs = apply_platt(test_logits, platt_cals)
    results["platt_scaling"] = compute_all_metrics(platt_test_probs, test_labels, N_BINS)
    print("Platt scaling: fitted (10 one-vs-rest logistic regressions)")

    cal_probs_baseline = softmax(cal_logits)
    iso_cals = fit_isotonic(cal_probs_baseline, cal_labels, NUM_CLASSES)
    iso_test_probs = apply_isotonic(baseline_test_probs, iso_cals)
    results["isotonic_regression"] = compute_all_metrics(iso_test_probs, test_labels, N_BINS)
    print("Isotonic regression: fitted (10 one-vs-rest isotonic curves)")

    print("\n" + "=" * 90)
    print(f"{'Method':<22}{'Accuracy':>10}{'ECE(ew)':>10}{'ECE(em)':>10}{'MCE':>10}{'Brier':>10}{'NLL':>10}")
    print("=" * 90)
    for method, m in results.items():
        print(f"{method:<22}{m['accuracy']:>10.4f}{m['ece_equal_width']:>10.4f}"
              f"{m['ece_equal_mass']:>10.4f}{m['mce']:>10.4f}{m['brier']:>10.4f}{m['nll']:>10.4f}")
    print("=" * 90)

    results_json_safe = {
        method: {k: float(v) for k, v in m.items()}
        for method, m in results.items()
    }
    with open(TABLE_JSON_OUT, "w") as f:
        json.dump(results_json_safe, f, indent=2)

    with open(TABLE_CSV_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "accuracy", "ece_equal_width", "ece_equal_mass",
                          "mce", "brier", "nll"])
        for method, m in results.items():
            writer.writerow([method, m["accuracy"], m["ece_equal_width"],
                              m["ece_equal_mass"], m["mce"], m["brier"], m["nll"]])
    print(f"\nSaved table to {TABLE_CSV_OUT} and {TABLE_JSON_OUT}")

    accs = [results[m]["accuracy"] for m in results]
    acc_spread = max(accs) - min(accs)
    print(f"\nAccuracy spread across methods: {acc_spread:.5f} "
          f"({'as expected, all near-identical' if acc_spread < 0.005 else 'LARGER than expected, investigate'})")

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    plot_reliability_subplot(axes[0, 0], baseline_test_probs, test_labels, N_BINS, "Uncalibrated")
    plot_reliability_subplot(axes[0, 1], temp_test_probs, test_labels, N_BINS, "Temperature Scaling")
    plot_reliability_subplot(axes[1, 0], platt_test_probs, test_labels, N_BINS, "Platt Scaling")
    plot_reliability_subplot(axes[1, 1], iso_test_probs, test_labels, N_BINS, "Isotonic Regression")
    fig.suptitle("Reliability Diagrams: Before vs. After Calibration", fontsize=13)
    fig.tight_layout()
    fig.savefig(DIAGRAMS_OUT, dpi=150)
    plt.close(fig)
    print(f"Saved comparison reliability diagrams to {DIAGRAMS_OUT}")


if __name__ == "__main__":
    main()
