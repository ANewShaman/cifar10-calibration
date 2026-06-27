"""
Calibration methods: Temperature Scaling, Platt Scaling, Isotonic Regression.

All three are fit ONLY on the calibration set (cal_logits.npz) and applied
to the test set (test_logits.npz). Never fit on test data - that would be
the exact leak this whole project is designed to avoid.

Temperature scaling: hand-rolled. It's a single scalar T fit by minimizing
NLL on the calibration set logits via 1D optimization (scipy.optimize). This
is simple enough that a library wrapper would add nothing - the entire
method IS a five-line optimization loop.

Platt scaling & isotonic regression: use sklearn.linear_model.LogisticRegression
(for Platt) and sklearn.isotonic.IsotonicRegression directly, in a one-vs-rest
loop over the 10 classes, then renormalize so probabilities sum to 1.

WHY renormalization is necessary: fitting 10 independent binary calibrators
(one per class, "is it class c or not") gives you 10 independent probability
estimates that have no reason to sum to 1 across classes. Renormalizing
(dividing each by the row sum) is the standard fix - it's a reasonable but
NOT theoretically perfect solution (the one-vs-rest decomposition itself is
an approximation for multiclass problems), which is part of why temperature
scaling's clean "argmax-preserving, sums-to-1-by-construction" property is
attractive by comparison. Worth saying explicitly in the writeup.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.metrics import softmax, negative_log_likelihood



# Temperature Scaling

def fit_temperature(cal_logits, cal_labels):
    """
    Fit a single scalar T > 0 by minimizing NLL of softmax(logits / T) on
    the calibration set. This is a convex 1D optimization -- we use scipy's
    bounded scalar minimizer rather than full LBFGS since one parameter
    doesn't need anything fancier.

    Returns: optimal T (float)
    """
    def nll_for_temperature(T):
        scaled_probs = softmax(cal_logits / T)
        return negative_log_likelihood(scaled_probs, cal_labels)

    result = minimize_scalar(
        nll_for_temperature, bounds=(0.05, 10.0), method="bounded"
    )
    return result.x


def apply_temperature(logits, T):
    """Apply a fitted temperature to a logits array, return calibrated probs."""
    return softmax(logits / T)

# Platt Scaling (one-vs-rest logistic regression on logits)

def fit_platt(cal_logits, cal_labels, num_classes=10):
    """
    Fit one binary logistic regression per class: for class c, the binary
    target is 1 if the true label is c, else 0, and the single input
    feature is that class's raw logit.

    Returns: list of fitted LogisticRegression objects, one per class.
    """
    calibrators = []
    for c in range(num_classes):
        binary_target = (cal_labels == c).astype(int)
        feature = cal_logits[:, c].reshape(-1, 1)

        lr = LogisticRegression()
        lr.fit(feature, binary_target)
        calibrators.append(lr)
    return calibrators


def apply_platt(logits, calibrators):
    """
    Apply fitted per-class Platt calibrators, then renormalize across
    classes so each row sums to 1.
    """
    num_classes = len(calibrators)
    n = logits.shape[0]
    raw_probs = np.zeros((n, num_classes))

    for c in range(num_classes):
        feature = logits[:, c].reshape(-1, 1)
        raw_probs[:, c] = calibrators[c].predict_proba(feature)[:, 1]

    row_sums = raw_probs.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-12, None)
    normalized_probs = raw_probs / row_sums
    return normalized_probs

# Isotonic Regression (one-vs-rest)

def fit_isotonic(cal_probs, cal_labels, num_classes=10):
    """
    Fit one isotonic regression per class, mapping that class's predicted
    probability -> calibrated probability via a monotonic step function.

    NOTE: isotonic regression is fit on PROBABILITIES (post-softmax), not
    raw logits, since it needs a bounded [0,1] input to fit a monotonic
    function meaningfully.

    Returns: list of fitted IsotonicRegression objects, one per class.
    """
    calibrators = []
    for c in range(num_classes):
        binary_target = (cal_labels == c).astype(int)
        feature = cal_probs[:, c]

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(feature, binary_target)
        calibrators.append(iso)
    return calibrators


def apply_isotonic(probs, calibrators):
    """
    Apply fitted per-class isotonic calibrators to (post-softmax)
    probabilities, then renormalize across classes.
    """
    num_classes = len(calibrators)
    n = probs.shape[0]
    raw_probs = np.zeros((n, num_classes))

    for c in range(num_classes):
        raw_probs[:, c] = calibrators[c].predict(probs[:, c])

    row_sums = raw_probs.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-12, None)
    normalized_probs = raw_probs / row_sums
    return normalized_probs


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, num_classes = 2000, 10
    cal_labels = rng.integers(0, num_classes, size=n)
    cal_logits = rng.normal(0, 1, size=(n, num_classes))
    for i in range(n):
        cal_logits[i, cal_labels[i]] += rng.uniform(2, 5)

    T = fit_temperature(cal_logits, cal_labels)
    temp_probs = apply_temperature(cal_logits, T)
    assert np.allclose(temp_probs.sum(axis=1), 1.0, atol=1e-6)
    print(f"Temperature scaling: T={T:.4f}, probs sum to 1: confirmed")

    platt_cals = fit_platt(cal_logits, cal_labels, num_classes)
    platt_probs = apply_platt(cal_logits, platt_cals)
    assert np.allclose(platt_probs.sum(axis=1), 1.0, atol=1e-6)
    print(f"Platt scaling: probs sum to 1: confirmed")

    cal_probs_baseline = softmax(cal_logits)
    iso_cals = fit_isotonic(cal_probs_baseline, cal_labels, num_classes)
    iso_probs = apply_isotonic(cal_probs_baseline, iso_cals)
    assert np.allclose(iso_probs.sum(axis=1), 1.0, atol=1e-6)
    print(f"Isotonic regression: probs sum to 1: confirmed")

    print("\nAll three calibration methods run end-to-end correctly.")
