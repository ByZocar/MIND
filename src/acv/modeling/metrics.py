"""Clinical-aware metrics for the ACV window classifier.

The model is judged primarily by **sensitivity (recall) of class 1**
("evolution > 4.5h, outside therapy window") *subject to* a minimum
specificity. This asymmetry is medical: see agent.md §2 KPIs and §6.4.

Threshold selection rule (`pick_threshold_max_sensitivity_at_specificity`):
    Among thresholds where specificity >= MIN_SPECIFICITY, pick the one
    with maximum sensitivity. Ties broken by higher Youden's J.

All metric helpers accept arbitrary `y_true` / `y_prob` arrays. The
bootstrap helper supports a `groups` arg so resampling is done at the
patient level — never resample slices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ThresholdChoice:
    threshold: float
    sensitivity: float
    specificity: float
    youden_j: float


def confusion_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, int]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """recall of class 1."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp, fn = cm[1, 1], cm[1, 0]
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """recall of class 0."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")


def pick_threshold_max_sensitivity_at_specificity(
    y_true: np.ndarray, y_prob: np.ndarray, min_specificity: float
) -> ThresholdChoice:
    """Sweep all candidate thresholds and pick per the spec rule.

    If no threshold meets `min_specificity`, falls back to Youden's J.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    sens = tpr
    eligible = spec >= min_specificity
    if eligible.any():
        idx_pool = np.where(eligible)[0]
        # pick the one with max sensitivity; break ties by max Youden's J
        best_idx = idx_pool[np.argmax(sens[idx_pool] + 1e-9 * (sens[idx_pool] + spec[idx_pool] - 1))]
    else:
        best_idx = int(np.argmax(sens + spec - 1))  # Youden's J fallback
    return ThresholdChoice(
        threshold=float(thresholds[best_idx]),
        sensitivity=float(sens[best_idx]),
        specificity=float(spec[best_idx]),
        youden_j=float(sens[best_idx] + spec[best_idx] - 1),
    )


@dataclass
class MetricsBundle:
    threshold: float
    sensitivity: float
    specificity: float
    youden_j: float
    auroc: float
    auprc: float
    brier: float
    accuracy: float
    n_pos: int
    n_neg: int
    confusion: dict[str, int]

    def as_row(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "youden_j": self.youden_j,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "brier": self.brier,
            "accuracy": self.accuracy,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            **{f"cm_{k}": v for k, v in self.confusion.items()},
        }


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> MetricsBundle:
    y_pred = (y_prob >= threshold).astype(int)
    sens = sensitivity(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    return MetricsBundle(
        threshold=float(threshold),
        sensitivity=sens,
        specificity=spec,
        youden_j=float(sens + spec - 1),
        auroc=float(roc_auc_score(y_true, y_prob)),
        auprc=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        accuracy=float((y_pred == y_true).mean()),
        n_pos=int((y_true == 1).sum()),
        n_neg=int((y_true == 0).sum()),
        confusion=confusion_at_threshold(y_true, y_prob, threshold),
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (patient-aware)
# ---------------------------------------------------------------------------


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
    groups: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high).

    When `groups` is provided, resampling is performed at the group
    (patient) level: pick patients with replacement, then take all of
    their rows. For our flat patient-level matrix this is equivalent to
    standard sampling-with-replacement, but the API is kept for
    portability to slice-level evaluation in the future.
    """
    rng = np.random.default_rng(random_state)
    point = float(metric_fn(y_true, y_prob))
    n = len(y_true)
    if groups is None:
        sample_units = np.arange(n)
    else:
        sample_units = np.unique(groups)
    samples = []
    for _ in range(n_resamples):
        picked = rng.choice(sample_units, size=sample_units.size, replace=True)
        if groups is None:
            idx = picked
        else:
            idx = np.concatenate([np.where(groups == g)[0] for g in picked])
        try:
            samples.append(float(metric_fn(y_true[idx], y_prob[idx])))
        except ValueError:
            continue  # all-positive or all-negative resample
    samples = np.asarray(samples)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
