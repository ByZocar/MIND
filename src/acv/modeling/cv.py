"""Group-aware cross-validation runner.

For each pipeline:
    1. StratifiedGroupKFold(n_splits=5) on (y_train, groups=patient_sk).
    2. Fit pipeline on each train fold; predict probabilities on the
       validation fold.
    3. Concatenate validation predictions into an out-of-fold (OOF)
       prediction vector covering ALL train patients exactly once.
    4. Tune the decision threshold on the OOF predictions, subject to
       the minimum specificity constraint.
    5. Compute the metrics bundle on OOF predictions at that threshold.

The OOF approach is more reliable than per-fold metric averaging on a
small N=55 train set, because the per-fold validation sets are tiny
(~11 patients with only ~3 positives each) — individual fold metrics
swing wildly. Aggregating into OOF gives one stable estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from acv.config import settings
from acv.modeling.metrics import (
    MetricsBundle,
    bootstrap_metric,
    compute_metrics,
    pick_threshold_max_sensitivity_at_specificity,
    sensitivity,
)


@dataclass
class CVResult:
    name: str
    threshold: float
    metrics: MetricsBundle
    sens_ci: tuple[float, float, float]  # (point, low, high)
    auroc_ci: tuple[float, float, float]
    oof_proba: np.ndarray
    oof_true: np.ndarray
    oof_groups: np.ndarray


def run_cv(
    name: str,
    pipeline_factory,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = None,
    min_specificity: float = None,
    bootstrap_n: int = 1000,
    k: int | None = None,
) -> CVResult:
    """Run one pipeline through StratifiedGroupKFold and return its OOF result."""
    seed = seed if seed is not None else settings.random_seed
    min_specificity = (
        min_specificity if min_specificity is not None else settings.min_specificity
    )

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_proba = np.zeros(len(X), dtype=float)
    oof_assigned = np.zeros(len(X), dtype=bool)

    for fold_idx, (tr, va) in enumerate(cv.split(X, y, groups=groups)):
        pipe = pipeline_factory(seed=seed) if k is None else pipeline_factory(k=k, seed=seed)
        pipe.fit(X.iloc[tr], y[tr])
        if hasattr(pipe.named_steps["est"], "predict_proba"):
            proba = pipe.predict_proba(X.iloc[va])[:, 1]
        else:
            proba = pipe.predict(X.iloc[va]).astype(float)
        oof_proba[va] = proba
        oof_assigned[va] = True
    assert oof_assigned.all(), "Some training rows were not covered by CV folds."

    choice = pick_threshold_max_sensitivity_at_specificity(y, oof_proba, min_specificity)
    metrics = compute_metrics(y, oof_proba, choice.threshold)

    sens_at_thr = lambda yt, yp: sensitivity(yt, (yp >= choice.threshold).astype(int))
    sens_ci = bootstrap_metric(
        y, oof_proba, sens_at_thr, n_resamples=bootstrap_n,
        groups=groups, random_state=seed,
    )
    from sklearn.metrics import roc_auc_score
    auroc_ci = bootstrap_metric(
        y, oof_proba, roc_auc_score, n_resamples=bootstrap_n,
        groups=groups, random_state=seed,
    )

    return CVResult(
        name=name,
        threshold=choice.threshold,
        metrics=metrics,
        sens_ci=sens_ci,
        auroc_ci=auroc_ci,
        oof_proba=oof_proba,
        oof_true=y,
        oof_groups=groups,
    )
