"""Slice-level Cross Validation with patient-level metrics.

The runner does:
    1. StratifiedGroupKFold on (y_slice, groups=patient_sk).
       This keeps a patient's slices together (in train OR in val).
    2. For each fold:
        - Fit pipeline on train SLICES.
        - Predict probabilities on val SLICES.
    3. Concatenate val-slice predictions into a full OOF vector
       covering every train slice exactly once.
    4. Aggregate slice-level OOF probs to patient-level (one per
       patient) using one of `acv.modeling.aggregate.AGGREGATORS`.
    5. Tune threshold on the patient-level OOF probs.
    6. Compute metrics + bootstrap CI at patient level.

Two CV outputs are kept:
    - The slice-level OOF (for diagnostics, calibration).
    - The patient-level OOF + metrics (the medically relevant one).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from acv.config import settings
from acv.modeling.aggregate import AGGREGATORS, aggregate_to_patient
from acv.modeling.metrics import (
    MetricsBundle,
    bootstrap_metric,
    compute_metrics,
    pick_threshold_max_sensitivity_at_specificity,
    sensitivity,
)


@dataclass
class SliceCVResult:
    name: str
    aggregator: str
    threshold: float
    metrics: MetricsBundle
    sens_ci: tuple[float, float, float]
    auroc_ci: tuple[float, float, float]
    oof_slice_proba: np.ndarray
    oof_patient_proba: np.ndarray
    oof_patient_true: np.ndarray
    oof_patient_groups: np.ndarray
    slice_groups: np.ndarray


def _slice_oof(pipeline_factory, X, y, groups, *, n_splits: int, seed: int, k: int | None):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)
    covered = np.zeros(len(X), dtype=bool)
    for tr, va in cv.split(X, y, groups=groups):
        pipe = pipeline_factory(seed=seed) if k is None else pipeline_factory(k=k, seed=seed)
        pipe.fit(X.iloc[tr], y[tr])
        if hasattr(pipe.named_steps["est"], "predict_proba"):
            proba = pipe.predict_proba(X.iloc[va])[:, 1]
        else:
            proba = pipe.predict(X.iloc[va]).astype(float)
        oof[va] = proba
        covered[va] = True
    assert covered.all(), "Some slices uncovered by CV."
    return oof


def _patient_truth(y_slice: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(groups)
    y_slice = np.asarray(y_slice, dtype=int)
    unique = np.sort(np.unique(groups))
    truth = np.empty(unique.shape, dtype=int)
    for i, g in enumerate(unique):
        truth[i] = int(y_slice[groups == g].max())
    return truth, unique


def run_slice_cv(
    name: str,
    pipeline_factory,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    *,
    aggregator: str = "mean",
    n_splits: int = 5,
    seed: int | None = None,
    min_specificity: float | None = None,
    bootstrap_n: int = 500,
    k: int | None = None,
    calibrate: bool = False,
    oof_slice_precomputed: np.ndarray | None = None,
) -> SliceCVResult:
    seed = seed if seed is not None else settings.random_seed
    min_specificity = (
        min_specificity if min_specificity is not None else settings.min_specificity
    )

    if oof_slice_precomputed is not None:
        oof_slice = oof_slice_precomputed
    else:
        oof_slice = _slice_oof(pipeline_factory, X, y, groups, n_splits=n_splits, seed=seed, k=k)
    pat_proba, pat_groups = aggregate_to_patient(
        oof_slice, groups, weights, aggregator=aggregator
    )
    pat_truth, pat_truth_groups = _patient_truth(y, groups)

    if calibrate:
        # Isotonic on patient-level OOF probs against patient-level truth.
        # Honest because the OOF probs were never trained on these labels.
        iso = IsotonicRegression(out_of_bounds="clip")
        pat_proba = iso.fit_transform(pat_proba, pat_truth)
    # ensure aligned by patient_sk
    assert np.array_equal(pat_groups, pat_truth_groups), "patient ordering mismatch"

    choice = pick_threshold_max_sensitivity_at_specificity(
        pat_truth, pat_proba, min_specificity
    )
    metrics = compute_metrics(pat_truth, pat_proba, choice.threshold)

    sens_at_thr = lambda yt, yp: sensitivity(yt, (yp >= choice.threshold).astype(int))
    sens_ci = bootstrap_metric(
        pat_truth, pat_proba, sens_at_thr, n_resamples=bootstrap_n,
        groups=pat_groups, random_state=seed,
    )
    auroc_ci = bootstrap_metric(
        pat_truth, pat_proba, roc_auc_score, n_resamples=bootstrap_n,
        groups=pat_groups, random_state=seed,
    )

    return SliceCVResult(
        name=name,
        aggregator=aggregator,
        threshold=choice.threshold,
        metrics=metrics,
        sens_ci=sens_ci,
        auroc_ci=auroc_ci,
        oof_slice_proba=oof_slice,
        oof_patient_proba=pat_proba,
        oof_patient_true=pat_truth,
        oof_patient_groups=pat_groups,
        slice_groups=groups,
    )
