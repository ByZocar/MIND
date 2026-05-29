"""Slice -> patient probability aggregation.

After a slice-level classifier predicts P(>4.5h | slice_i) for each
slice belonging to a patient, we need to collapse those probabilities
into ONE patient-level probability. The choice of aggregation function
is itself a hyperparameter:

- mean    : averages all slice probabilities. Stable, ignores extremes.
- max     : the most pessimistic slice wins. Sensitive to one strong
            piece of evidence; can over-trigger.
- p75     : 75th-percentile probability. Robust max.
- noisy_or: 1 - prod(1 - P_i). Treats each slice as an independent
            chance of triggering the >4.5h label. Mathematically natural
            for "any slice evidences out-of-window".
- weighted_mean: same as mean but weighted by slice lesion-area share.

The runner in `acv.modeling.slice_cv` tries multiple aggregators and
reports the best one.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

AggFn = Callable[[np.ndarray, np.ndarray], float]


def agg_mean(p: np.ndarray, w: np.ndarray) -> float:
    return float(np.mean(p))


def agg_max(p: np.ndarray, w: np.ndarray) -> float:
    return float(np.max(p))


def agg_p75(p: np.ndarray, w: np.ndarray) -> float:
    return float(np.percentile(p, 75))


def agg_noisy_or(p: np.ndarray, w: np.ndarray) -> float:
    p_clip = np.clip(p, 0.0, 0.999999)
    return float(1.0 - np.prod(1.0 - p_clip))


def agg_weighted_mean(p: np.ndarray, w: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    if w.shape != p.shape or w.sum() <= 0:
        return float(np.mean(p))
    return float(np.average(p, weights=w))


def agg_geom_mean(p: np.ndarray, w: np.ndarray) -> float:
    """Geometric mean: penalizes near-zero probabilities; stricter than mean."""
    p = np.asarray(p, dtype=float).ravel()
    p_clip = np.clip(p, 1e-6, 1.0)
    return float(np.exp(np.mean(np.log(p_clip))))


def agg_voting(p: np.ndarray, w: np.ndarray, *, vote_thr: float = 0.5) -> float:
    """Fraction of slices that individually predict positive at vote_thr."""
    p = np.asarray(p, dtype=float).ravel()
    return float(np.mean(p >= vote_thr))


def agg_topk_mean(p: np.ndarray, w: np.ndarray, *, k_frac: float = 0.3) -> float:
    """Average over the top fraction k_frac of slice probabilities.

    Motivation: the lesion is heterogeneous; the most-evident slices may
    carry the signal, while the rest dilute it.
    """
    p = np.asarray(p, dtype=float).ravel()
    n = max(1, int(np.ceil(k_frac * p.size)))
    return float(np.mean(np.sort(p)[-n:]))


AGGREGATORS: dict[str, AggFn] = {
    "mean": agg_mean,
    "max": agg_max,
    "p75": agg_p75,
    "noisy_or": agg_noisy_or,
    "weighted_mean": agg_weighted_mean,
    "geom_mean": agg_geom_mean,
    "voting": agg_voting,
    "topk_mean": agg_topk_mean,
}


def aggregate_to_patient(
    slice_proba: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    *,
    aggregator: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (patient_proba, patient_groups) — sorted by patient key."""
    if aggregator not in AGGREGATORS:
        raise KeyError(f"Unknown aggregator: {aggregator}")
    fn = AGGREGATORS[aggregator]

    groups = np.asarray(groups)
    slice_proba = np.asarray(slice_proba, dtype=float)
    weights = np.asarray(weights, dtype=float)

    unique_groups = np.sort(np.unique(groups))
    out = np.empty(unique_groups.shape, dtype=float)
    for i, g in enumerate(unique_groups):
        mask = groups == g
        out[i] = float(fn(slice_proba[mask], weights[mask]))
    return out, unique_groups
