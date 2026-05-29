"""Model pipelines for Phase 6.

Each factory returns a sklearn `Pipeline` that contains the FULL set
of train-fold transformations:
    SimpleImputer(median) -> StandardScaler -> SelectKBest(MI) -> estimator

This is critical: fitting the imputer or the MI-based selector on the
entire dataset (instead of inside each fold) would leak target
information through the variance and the MI ranking. The Pipeline is
constructed once and re-fit on each fold's training portion.

The MI-based SelectKBest uses scikit-learn's `mutual_info_classif`
internally, which is non-deterministic across calls. We set
`random_state` on the selector for reproducibility.
"""
from __future__ import annotations

from functools import partial
from typing import Callable

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


# Class balance in train (constant for this project): 39 negative, 16 positive.
SCALE_POS_WEIGHT = 39.0 / 16.0


def _mi_score(X, y, *, random_state: int = 0):
    """Top-level scoring callable (picklable, unlike a closure)."""
    return mutual_info_classif(
        X, y, random_state=random_state, n_neighbors=3, discrete_features=False
    )


def _mi_selector(k: int, seed: int) -> SelectKBest:
    return SelectKBest(
        score_func=partial(_mi_score, random_state=seed),
        k=k,
    )


def _prefix(name: str, k: int, seed: int) -> list:
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", _mi_selector(k=k, seed=seed)),
    ]


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def baseline_majority(*, k: int = 1, seed: int = 0) -> Pipeline:
    # No selection needed for the dummy, but we keep the imputer for safety.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("est", DummyClassifier(strategy="most_frequent")),
    ])


def baseline_clinical_logreg(*, k: int = 7, seed: int = 0) -> Pipeline:
    """Clinical-only LogReg. `k` defaults to "all clinical features"."""
    return Pipeline([
        *_prefix("clinical", k=k, seed=seed),
        ("est", LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear",
            class_weight="balanced", max_iter=2000, random_state=seed,
        )),
    ])


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


def logreg_l2(*, k: int = 15, seed: int = 0) -> Pipeline:
    return Pipeline([
        *_prefix("logreg_l2", k=k, seed=seed),
        ("est", LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear",
            class_weight="balanced", max_iter=2000, random_state=seed,
        )),
    ])


def logreg_elasticnet(*, k: int = 15, seed: int = 0) -> Pipeline:
    return Pipeline([
        *_prefix("logreg_en", k=k, seed=seed),
        ("est", LogisticRegression(
            penalty="elasticnet", solver="saga",
            l1_ratio=0.5, C=0.5,
            class_weight="balanced", max_iter=5000, random_state=seed,
        )),
    ])


def random_forest(*, k: int = 15, seed: int = 0) -> Pipeline:
    return Pipeline([
        *_prefix("rf", k=k, seed=seed),
        ("est", RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=3,
            max_features="sqrt", class_weight="balanced", n_jobs=-1,
            random_state=seed,
        )),
    ])


def lightgbm(*, k: int = 15, seed: int = 0) -> Pipeline:
    if not _HAS_LGBM:
        raise RuntimeError("lightgbm is not installed in this environment.")
    return Pipeline([
        *_prefix("lgbm", k=k, seed=seed),
        ("est", LGBMClassifier(
            objective="binary",
            n_estimators=300,
            num_leaves=8,
            min_child_samples=5,
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=SCALE_POS_WEIGHT,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


PipelineFactory = Callable[..., Pipeline]

PIPELINES: dict[str, PipelineFactory] = {
    "baseline_majority": baseline_majority,
    "baseline_clinical_logreg": baseline_clinical_logreg,
    "logreg_l2": logreg_l2,
    "logreg_elasticnet": logreg_elasticnet,
    "random_forest": random_forest,
}
if _HAS_LGBM:
    PIPELINES["lightgbm"] = lightgbm
