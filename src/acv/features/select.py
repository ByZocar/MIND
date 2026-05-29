"""Feature-selection helpers used by Phase 5 and Phase 6.

These are *pure functions* over a wide DataFrame; they never call the DB
and they never carry global state. This keeps them easy to test and to
re-use inside a `sklearn.Pipeline` for the modeling phase.

Selection strategy (v1):
1. drop_zero_variance       : remove constants and near-constants.
2. drop_high_correlation    : remove |Spearman| > threshold, keeping the
                              one with higher MI vs the target.
3. rank_by_mutual_info      : Top-K ranking on the survivors.

Notes:
- For high-correlation pruning we use Spearman (rank-based), robust to
  outliers and monotonic non-linearities.
- For MI we use `mutual_info_classif`. With N=55 patients in train, the
  estimates are noisy; this is a *soft* signal — the model in Phase 6
  does its own L1 / Boruta selection inside CV folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from acv.features.inventory import (
    GROUP_KEY,
    NATURAL_PATIENT_KEY,
    PATIENT_KEY,
    PROHIBITED_AS_FEATURE,
    TARGET_COL,
)

NON_FEATURE_COLS = frozenset(
    {PATIENT_KEY, NATURAL_PATIENT_KEY, GROUP_KEY, TARGET_COL} | PROHIBITED_AS_FEATURE
)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def drop_zero_variance(df: pd.DataFrame, eps: float = 1e-10) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns whose stddev is below `eps`. Returns (cleaned, dropped)."""
    feats = _feature_columns(df)
    keep, dropped = [], []
    for c in feats:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].std(skipna=True) > eps:
            keep.append(c)
        else:
            dropped.append(c)
    return df[[*[c for c in df.columns if c not in feats], *keep]], dropped


def drop_high_correlation(
    df: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.95,
    method: str = "spearman",
) -> tuple[pd.DataFrame, list[tuple[str, str, float]]]:
    """For each pair with |corr| > threshold, keep the feature with higher MI vs y.

    Returns (cleaned_df, dropped_pairs) where each pair is
    (kept_feature, dropped_feature, corr).
    """
    feats = _feature_columns(df)
    X = df[feats].copy()
    X = X.fillna(X.median(numeric_only=True))

    # MI vs target for ranking inside each pair (small N -> use n_neighbors=3).
    mi = mutual_info_classif(
        X.to_numpy(), y.to_numpy(), random_state=0, n_neighbors=3, discrete_features=False
    )
    mi_series = pd.Series(mi, index=feats)

    corr = X.corr(method=method).abs()
    # Upper triangle (avoid (a,b) and (b,a) twice).
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    dropped: set[str] = set()
    dropped_pairs: list[tuple[str, str, float]] = []
    pairs = (
        upper.stack()
        .loc[lambda s: s > threshold]
        .sort_values(ascending=False)
    )
    for (a, b), corr_val in pairs.items():
        if a in dropped or b in dropped:
            continue
        # keep the one with higher MI
        loser = b if mi_series[a] >= mi_series[b] else a
        winner = a if loser == b else b
        dropped.add(loser)
        dropped_pairs.append((winner, loser, float(corr_val)))

    keep = [c for c in feats if c not in dropped]
    return df[[*[c for c in df.columns if c not in feats], *keep]], dropped_pairs


def rank_by_mutual_info(
    df: pd.DataFrame, y: pd.Series, random_state: int = 0
) -> pd.DataFrame:
    """Return a DataFrame with feature name + MI estimate, sorted desc."""
    feats = _feature_columns(df)
    X = df[feats].copy()
    X = X.fillna(X.median(numeric_only=True))
    mi = mutual_info_classif(
        X.to_numpy(),
        y.to_numpy(),
        random_state=random_state,
        n_neighbors=3,
        discrete_features=False,
    )
    return (
        pd.DataFrame({"feature": feats, "mi": mi})
        .sort_values("mi", ascending=False)
        .reset_index(drop=True)
    )
