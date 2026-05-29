"""Load the patient-level feature matrix and split into X/y/groups.

Single entry point so every script in `acv.modeling` consumes the
features in EXACTLY the same way. The split that matters here is
train vs test at the patient level — we never touch the test slice
during CV.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from acv.config import settings
from acv.features.inventory import (
    GROUP_KEY,
    NATURAL_PATIENT_KEY,
    PATIENT_KEY,
    PROHIBITED_AS_FEATURE,
    TARGET_COL,
)

FEATURES_PATH = settings.project_root / "data" / "processed" / "features_v1.parquet"
MI_RANK_PATH = settings.project_root / "data" / "processed" / "features_v1_mi_rank.parquet"

NON_FEATURE_COLS = frozenset(
    {PATIENT_KEY, NATURAL_PATIENT_KEY, GROUP_KEY, TARGET_COL} | PROHIBITED_AS_FEATURE
)

CLINICAL_ONLY_FEATURES = (
    "age_years",
    "sex_F",
    "nihss",
    "nihss_was_missing",
    "aspects",
    "aspects_was_missing",
    "n_slices",
)


@dataclass
class DataSplit:
    """Container with the train/test matrices and metadata."""

    X_train: pd.DataFrame
    y_train: np.ndarray
    groups_train: np.ndarray  # patient_sk; used by GroupKFold
    X_test: pd.DataFrame
    y_test: np.ndarray
    groups_test: np.ndarray
    feature_names: list[str]

    @property
    def n_train_patients(self) -> int:
        return len(self.X_train)

    @property
    def n_test_patients(self) -> int:
        return len(self.X_test)


def load_features(parquet_path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Read the feature matrix produced by `acv.features.cli`."""
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found. Run `python -m acv.features.cli` first."
        )
    return pd.read_parquet(parquet_path)


def make_split(
    df: pd.DataFrame | None = None,
    clinical_only: bool = False,
) -> DataSplit:
    """Produce X/y/groups for train and test.

    Args:
        df:               optional already-loaded feature DataFrame.
        clinical_only:    if True, keep only the clinical features
                          (used by the clinical-only baseline).
    """
    if df is None:
        df = load_features()

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if clinical_only:
        feature_cols = [c for c in feature_cols if c in CLINICAL_ONLY_FEATURES]

    train_mask = df[GROUP_KEY] == "train"
    test_mask = ~train_mask

    return DataSplit(
        X_train=df.loc[train_mask, feature_cols].reset_index(drop=True),
        y_train=df.loc[train_mask, TARGET_COL].to_numpy().astype(int),
        groups_train=df.loc[train_mask, PATIENT_KEY].to_numpy(),
        X_test=df.loc[test_mask, feature_cols].reset_index(drop=True),
        y_test=df.loc[test_mask, TARGET_COL].to_numpy().astype(int),
        groups_test=df.loc[test_mask, PATIENT_KEY].to_numpy(),
        feature_names=feature_cols,
    )
