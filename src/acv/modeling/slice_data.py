"""Slice-level data access for Multi-Instance-Learning experiments.

Where the patient-level loader (`acv.modeling.data`) collapses each
patient's ~36 slices into a single feature vector via aggregations, the
SLICE-level loader treats each slice as an independent observation with
the patient's target inherited.

Why bother?
- N effective jumps from 55 train patients to ~1998 train slices.
- The model is no longer forced to summarize spatial information;
  it can find per-slice signatures that distinguish lesions in/out of
  the therapeutic window.
- The aggregation back to patient level happens AT INFERENCE only,
  via `acv.modeling.aggregate`. So the medical decision is still
  per-patient — we just sample evidence more densely.

Critical anti-leakage rules (same as patient-level):
- `evolution_minutes` / `evolution_hours` are NEVER loaded.
- CV uses GroupKFold with `groups = patient_sk`, so a patient's
  slices never split between train and val.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from acv.config import settings
from acv.features.inventory import (
    ALL_RADIOMIC,
    GROUP_KEY,
    PATIENT_KEY,
    PROHIBITED_AS_FEATURE,
    TARGET_COL,
)
from acv.io.db import get_engine


@dataclass
class SliceSplit:
    """Slice-level matrices with patient_sk groups for honest CV."""

    X_train: pd.DataFrame
    y_train: np.ndarray
    groups_train: np.ndarray
    slice_weights_train: np.ndarray  # for weighted aggregation later
    X_test: pd.DataFrame
    y_test: np.ndarray
    groups_test: np.ndarray
    slice_weights_test: np.ndarray
    feature_names: list[str]

    @property
    def n_train_patients(self) -> int:
        return int(np.unique(self.groups_train).size)

    @property
    def n_test_patients(self) -> int:
        return int(np.unique(self.groups_test).size)


SLICE_QUERY = """
    SELECT
        f.patient_sk,
        p.dataset_origin,
        p.sex,
        p.age_years,
        f.slice_order,
        f.shape2d_meshsurface,
        f.nihss,
        f.nihss_was_missing,
        f.aspects,
        f.aspects_was_missing,
        f.is_over_window,
        {radiomic_cols}
    FROM analytics.fct_slice f
    JOIN analytics.dim_patient p ON p.patient_sk = f.patient_sk
"""


def _radiomic_select() -> str:
    return ", ".join(f"f.{c}" for c in ALL_RADIOMIC)


def load_slice_frame() -> pd.DataFrame:
    sql = SLICE_QUERY.format(radiomic_cols=_radiomic_select())
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn)


def make_slice_split() -> SliceSplit:
    """Produce slice-level X/y/groups and a slice-weight vector."""
    df = load_slice_frame()

    # safety net: confirm no leakage columns slipped in.
    leak = set(df.columns) & (PROHIBITED_AS_FEATURE - {TARGET_COL})
    assert not leak, f"Leakage columns in slice loader: {leak}"

    df["sex_F"] = (df["sex"] == "F").astype(int)
    df["nihss_was_missing"] = df["nihss_was_missing"].fillna(False).astype(int)
    df["aspects_was_missing"] = df["aspects_was_missing"].fillna(False).astype(int)

    feature_cols = (
        list(ALL_RADIOMIC)
        + ["age_years", "sex_F", "nihss", "nihss_was_missing",
           "aspects", "aspects_was_missing"]
    )

    train_mask = df[GROUP_KEY] == "train"
    test_mask = ~train_mask

    # Slice "weight" for weighted aggregation later: lesion area in this
    # slice relative to that patient's max area (so the dominant slice
    # gets weight 1, smaller slices proportionally less).
    by_patient_max = df.groupby(PATIENT_KEY)["shape2d_meshsurface"].transform("max")
    weights = (df["shape2d_meshsurface"] / by_patient_max.clip(lower=1e-9)).to_numpy()

    return SliceSplit(
        X_train=df.loc[train_mask, feature_cols].reset_index(drop=True),
        y_train=df.loc[train_mask, TARGET_COL].to_numpy().astype(int),
        groups_train=df.loc[train_mask, PATIENT_KEY].to_numpy(),
        slice_weights_train=weights[train_mask.to_numpy()],
        X_test=df.loc[test_mask, feature_cols].reset_index(drop=True),
        y_test=df.loc[test_mask, TARGET_COL].to_numpy().astype(int),
        groups_test=df.loc[test_mask, PATIENT_KEY].to_numpy(),
        slice_weights_test=weights[test_mask.to_numpy()],
        feature_names=feature_cols,
    )
