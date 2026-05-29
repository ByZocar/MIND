"""Patient-level feature engineering for the ACV window classifier.

Inputs come from `analytics.fct_slice` (slice grain) and
`analytics.dim_patient`. Output is a *wide* DataFrame at patient grain:

    one row per patient, ~660 columns:
      - identifiers: patient_sk, patient_id, dataset_origin, is_over_window
      - n_slices
      - clinical: age_years, sex_F, nihss, nihss_was_missing, aspects, aspects_was_missing
      - 66 radiomic features × 9 aggregations  = 594 columns
      - 66 radiomic features from the DOMINANT slice (largest MeshSurface)
                                                  = 66 columns
      Total: ~668 columns

The wide matrix is intentionally over-complete; Phase 5's selection step
prunes it (zero-variance, near-duplicates, low-MI).

Anti-leakage invariants enforced here:
- `evolution_minutes` and `evolution_hours` are stripped from the slice
  set before aggregation.
- `is_over_window` is preserved only as the target column, never as a
  feature.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

# scipy emits RuntimeWarnings for skew/kurt on near-constant series.
# This is expected for radiomic features with low variance across slices
# (e.g. a small lesion). Suppress to keep the build log clean — the
# precision-loss case still returns 0.0 which is the correct value.
warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation*")

from acv.features.inventory import (
    ALL_RADIOMIC,
    CLINICAL_NUMERIC,
    PROHIBITED_AS_FEATURE,
    TARGET_COL,
)
from acv.io.db import get_engine

AGG_FUNCS: dict[str, callable] = {
    "mean": np.mean,
    "std": np.std,
    "min": np.min,
    "max": np.max,
    "p10": lambda x: np.percentile(x, 10),
    "p50": lambda x: np.percentile(x, 50),
    "p90": lambda x: np.percentile(x, 90),
    "skew": lambda x: float(stats.skew(x, bias=False)) if len(x) >= 3 else 0.0,
    "kurt": lambda x: float(stats.kurtosis(x, bias=False)) if len(x) >= 4 else 0.0,
}


# ---------------------------------------------------------------------------
# data load
# ---------------------------------------------------------------------------


def load_slice_frame() -> pd.DataFrame:
    """Pull `analytics.fct_slice` joined with `dim_patient` from Postgres."""
    sql = """
        SELECT
            f.patient_sk,
            p.patient_id,
            p.dataset_origin,
            p.sex,
            p.age_years,
            f.slice_order,
            f.shape2d_meshsurface, f.shape2d_perimeter, f.shape2d_sphericity,
            f.shape2d_sphericaldisproportion, f.shape2d_maximumdiameter,
            f.shape2d_majoraxislength, f.shape2d_minoraxislength, f.shape2d_elongation,
            f.firstorder_entropy, f.firstorder_minimum, f.firstorder_10percentile,
            f.firstorder_90percentile, f.firstorder_maximum, f.firstorder_mean,
            f.firstorder_median, f.firstorder_interquartilerange, f.firstorder_range,
            f.firstorder_meanabsolutedeviation, f.firstorder_robustmeanabsolutedeviation,
            f.firstorder_rootmeansquared, f.firstorder_standarddeviation,
            f.firstorder_skewness, f.firstorder_kurtosis, f.firstorder_variance,
            f.firstorder_uniformity,
            f.glcm_autocorrelation, f.glcm_jointaverage, f.glcm_clusterprominence,
            f.glcm_clustershade, f.glcm_clustertendency, f.glcm_contrast,
            f.glcm_correlation, f.glcm_differenceaverage, f.glcm_differenceentropy,
            f.glcm_differencevariance, f.glcm_jointenergy, f.glcm_jointentropy,
            f.glcm_imc1, f.glcm_imc2, f.glcm_idm, f.glcm_mcc, f.glcm_idmn,
            f.glcm_id, f.glcm_idn, f.glcm_maximumprobability, f.glcm_sumaverage,
            f.glcm_sumentropy, f.glcm_sumsquares,
            f.glszm_smallareaemphasis, f.glszm_largeareaemphasis,
            f.glszm_graylevelnonuniformity, f.glszm_graylevelnonuniformitynormalized,
            f.glszm_sizezonenonuniformity, f.glszm_sizezonenonuniformitynormalized,
            f.glszm_zonepercentage, f.glszm_graylevelvariance, f.glszm_zonevariance,
            f.glszm_zoneentropy, f.glszm_lowgraylevelzoneemphasis,
            f.glszm_highgraylevelzoneemphasis, f.glszm_smallarealowgraylevelemphasis,
            f.glszm_smallareahighgraylevelemphasis, f.glszm_largearealowgraylevelemphasis,
            f.glszm_largeareahighgraylevelemphasis,
            f.nihss, f.nihss_was_missing,
            f.aspects, f.aspects_was_missing,
            f.is_over_window
        FROM analytics.fct_slice f
        JOIN analytics.dim_patient p ON p.patient_sk = f.patient_sk
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _aggregate_radiomics(slices: pd.DataFrame) -> pd.DataFrame:
    """Patient-level mean/std/min/max/p10/p50/p90/skew/kurt for each radiomic feature."""
    grouped = slices.groupby("patient_sk")
    out = {}
    for col in ALL_RADIOMIC:
        s = grouped[col]
        for agg_name, fn in AGG_FUNCS.items():
            try:
                out[f"{col}__{agg_name}"] = s.apply(lambda x, fn=fn: fn(x.dropna().to_numpy()))
            except Exception:
                out[f"{col}__{agg_name}"] = grouped[col].mean()
    return pd.DataFrame(out)


def _dominant_slice_features(slices: pd.DataFrame) -> pd.DataFrame:
    """Snapshot of the slice with the largest 2D lesion area, per patient."""
    idx = slices.groupby("patient_sk")["shape2d_meshsurface"].idxmax()
    dom = slices.loc[idx, ["patient_sk", *ALL_RADIOMIC]].set_index("patient_sk")
    dom.columns = [f"{c}__dominant" for c in dom.columns]
    return dom


def _patient_clinical(slices: pd.DataFrame) -> pd.DataFrame:
    """Clinical & demographic columns are constant per patient -> first()."""
    first = (
        slices.groupby("patient_sk")[
            ["patient_id", "dataset_origin", "sex", "age_years", "nihss",
             "nihss_was_missing", "aspects", "aspects_was_missing", "is_over_window"]
        ].first()
    )
    first["n_slices"] = slices.groupby("patient_sk").size()
    first["sex_F"] = (first["sex"] == "F").astype(int)
    # Cast missing-indicator booleans to int so the persistence layer
    # (which uses INTEGER DDL) accepts them via psycopg2.
    for c in ("nihss_was_missing", "aspects_was_missing"):
        first[c] = first[c].fillna(False).astype(int)
    first = first.drop(columns=["sex"])
    return first


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def build_features_v1() -> pd.DataFrame:
    """Build the patient-level wide feature matrix.

    Returns a DataFrame indexed by patient_sk with:
      - id/group columns: patient_id, dataset_origin
      - target column:    is_over_window  (NOT a feature, but persisted)
      - everything else:  numeric features.
    """
    slices = load_slice_frame()

    # Strip leakage columns up front. evolution_* never leaves this function.
    leakage_present = [c for c in slices.columns if c in PROHIBITED_AS_FEATURE - {TARGET_COL}]
    slices = slices.drop(columns=leakage_present, errors="ignore")

    radiomic_agg = _aggregate_radiomics(slices)
    dominant = _dominant_slice_features(slices)
    clinical = _patient_clinical(slices)

    features = clinical.join(radiomic_agg, how="left").join(dominant, how="left")

    # Defensive: enforce the leakage contract on the FINAL frame.
    leak = set(features.columns) & (PROHIBITED_AS_FEATURE - {TARGET_COL})
    if leak:
        raise RuntimeError(f"Leakage columns leaked into feature set: {leak}")

    return features.reset_index()
