"""Quality-gate suites for the ACV staging layer.

Library-agnostic on purpose: each suite is a list of `Expectation`
dataclasses that mirror the shape of Great Expectations'
`ExpectationConfiguration` objects, but without importing GE.

Why?
- GE 0.x and GE 1.x have incompatible APIs.
- Inside the Airflow container Airflow's constraints decide which GE
  version we get, which historically caused import-time failures
  (Python 3.12 ForwardRef compatibility).
- Our use case is a simple, deterministic set of ~12 expectations that
  the runner in `acv.quality.gate` evaluates against a pandas DataFrame.
  We don't need GE's DataContext, Stores, Checkpoints, or Data Docs.

The shape (`expectation_type`, `kwargs`) is *intentionally identical*
to GE's so that a future v2 can swap in the real GE library without
modifying the suite definitions or the persisted JSON reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Column inventories (kept here as the single source of truth)
# ---------------------------------------------------------------------------

SHAPE2D_COLS = [
    "shape2d_meshsurface",
    "shape2d_perimeter",
    "shape2d_sphericity",
    "shape2d_sphericaldisproportion",
    "shape2d_maximumdiameter",
    "shape2d_majoraxislength",
    "shape2d_minoraxislength",
    "shape2d_elongation",
]

FIRSTORDER_COLS = [
    "firstorder_entropy",
    "firstorder_minimum",
    "firstorder_10percentile",
    "firstorder_90percentile",
    "firstorder_maximum",
    "firstorder_mean",
    "firstorder_median",
    "firstorder_interquartilerange",
    "firstorder_range",
    "firstorder_meanabsolutedeviation",
    "firstorder_robustmeanabsolutedeviation",
    "firstorder_rootmeansquared",
    "firstorder_standarddeviation",
    "firstorder_skewness",
    "firstorder_kurtosis",
    "firstorder_variance",
    "firstorder_uniformity",
]

GLCM_COLS = [
    "glcm_autocorrelation",
    "glcm_jointaverage",
    "glcm_clusterprominence",
    "glcm_clustershade",
    "glcm_clustertendency",
    "glcm_contrast",
    "glcm_correlation",
    "glcm_differenceaverage",
    "glcm_differenceentropy",
    "glcm_differencevariance",
    "glcm_jointenergy",
    "glcm_jointentropy",
    "glcm_imc1",
    "glcm_imc2",
    "glcm_idm",
    "glcm_mcc",
    "glcm_idmn",
    "glcm_id",
    "glcm_idn",
    "glcm_maximumprobability",
    "glcm_sumaverage",
    "glcm_sumentropy",
    "glcm_sumsquares",
]

GLSZM_COLS = [
    "glszm_smallareaemphasis",
    "glszm_largeareaemphasis",
    "glszm_graylevelnonuniformity",
    "glszm_graylevelnonuniformitynormalized",
    "glszm_sizezonenonuniformity",
    "glszm_sizezonenonuniformitynormalized",
    "glszm_zonepercentage",
    "glszm_graylevelvariance",
    "glszm_zonevariance",
    "glszm_zoneentropy",
    "glszm_lowgraylevelzoneemphasis",
    "glszm_highgraylevelzoneemphasis",
    "glszm_smallarealowgraylevelemphasis",
    "glszm_smallareahighgraylevelemphasis",
    "glszm_largearealowgraylevelemphasis",
    "glszm_largeareahighgraylevelemphasis",
]

ALL_RADIOMIC = SHAPE2D_COLS + FIRSTORDER_COLS + GLCM_COLS + GLSZM_COLS


# ---------------------------------------------------------------------------
# Expectation dataclass — GE-shaped, library-free
# ---------------------------------------------------------------------------


@dataclass
class Expectation:
    expectation_type: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Suite:
    name: str
    expectations: list[Expectation]


def _ec(expectation_type: str, **kwargs: Any) -> Expectation:
    return Expectation(expectation_type=expectation_type, kwargs=kwargs)


# ---------------------------------------------------------------------------
# Suite 1: slice_schema
# ---------------------------------------------------------------------------


def slice_schema_suite() -> Suite:
    expectations: list[Expectation] = []

    structural_cols = [
        "patient_id",
        "slice_order",
        "dataset_origin",
        *ALL_RADIOMIC,
        "age_years",
        "sex",
        "nihss",
        "nihss_was_missing",
        "aspects",
        "aspects_was_missing",
        "evolution_minutes",
        "evolution_hours",
        "is_over_window",
    ]
    expectations.append(_ec("expect_table_columns_to_match_set", column_set=structural_cols))

    for col in ("patient_id", "slice_order", "dataset_origin", "is_over_window"):
        expectations.append(_ec("expect_column_values_to_not_be_null", column=col))

    for col in (
        "shape2d_meshsurface",
        "shape2d_perimeter",
        "shape2d_maximumdiameter",
        "shape2d_majoraxislength",
        "shape2d_minoraxislength",
        "firstorder_range",
        "firstorder_variance",
        "firstorder_uniformity",
        "glcm_jointenergy",
        "glcm_maximumprobability",
        "glszm_zonepercentage",
    ):
        expectations.append(_ec("expect_column_values_to_be_between", column=col, min_value=0))

    for col in ("shape2d_sphericity", "shape2d_elongation"):
        expectations.append(
            _ec("expect_column_values_to_be_between", column=col, min_value=0, max_value=1.0001)
        )

    expectations.append(
        _ec("expect_column_values_to_be_in_set", column="dataset_origin", value_set=["train", "test"])
    )

    return Suite(name="slice_schema", expectations=expectations)


# ---------------------------------------------------------------------------
# Suite 2: slice_clinical
# ---------------------------------------------------------------------------


def slice_clinical_suite() -> Suite:
    expectations = [
        _ec("expect_column_values_to_be_between", column="nihss", min_value=0, max_value=42, mostly=1.0),
        _ec("expect_column_values_to_be_between", column="aspects", min_value=0, max_value=10, mostly=1.0),
        _ec("expect_column_values_to_be_between", column="age_years", min_value=0, max_value=120),
        _ec("expect_column_values_to_be_in_set", column="sex", value_set=["M", "F"]),
        _ec("expect_column_values_to_be_in_set", column="is_over_window", value_set=[0, 1]),
        _ec("expect_column_values_to_be_between", column="evolution_minutes", min_value=0),
    ]
    return Suite(name="slice_clinical", expectations=expectations)


# ---------------------------------------------------------------------------
# Suite 3: patient_integrity
# ---------------------------------------------------------------------------


def patient_integrity_suite() -> Suite:
    expectations = [
        _ec(
            "expect_column_pair_values_A_to_be_greater_than_B",
            column_A="evolution_minutes",
            column_B=270,
            or_equal=False,
            ignore_row_if="either_value_is_missing",
        ),
        _ec("expect_table_row_count_to_be_between", min_value=2820, max_value=2820),
        _ec(
            "expect_compound_columns_to_be_unique",
            column_list=["dataset_origin", "patient_id", "slice_order"],
        ),
    ]
    return Suite(name="patient_integrity", expectations=expectations)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


SUITES: dict[str, Callable[[], Suite]] = {
    "slice_schema": slice_schema_suite,
    "slice_clinical": slice_clinical_suite,
    "patient_integrity": patient_integrity_suite,
}
