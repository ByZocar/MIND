"""Canonical inventories shared across feature engineering modules.

Single source of truth for:
- Radiomic feature column names (66 features).
- Columns that are PROHIBITED in any feature set (leakage).
- Clinical columns used as features.

Importing from here (instead of redefining inside each module) keeps the
audit trail clean and means an anti-leakage test can simply diff a built
feature set against PROHIBITED_AS_FEATURE.
"""
from __future__ import annotations

SHAPE2D_COLS: tuple[str, ...] = (
    "shape2d_meshsurface",
    "shape2d_perimeter",
    "shape2d_sphericity",
    "shape2d_sphericaldisproportion",
    "shape2d_maximumdiameter",
    "shape2d_majoraxislength",
    "shape2d_minoraxislength",
    "shape2d_elongation",
)

FIRSTORDER_COLS: tuple[str, ...] = (
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
)

GLCM_COLS: tuple[str, ...] = (
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
)

GLSZM_COLS: tuple[str, ...] = (
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
)

ALL_RADIOMIC: tuple[str, ...] = SHAPE2D_COLS + FIRSTORDER_COLS + GLCM_COLS + GLSZM_COLS

CLINICAL_NUMERIC: tuple[str, ...] = ("age_years", "nihss", "aspects")

# Columns that derive directly from the target (or ARE the target) and
# must NEVER appear as features. The anti-leakage test asserts this set
# has zero intersection with any feature set.
PROHIBITED_AS_FEATURE: frozenset[str] = frozenset(
    {
        "evolution_minutes",
        "evolution_hours",
        "is_over_window",
    }
)

TARGET_COL: str = "is_over_window"
PATIENT_KEY: str = "patient_sk"
NATURAL_PATIENT_KEY: str = "patient_id"
GROUP_KEY: str = "dataset_origin"
