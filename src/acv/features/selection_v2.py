"""Advanced feature selection with LASSO, PCA, and stability analysis.

This module implements rigorous feature selection as recommended in medical
radiomics literature, including:
- LASSO with nested cross-validation for λ selection
- PCA for dimensionality reduction (not selection)
- Stability analysis (bootstrap feature selection)
- mRMR (minimum Redundancy Maximum Relevance) approximation
- Contralateral difference features
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from scipy import stats

from acv.features.inventory import ALL_RADIOMIC

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Container for feature selection results."""
    selected_features: list[str]
    feature_scores: dict[str, float]
    method: str
    metadata: dict[str, Any]


class LassoSelector:
    """LASSO feature selection with nested CV for λ selection.
    
    Based on BMC Medical Imaging (2024) approach:
    - L1 regularization for sparse selection
    - 10-fold CV for λ optimization
    - Stability analysis via bootstrap
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        n_bootstrap: int = 100,
        stability_threshold: float = 0.8,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.n_folds = n_folds
        self.n_bootstrap = n_bootstrap
        self.stability_threshold = stability_threshold
        self.class_weight = class_weight
        self.random_state = random_state
        self.selected_features_: list[str] = []
        self.stability_scores_: dict[str, float] = {}
        self.lambda_optimal_: float | None = None
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series | None = None,
    ) -> "LassoSelector":
        """Fit LASSO with stability analysis.
        
        Args:
            X: Feature matrix (scaled)
            y: Target vector
            groups: Group labels for GroupKFold (e.g., patient_id)
        """
        feature_names = X.columns.tolist()
        X_array = X.values
        y_array = y.values
        
        # Outer CV for stability estimation
        if groups is not None:
            cv = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            split_iter = cv.split(X_array, y_array, groups.values)
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            split_iter = cv.split(X_array, y_array)
        
        # Store selections from each fold
        fold_selections: list[list[str]] = []
        fold_lambdas: list[float] = []
        
        for train_idx, _ in split_iter:
            X_train = X_array[train_idx]
            y_train = y_array[train_idx]
            
            # Inner CV for λ selection
            lasso_cv = LogisticRegressionCV(
                Cs=np.logspace(-4, 1, 50),
                cv=3,
                penalty="l1",
                solver="saga",
                class_weight=self.class_weight,
                scoring="roc_auc",
                max_iter=2000,
                random_state=self.random_state,
                tol=1e-4,
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lasso_cv.fit(X_train, y_train)
            
            # Get optimal C (inverse of λ)
            C_optimal = lasso_cv.C_[0]
            fold_lambdas.append(1.0 / C_optimal)
            
            # Fit final model with optimal C
            lasso_final = LogisticRegression(
                C=C_optimal,
                penalty="l1",
                solver="saga",
                class_weight=self.class_weight,
                max_iter=2000,
                random_state=self.random_state,
                tol=1e-4,
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lasso_final.fit(X_train, y_train)
            
            # Get selected features (non-zero coefficients)
            coefs = lasso_final.coef_[0]
            selected_idx = np.where(np.abs(coefs) > 1e-6)[0]
            selected = [feature_names[i] for i in selected_idx]
            fold_selections.append(selected)
        
        # Calculate stability scores
        feature_counts = {f: 0 for f in feature_names}
        for selected in fold_selections:
            for feat in selected:
                feature_counts[feat] += 1
        
        self.stability_scores_ = {
            f: count / self.n_folds 
            for f, count in feature_counts.items()
        }
        
        # Select features with stability >= threshold
        self.selected_features_ = [
            f for f, score in self.stability_scores_.items()
            if score >= self.stability_threshold
        ]
        
        self.lambda_optimal_ = np.median(fold_lambdas)
        
        logger.info(
            "LASSO selected %d features (stability >= %.2f)",
            len(self.selected_features_),
            self.stability_threshold
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return dataframe with selected features only."""
        return X[self.selected_features_]
    
    def get_stability_report(self) -> pd.DataFrame:
        """Generate stability report."""
        return pd.DataFrame([
            {"feature": f, "stability": s}
            for f, s in sorted(self.stability_scores_.items(), key=lambda x: -x[1])
            if s > 0
        ])


class PCAReducer:
    """PCA for dimensionality reduction with component interpretation.
    
    Unlike selection, PCA creates new features (components) that are
    linear combinations of original features. We track which original
    features contribute most to each component for interpretability.
    """
    
    def __init__(
        self,
        n_components: float | int = 0.95,
        interpret_top_n: int = 5,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.interpret_top_n = interpret_top_n
        self.random_state = random_state
        self.pca_: PCA | None = None
        self.component_interpretations_: dict[int, list[tuple[str, float]]] = {}
        
    def fit(self, X: pd.DataFrame) -> "PCAReducer":
        """Fit PCA and generate component interpretations."""
        self.feature_names_ = X.columns.tolist()
        
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca_.fit(X)
        
        # Interpret each component by top contributing features
        n_components = self.pca_.n_components_
        components = self.pca_.components_
        
        for i in range(n_components):
            # Get absolute loadings
            loadings = np.abs(components[i])
            top_idx = np.argsort(loadings)[-self.interpret_top_n:][::-1]
            
            self.component_interpretations_[i] = [
                (self.feature_names_[idx], components[i][idx])
                for idx in top_idx
            ]
        
        logger.info(
            "PCA reduced %d features to %d components (explained variance: %.3f)",
            len(self.feature_names_),
            n_components,
            np.sum(self.pca_.explained_variance_ratio_)
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform to principal components."""
        X_transformed = self.pca_.transform(X)
        
        # Create dataframe with meaningful column names
        n_components = X_transformed.shape[1]
        col_names = [f"PC{i+1}" for i in range(n_components)]
        
        return pd.DataFrame(X_transformed, index=X.index, columns=col_names)
    
    def get_interpretation_report(self) -> pd.DataFrame:
        """Generate component interpretation report."""
        rows = []
        for comp_idx, features in self.component_interpretations_.items():
            variance = self.pca_.explained_variance_ratio_[comp_idx]
            for feat_name, loading in features:
                rows.append({
                    "component": f"PC{comp_idx+1}",
                    "explained_variance_ratio": variance,
                    "feature": feat_name,
                    "loading": loading,
                })
        
        return pd.DataFrame(rows)


class MutualInformationSelector:
    """Mutual Information feature selection with redundancy pruning.
    
    Approximates mRMR (minimum Redundancy Maximum Relevance) by:
    1. Ranking features by MI with target (relevance)
    2. Pruning highly correlated features (redundancy)
    """
    
    def __init__(
        self,
        n_features: int | None = None,
        redundancy_threshold: float = 0.90,
        random_state: int = 42,
    ):
        self.n_features = n_features
        self.redundancy_threshold = redundancy_threshold
        self.random_state = random_state
        self.mi_scores_: dict[str, float] = {}
        self.selected_features_: list[str] = []
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "MutualInformationSelector":
        """Fit MI selector with redundancy pruning."""
        # Calculate MI scores
        mi_scores = mutual_info_classif(
            X, y, 
            random_state=self.random_state,
            n_neighbors=3  # Reduced for small samples
        )
        
        self.mi_scores_ = dict(zip(X.columns, mi_scores))
        
        # Rank by MI
        ranked = sorted(self.mi_scores_.items(), key=lambda x: -x[1])
        
        # Greedy selection: pick top MI, then skip if redundant with selected
        selected = []
        X_selected: pd.DataFrame | None = None
        
        for feat, score in ranked:
            if score <= 0:
                continue
                
            if X_selected is None:
                selected.append(feat)
                X_selected = X[[feat]].copy()
            else:
                # Check redundancy with already selected features
                redundant = False
                for col in X_selected.columns:
                    corr = np.abs(X[feat].corr(X_selected[col]))
                    if corr > self.redundancy_threshold:
                        redundant = True
                        break
                
                if not redundant:
                    selected.append(feat)
                    X_selected = pd.concat([X_selected, X[[feat]]], axis=1)
            
            # Stop if we have enough features
            if self.n_features and len(selected) >= self.n_features:
                break
        
        self.selected_features_ = selected
        
        logger.info(
            "MI selector: %d features from %d (redundancy threshold: %.2f)",
            len(selected),
            len(ranked),
            self.redundancy_threshold
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return dataframe with selected features."""
        return X[self.selected_features_]


class RadiomicFamilyAnalyzer:
    """Analyze predictive power by radiomic feature family.
    
    Compares Shape2D, FirstOrder, GLCM, GLSZM families to understand
    which types of features are most predictive.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.family_scores_: dict[str, dict] = {}
        
    def analyze(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Analyze each feature family separately.
        
        Returns dataframe with family performance metrics.
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        
        families = {
            "shape2d": [c for c in X.columns if c.startswith("shape2d_")],
            "firstorder": [c for c in X.columns if c.startswith("firstorder_")],
            "glcm": [c for c in X.columns if c.startswith("glcm_")],
            "glszm": [c for c in X.columns if c.startswith("glszm_")],
        }
        
        results = []
        
        for family_name, cols in families.items():
            if not cols:
                continue
                
            X_family = X[cols]
            
            # Simple logistic regression with CV
            model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
                solver="lbfgs",
            )
            
            if groups is not None:
                cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(model, X_family, y, cv=cv, groups=groups, scoring="roc_auc")
            else:
                from sklearn.model_selection import StratifiedKFold
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(model, X_family, y, cv=cv, scoring="roc_auc")
            
            # Also get MI scores
            mi_scores = mutual_info_classif(X_family, y, random_state=self.random_state)
            
            results.append({
                "family": family_name,
                "n_features": len(cols),
                "mean_auroc": scores.mean(),
                "std_auroc": scores.std(),
                "mean_mi": mi_scores.mean(),
                "max_mi": mi_scores.max(),
            })
            
            self.family_scores_[family_name] = {
                "auroc_scores": scores,
                "mi_scores": mi_scores,
                "features": cols,
            }
        
        return pd.DataFrame(results).sort_values("mean_auroc", ascending=False)


def add_contralateral_features(
    df: pd.DataFrame,
    patient_col: str = "patient_sk",
    side_col: str | None = None,
) -> pd.DataFrame:
    """Add contralateral difference features.
    
    In stroke, comparing lesion side with healthy side is crucial.
    This function creates features representing (lesion - contralateral)/contralateral.
    
    Note: This requires knowing which side is the lesion. If side_col is not
    provided, assumes data already represents lesion regions only.
    
    For our dataset with per-slice features, we approximate by comparing
    each slice to the patient's mean (as reference).
    
    Args:
        df: DataFrame with radiomic features
        patient_col: Column identifying patients
        side_col: Optional column indicating lesion side (L/R)
    
    Returns:
        DataFrame with added contralateral difference features
    """
    radiomic_cols = [c for c in df.columns if c in ALL_RADIOMIC]
    
    # Strategy: For each patient, calculate difference from patient mean
    # This approximates "deviation from normal" for each slice
    
    df_result = df.copy()
    
    for col in radiomic_cols:
        # Calculate patient mean for this feature
        patient_mean = df.groupby(patient_col)[col].transform("mean")
        
        # Relative difference (avoid division by zero)
        diff_col = f"{col}_diff_from_patient_mean"
        df_result[diff_col] = (df[col] - patient_mean) / (patient_mean.abs() + 1e-8)
        
        # Absolute difference
        abs_diff_col = f"{col}_abs_diff_from_patient_mean"
        df_result[abs_diff_col] = np.abs(df[col] - patient_mean)
    
    logger.info(
        "Added %d contralateral difference features",
        len(radiomic_cols) * 2
    )
    
    return df_result


def select_features_comprehensive(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series | None = None,
    methods: list[str] | None = None,
    lasso_threshold: float = 0.8,
    pca_components: float = 0.95,
) -> dict[str, SelectionResult]:
    """Run comprehensive feature selection with multiple methods.
    
    Args:
        X_train: Training features
        y_train: Training labels
        groups_train: Group labels for CV
        methods: List of methods to run ("lasso", "pca", "mi")
        lasso_threshold: Stability threshold for LASSO
        pca_components: Variance to retain in PCA
    
    Returns:
        Dictionary of method name -> SelectionResult
    """
    methods = methods or ["lasso", "pca", "mi"]
    results = {}
    
    # Standardize for methods that need it
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    if "lasso" in methods:
        logger.info("Running LASSO selection...")
        lasso = LassoSelector(
            n_folds=5,
            stability_threshold=lasso_threshold,
            random_state=42,
        )
        lasso.fit(X_scaled, y_train, groups_train)
        
        results["lasso"] = SelectionResult(
            selected_features=lasso.selected_features_,
            feature_scores=lasso.stability_scores_,
            method="LASSO with stability analysis",
            metadata={
                "lambda_optimal": lasso.lambda_optimal_,
                "stability_threshold": lasso.stability_threshold,
            }
        )
    
    if "pca" in methods:
        logger.info("Running PCA reduction...")
        pca = PCAReducer(n_components=pca_components)
        pca.fit(X_scaled)
        
        # PCA doesn't select features but creates components
        # We track original features that contribute most
        top_features = []
        for comp_features in pca.component_interpretations_.values():
            for feat, _ in comp_features:
                if feat not in top_features:
                    top_features.append(feat)
        
        results["pca"] = SelectionResult(
            selected_features=top_features[:20],  # Top contributors
            feature_scores={
                f"PC{i+1}": pca.pca_.explained_variance_ratio_[i]
                for i in range(pca.pca_.n_components_)
            },
            method="PCA (dimensionality reduction)",
            metadata={
                "n_components": pca.pca_.n_components_,
                "explained_variance_ratio_sum": float(np.sum(pca.pca_.explained_variance_ratio_)),
                "component_interpretations": pca.component_interpretations_,
            }
        )
    
    if "mi" in methods:
        logger.info("Running Mutual Information selection...")
        mi = MutualInformationSelector(
            n_features=20,
            redundancy_threshold=0.90,
        )
        mi.fit(X_train, y_train)
        
        results["mi"] = SelectionResult(
            selected_features=mi.selected_features_,
            feature_scores=mi.mi_scores_,
            method="Mutual Information with redundancy pruning",
            metadata={
                "redundancy_threshold": mi.redundancy_threshold,
            }
        )
    
    return results
