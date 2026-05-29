# EDA Findings — Phase 1 Discovery

_Generated automatically by `notebooks/01_data_discovery.ipynb` at 2026-05-25 11:25._

## 1. Grain & cohort
- Train: **1998 slices**, **55 unique patients**.
- Test:  **822 slices**, **19 unique patients**.
- Train↔test patient overlap: **0** (must be 0).
- Slices per patient (train): min=2, median=29, max=117.

## 2. Class balance (patient-level)
- Train: class_0=39, class_1=16.
- Test:  class_0=11, class_1=8.
- Inconsistent target patients: **0** (must be 0).

## 3. Missingness (slice-level, train)
- NIHSS: 198 (9.9%).
- ASPECTS: 169 (8.5%).
- Decision: keep rows; impute with median + add `*_was_missing` indicator in Phase 5.

## 4. Units & leakage check
- `Evolution Time` unit inferred: **MINUTES** (best cutoff = 271.21, agreement = 100.00%).
- Clinical cutoff (4.5 h) = 270 minutes.
- Decision: **drop `Evolution Time` from feature set** (it defines the target arithmetically).
- Decision: in staging, store both `evolution_minutes` (raw) and `evolution_hours` (derived) for clinical reporting only.

## 5. Demographic drift train ↔ test
- Sex chi²: p = 0.3445 (no drift).
- Age KS:   p = 0.1038 (no drift).
- Decision: report per-subgroup performance in Phase 6; do not stratify training by sex (would leak test priors).

## 6. Architectural implications confirmed
- **Small-N clinical cohort** (74 unique patients) — DL from scratch is off-limits in v1.
- **GroupKFold on `ID` is mandatory** for all CV.
- **Patient-level aggregation (MIL)** is the path for Phase 5.
- **Class 1 (>4.5h) is the clinically critical minority** — optimize sensitivity, not accuracy.

## 7. Figures produced
- `reports/figures/01_slices_per_patient.png`
- `reports/figures/02_leakage_evolution_time.png`
- `reports/figures/03_demographic_drift.png`
- `reports/figures/04_probe_features_by_class.png`

## 8. Next phase entry-conditions
- Data dictionary (`docs/data_dictionary.md`) reflects everything above.
- Ready to design DW (Phase 2): `raw` schema receives CSV as-is; `staging` renames the `Patien` typo and casts numeric columns.