"""Quality gate runner.

Loads `staging.fct_slice_stg` into a pandas DataFrame, runs the suites
defined in `acv.quality.suites`, and persists results.

Design choices:
- We use Great Expectations *programmatically* via `PandasDataset` (V2
  API) so the runner has zero dependence on the GE DataContext layout
  on disk. This keeps things reproducible and CI-friendly.
- A failure is HARD-FAILED. The DAG halts and analytics is NOT refreshed.
- Each run logs a JSON report under `reports/ge/<load_id>__<suite>.json`
  so the medical team can audit any rejection later.

For the "patient_integrity" suite we manually evaluate the cross-row
expectations against the DataFrame because the v2 PandasDataset doesn't
have a `column_pair_to_be_greater_than_scalar` expectation out of the
box — we encode it directly with pandas. This keeps the API surface
small while still capturing the rule in a versioned suite.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from sqlalchemy import text

from acv.config import settings
from acv.io.db import get_engine
from acv.quality.suites import SUITES

console = Console()

REPORTS_DIR = settings.project_root / "reports" / "ge"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# data models
# ---------------------------------------------------------------------------


@dataclass
class ExpectationResult:
    expectation_type: str
    column: str | None
    success: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SuiteResult:
    suite_name: str
    success: bool
    n_rows: int
    n_expectations: int
    n_failed: int
    results: list[ExpectationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "success": self.success,
            "n_rows": self.n_rows,
            "n_expectations": self.n_expectations,
            "n_failed": self.n_failed,
            "results": [
                {
                    "expectation_type": r.expectation_type,
                    "column": r.column,
                    "success": r.success,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------


def load_staging() -> pd.DataFrame:
    sql = "SELECT * FROM staging.fct_slice_stg"
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn)


# ---------------------------------------------------------------------------
# expectation evaluators
# ---------------------------------------------------------------------------


def _expect_column_values_to_not_be_null(df: pd.DataFrame, column: str) -> ExpectationResult:
    n_null = int(df[column].isna().sum())
    return ExpectationResult(
        expectation_type="expect_column_values_to_not_be_null",
        column=column,
        success=n_null == 0,
        details={"n_null": n_null, "n_total": len(df)},
    )


def _expect_column_values_to_be_between(
    df: pd.DataFrame,
    column: str,
    min_value: float | None = None,
    max_value: float | None = None,
    mostly: float = 1.0,
) -> ExpectationResult:
    s = df[column]
    s_nn = s.dropna()
    mask = pd.Series(True, index=s_nn.index)
    if min_value is not None:
        mask &= s_nn >= min_value
    if max_value is not None:
        mask &= s_nn <= max_value
    rate = float(mask.mean()) if len(s_nn) else 1.0
    return ExpectationResult(
        expectation_type="expect_column_values_to_be_between",
        column=column,
        success=rate >= mostly,
        details={
            "min_value": min_value,
            "max_value": max_value,
            "mostly": mostly,
            "pass_rate": round(rate, 4),
            "n_violations": int((~mask).sum()),
        },
    )


def _expect_column_values_to_be_in_set(
    df: pd.DataFrame, column: str, value_set: list[Any]
) -> ExpectationResult:
    s = df[column].dropna()
    if s.empty:
        return ExpectationResult(
            expectation_type="expect_column_values_to_be_in_set",
            column=column,
            success=True,
            details={"value_set": value_set, "n_violations": 0},
        )
    violations = int((~s.isin(value_set)).sum())
    return ExpectationResult(
        expectation_type="expect_column_values_to_be_in_set",
        column=column,
        success=violations == 0,
        details={"value_set": value_set, "n_violations": violations},
    )


def _expect_table_columns_to_match_set(
    df: pd.DataFrame, column_set: list[str]
) -> ExpectationResult:
    missing = sorted(set(column_set) - set(df.columns))
    return ExpectationResult(
        expectation_type="expect_table_columns_to_match_set",
        column=None,
        success=len(missing) == 0,
        details={"missing_columns": missing},
    )


def _expect_table_row_count_to_be_between(
    df: pd.DataFrame, min_value: int, max_value: int
) -> ExpectationResult:
    n = len(df)
    return ExpectationResult(
        expectation_type="expect_table_row_count_to_be_between",
        column=None,
        success=min_value <= n <= max_value,
        details={"min_value": min_value, "max_value": max_value, "actual": n},
    )


def _expect_compound_columns_to_be_unique(
    df: pd.DataFrame, column_list: list[str]
) -> ExpectationResult:
    dup = df.duplicated(subset=column_list).sum()
    return ExpectationResult(
        expectation_type="expect_compound_columns_to_be_unique",
        column=None,
        success=int(dup) == 0,
        details={"columns": column_list, "n_duplicates": int(dup)},
    )


def _expect_column_pair_values_A_to_be_greater_than_B(
    df: pd.DataFrame, column_A: str, column_B: Any, or_equal: bool, ignore_row_if: str
) -> ExpectationResult:
    """Adapted to also handle scalar B (used for the >270-minutes target rule).

    For the target-mapping check: we verify that
       (evolution_minutes > 270)::int == is_over_window
    """
    if column_A == "evolution_minutes" and column_B == 270:
        evo = df["evolution_minutes"]
        target = df["is_over_window"]
        keep = evo.notna() & target.notna()
        if not keep.any():
            return ExpectationResult(
                expectation_type="expect_column_pair_values_A_to_be_greater_than_B",
                column=column_A,
                success=False,
                details={"reason": "no usable rows", "n_rows": 0},
            )
        derived = (evo[keep] > 270).astype(int)
        agree = float((derived == target[keep]).mean())
        return ExpectationResult(
            expectation_type="expect_column_pair_values_A_to_be_greater_than_B",
            column=column_A,
            success=agree >= 1.0,
            details={"rule": "evolution_minutes > 270 == is_over_window", "agreement": round(agree, 6)},
        )
    return ExpectationResult(
        expectation_type="expect_column_pair_values_A_to_be_greater_than_B",
        column=column_A,
        success=False,
        details={"reason": "unsupported pair", "column_A": column_A, "column_B": column_B},
    )


EVALUATORS = {
    "expect_column_values_to_not_be_null": _expect_column_values_to_not_be_null,
    "expect_column_values_to_be_between": _expect_column_values_to_be_between,
    "expect_column_values_to_be_in_set": _expect_column_values_to_be_in_set,
    "expect_table_columns_to_match_set": _expect_table_columns_to_match_set,
    "expect_table_row_count_to_be_between": _expect_table_row_count_to_be_between,
    "expect_compound_columns_to_be_unique": _expect_compound_columns_to_be_unique,
    "expect_column_pair_values_A_to_be_greater_than_B": _expect_column_pair_values_A_to_be_greater_than_B,
}


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_suite(df: pd.DataFrame, suite_name: str) -> SuiteResult:
    if suite_name not in SUITES:
        raise KeyError(f"Unknown suite: {suite_name}")
    suite = SUITES[suite_name]()
    results: list[ExpectationResult] = []
    for ec in suite.expectations:
        kwargs = dict(ec.kwargs)
        evaluator = EVALUATORS.get(ec.expectation_type)
        if evaluator is None:
            results.append(
                ExpectationResult(
                    expectation_type=ec.expectation_type,
                    column=kwargs.get("column"),
                    success=False,
                    details={"reason": "evaluator not implemented"},
                )
            )
            continue
        results.append(evaluator(df, **kwargs))

    n_failed = sum(1 for r in results if not r.success)
    return SuiteResult(
        suite_name=suite_name,
        success=n_failed == 0,
        n_rows=len(df),
        n_expectations=len(results),
        n_failed=n_failed,
        results=results,
    )


def run_all(load_id: int | None = None) -> dict[str, SuiteResult]:
    df = load_staging()
    out: dict[str, SuiteResult] = {}
    ts = int(time.time() * 1000)
    label = str(load_id or ts)
    for name in SUITES:
        result = run_suite(df, name)
        out[name] = result
        path = REPORTS_DIR / f"{label}__{name}.json"
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return out


def hard_gate(load_id: int | None = None) -> dict[str, SuiteResult]:
    """Run all suites; raise if any suite failed.

    The DAG calls this between staging and analytics; if it raises,
    the analytics tasks are skipped and the run goes red.
    """
    results = run_all(load_id=load_id)
    failed = {n: r for n, r in results.items() if not r.success}
    for name, r in results.items():
        flag = "OK " if r.success else "FAIL"
        console.print(f"  [{flag}] {name:22s} expectations={r.n_expectations} failed={r.n_failed}")
        if not r.success:
            for er in r.results:
                if not er.success:
                    console.print(
                        f"      - {er.expectation_type} col={er.column} details={er.details}"
                    )
    if failed:
        raise RuntimeError(
            f"Quality gate failed: {list(failed)}. Analytics will NOT be refreshed."
        )
    return results


if __name__ == "__main__":
    hard_gate()
