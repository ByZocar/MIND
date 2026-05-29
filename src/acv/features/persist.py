"""Persist the patient-level feature matrix to Postgres and parquet.

The wide table (~660 columns) is created with a programmatically-generated
DDL so we never hand-maintain that many column definitions. It lives in
the analytics schema so the rest of the platform (KPIs, dashboard, future
ad-hoc analyses) can query it consistently with the rest of the DW.

The parquet is the fast path for ML in Phase 6.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console
from sqlalchemy import text

from acv.config import settings
from acv.features.inventory import (
    GROUP_KEY,
    NATURAL_PATIENT_KEY,
    PATIENT_KEY,
    TARGET_COL,
)
from acv.io.db import get_engine

console = Console()

PARQUET_PATH = settings.project_root / "data" / "processed" / "features_v1.parquet"
PARQUET_RANK_PATH = settings.project_root / "data" / "processed" / "features_v1_mi_rank.parquet"


def _safe_pg_col(name: str) -> str:
    """Sanitize a feature name for Postgres: lowercase, only [a-z0-9_]."""
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower())
    if safe[0].isdigit():
        safe = "_" + safe
    return safe


def _build_ddl(df: pd.DataFrame, table_full: str) -> str:
    cols_sql: list[str] = [
        f"    {PATIENT_KEY}            BIGINT       PRIMARY KEY",
        f"    {NATURAL_PATIENT_KEY}    TEXT         NOT NULL",
        f"    {GROUP_KEY}              TEXT         NOT NULL",
        f"    {TARGET_COL}             SMALLINT     NOT NULL",
    ]
    reserved = {PATIENT_KEY, NATURAL_PATIENT_KEY, GROUP_KEY, TARGET_COL}
    integer_like = {"n_slices", "age_years", "nihss_was_missing", "aspects_was_missing", "sex_F"}
    for col in df.columns:
        if col in reserved:
            continue
        pg = _safe_pg_col(col)
        sql_type = "INTEGER" if col in integer_like else "DOUBLE PRECISION"
        cols_sql.append(f"    {pg:60s}{sql_type}")
    return f"DROP TABLE IF EXISTS {table_full} CASCADE;\nCREATE TABLE {table_full} (\n" + ",\n".join(cols_sql) + "\n);"


def save_to_postgres(df: pd.DataFrame, schema: str = "analytics", table: str = "features_v1") -> int:
    """Recreate the table and bulk-insert. Returns row count."""
    full = f"{schema}.{table}"
    engine = get_engine()
    ddl = _build_ddl(df, full)
    with engine.begin() as conn:
        conn.exec_driver_sql(ddl)
        conn.exec_driver_sql(f"CREATE INDEX IF NOT EXISTS ix_{table}_origin ON {full} ({GROUP_KEY});")
        conn.exec_driver_sql(f"CREATE INDEX IF NOT EXISTS ix_{table}_target ON {full} ({TARGET_COL});")
    # Rename df columns to sanitized form
    rename = {c: _safe_pg_col(c) for c in df.columns}
    df_pg = df.rename(columns=rename)
    df_pg.to_sql(
        table,
        engine,
        schema=schema,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=200,
    )
    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {full}")).scalar_one()
    return int(n)


def save_to_parquet(df: pd.DataFrame, path: Path = PARQUET_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def save_mi_rank(rank: pd.DataFrame, path: Path = PARQUET_RANK_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rank.to_parquet(path, index=False)
    return path
