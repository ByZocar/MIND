"""Load raw CSVs into Postgres and rebuild staging + analytics layers.

CLI:
    python -m acv.io.load_csv_to_raw --apply-ddl --reset

Phases performed (in order):
    1. apply_ddl    : execute every SQL file in sql/ (idempotent).
    2. reset (opt)  : truncate raw.acv_slice_csv + downstream tables.
    3. load_raw     : insert both CSVs into raw with a single new load_id.
    4. rebuild_stg  : TRUNCATE staging.fct_slice_stg and INSERT typed rows
                      from the latest load_id.
    5. rebuild_dw   : populate analytics.dim_patient, dim_severity, fct_slice
                      and REFRESH MATERIALIZED VIEW analytics.agg_patient.

The whole pipeline is idempotent: re-running fully recreates analytics
from raw. raw itself is append-only by load_id.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from sqlalchemy import text

from acv.config import settings
from acv.io.db import get_engine

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=False)

SQL_DIR = settings.project_root / "sql"
RAW_DIR = settings.project_root / "data" / "raw"

CSV_FILES = [
    ("train", RAW_DIR / "acv_clasif_train.csv"),
    ("test", RAW_DIR / "acv_clasif_test.csv"),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _new_load_id() -> int:
    return int(time.time() * 1000)


def _read_csv(path: Path, origin: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.replace({"": None})
    df.insert(0, "row_order", range(1, len(df) + 1))
    df.insert(0, "dataset_origin", origin)
    return df


def _apply_sql_file(conn, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    conn.exec_driver_sql(sql)


# ---------------------------------------------------------------------------
# phase 1: apply DDL
# ---------------------------------------------------------------------------


def apply_ddl() -> None:
    files = sorted(SQL_DIR.glob("*.sql"))
    if not files:
        raise RuntimeError(f"No SQL files found under {SQL_DIR}")
    console.print(f"[cyan]==> Applying {len(files)} SQL files[/]")
    engine = get_engine()
    with engine.begin() as conn:
        for f in files:
            console.print(f"   [dim]- {f.name}[/]")
            _apply_sql_file(conn, f)
    console.print("[green]   DDL applied.[/]")


# ---------------------------------------------------------------------------
# phase 2: reset
# ---------------------------------------------------------------------------


def reset_all() -> None:
    console.print("[cyan]==> Resetting raw + staging + analytics[/]")
    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql("TRUNCATE analytics.fct_slice CASCADE")
        conn.exec_driver_sql("TRUNCATE analytics.dim_patient CASCADE")
        conn.exec_driver_sql("TRUNCATE analytics.dim_severity CASCADE")
        conn.exec_driver_sql("TRUNCATE staging.fct_slice_stg CASCADE")
        conn.exec_driver_sql("TRUNCATE staging.quarantine_slice CASCADE")
        conn.exec_driver_sql("TRUNCATE raw.acv_slice_csv")
        conn.exec_driver_sql("TRUNCATE raw.load_catalog")
    console.print("[green]   Reset done.[/]")


# ---------------------------------------------------------------------------
# phase 3: load raw
# ---------------------------------------------------------------------------


def load_raw(load_id: int) -> int:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """INSERT INTO raw.load_catalog (load_id, source, status, notes)
                   VALUES (:load_id, :source, 'running', :notes)"""
            ),
            {
                "load_id": load_id,
                "source": ", ".join(p.name for _, p in CSV_FILES),
                "notes": "loader run",
            },
        )

    total = 0
    engine = get_engine()
    for origin, path in CSV_FILES:
        df = _read_csv(path, origin)
        df.insert(0, "load_id", load_id)
        console.print(f"   [dim]- {path.name}: {len(df)} rows -> raw.acv_slice_csv[/]")
        df.to_sql(
            "acv_slice_csv",
            engine,
            schema="raw",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )
        total += len(df)

    with engine.begin() as conn:
        conn.execute(
            text(
                """UPDATE raw.load_catalog
                   SET finished_at = now(), rows_inserted = :n, status = 'ok'
                   WHERE load_id = :load_id"""
            ),
            {"load_id": load_id, "n": total},
        )
    return total


# ---------------------------------------------------------------------------
# phase 4: rebuild staging
# ---------------------------------------------------------------------------

# Mapping of raw quoted column name -> staging column name.
# Kept here (not in SQL) so that the rename map lives next to the parser
# and is easy to audit/test.
RAW_TO_STG: dict[str, str] = {
    "ID": "patient_id",
    "Patien Age in Study": "age_years",
    "Patient Sex": "sex",
    "NIHSS": "nihss",
    "Evolution Time": "evolution_minutes",
    "ASPECTS": "aspects",
    "Evolution Time_Clas": "is_over_window",
}

RADIOMIC_PREFIXES = ("shape2D", "firstorder", "glcm", "glszm")


def _radiomic_select_lines() -> list[str]:
    """Return SELECT lines that cast each radiomic column to DOUBLE PRECISION
    and rename it to the staging snake_case column name."""
    import csv

    sample = RAW_DIR / "acv_clasif_train.csv"
    with sample.open(encoding="utf-8") as f:
        cols = next(csv.reader(f))
    lines: list[str] = []
    for col in cols:
        if any(col.startswith(f"original_{p}_") for p in RADIOMIC_PREFIXES):
            new_name = col.replace("original_", "").lower()
            lines.append(f'  NULLIF("{col}", \'\')::DOUBLE PRECISION AS {new_name}')
    return lines


def rebuild_staging(load_id: int) -> int:
    radiomic_lines = _radiomic_select_lines()
    radiomic_block = ",\n".join(radiomic_lines)

    sql = f"""
    TRUNCATE staging.fct_slice_stg RESTART IDENTITY CASCADE;

    INSERT INTO staging.fct_slice_stg (
        load_id, dataset_origin, row_order, patient_id, slice_order,
        {", ".join(line.split(" AS ")[-1].strip() for line in radiomic_lines)},
        age_years, sex,
        nihss, nihss_was_missing,
        aspects, aspects_was_missing,
        evolution_minutes, evolution_hours,
        is_over_window, loaded_at
    )
    SELECT
        load_id,
        dataset_origin,
        row_order,
        "ID" AS patient_id,
        ROW_NUMBER() OVER (PARTITION BY dataset_origin, "ID" ORDER BY row_order) AS slice_order,
{radiomic_block},
        -- Cast via DOUBLE PRECISION first: the CSV stores integer-valued
        -- fields as "10.0" / "51.0", which Postgres rejects as INTEGER text.
        NULLIF("Patien Age in Study", '')::DOUBLE PRECISION::INTEGER  AS age_years,
        UPPER(NULLIF("Patient Sex", ''))::CHAR(1)                      AS sex,
        NULLIF("NIHSS", '')::DOUBLE PRECISION                          AS nihss,
        ("NIHSS" IS NULL OR "NIHSS" = '')                              AS nihss_was_missing,
        NULLIF("ASPECTS", '')::DOUBLE PRECISION::INTEGER               AS aspects,
        ("ASPECTS" IS NULL OR "ASPECTS" = '')                          AS aspects_was_missing,
        NULLIF("Evolution Time", '')::DOUBLE PRECISION AS evolution_minutes,
        NULLIF("Evolution Time", '')::DOUBLE PRECISION / 60.0 AS evolution_hours,
        NULLIF("Evolution Time_Clas", '')::DOUBLE PRECISION::SMALLINT  AS is_over_window,
        loaded_at
    FROM raw.acv_slice_csv
    WHERE load_id = :load_id;
    """

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql), {"load_id": load_id})
        n = conn.execute(text("SELECT COUNT(*) FROM staging.fct_slice_stg")).scalar_one()
    return int(n)


# ---------------------------------------------------------------------------
# phase 5: rebuild analytics
# ---------------------------------------------------------------------------

REBUILD_ANALYTICS_SQL = """
TRUNCATE analytics.fct_slice    RESTART IDENTITY CASCADE;
TRUNCATE analytics.dim_patient  RESTART IDENTITY CASCADE;
TRUNCATE analytics.dim_severity RESTART IDENTITY CASCADE;

INSERT INTO analytics.dim_patient (patient_id, dataset_origin, sex, age_years, age_bucket, first_loaded_at)
SELECT
    patient_id,
    dataset_origin,
    MAX(sex)        AS sex,
    MAX(age_years)  AS age_years,
    CASE
        WHEN MAX(age_years) < 50 THEN '<50'
        WHEN MAX(age_years) < 65 THEN '50-64'
        WHEN MAX(age_years) < 80 THEN '65-79'
        ELSE '80+'
    END             AS age_bucket,
    MIN(loaded_at)  AS first_loaded_at
FROM staging.fct_slice_stg
GROUP BY patient_id, dataset_origin;

INSERT INTO analytics.dim_severity (nihss_band, aspects_band)
SELECT DISTINCT
    CASE
        WHEN nihss IS NULL THEN 'unknown'
        WHEN nihss = 0     THEN 'none'
        WHEN nihss <= 4    THEN 'minor'
        WHEN nihss <= 15   THEN 'moderate'
        WHEN nihss <= 20   THEN 'mod-severe'
        ELSE 'severe'
    END AS nihss_band,
    CASE
        WHEN aspects IS NULL THEN 'unknown'
        WHEN aspects <= 7    THEN 'significant'
        ELSE 'preserved'
    END AS aspects_band
FROM staging.fct_slice_stg;

INSERT INTO analytics.fct_slice (
    patient_sk, severity_sk, dataset_origin, slice_order,
    shape2d_meshsurface, shape2d_perimeter, shape2d_sphericity, shape2d_sphericaldisproportion,
    shape2d_maximumdiameter, shape2d_majoraxislength, shape2d_minoraxislength, shape2d_elongation,
    firstorder_entropy, firstorder_minimum, firstorder_10percentile, firstorder_90percentile,
    firstorder_maximum, firstorder_mean, firstorder_median, firstorder_interquartilerange,
    firstorder_range, firstorder_meanabsolutedeviation, firstorder_robustmeanabsolutedeviation,
    firstorder_rootmeansquared, firstorder_standarddeviation, firstorder_skewness,
    firstorder_kurtosis, firstorder_variance, firstorder_uniformity,
    glcm_autocorrelation, glcm_jointaverage, glcm_clusterprominence, glcm_clustershade,
    glcm_clustertendency, glcm_contrast, glcm_correlation, glcm_differenceaverage,
    glcm_differenceentropy, glcm_differencevariance, glcm_jointenergy, glcm_jointentropy,
    glcm_imc1, glcm_imc2, glcm_idm, glcm_mcc, glcm_idmn, glcm_id, glcm_idn,
    glcm_maximumprobability, glcm_sumaverage, glcm_sumentropy, glcm_sumsquares,
    glszm_smallareaemphasis, glszm_largeareaemphasis, glszm_graylevelnonuniformity,
    glszm_graylevelnonuniformitynormalized, glszm_sizezonenonuniformity,
    glszm_sizezonenonuniformitynormalized, glszm_zonepercentage, glszm_graylevelvariance,
    glszm_zonevariance, glszm_zoneentropy, glszm_lowgraylevelzoneemphasis,
    glszm_highgraylevelzoneemphasis, glszm_smallarealowgraylevelemphasis,
    glszm_smallareahighgraylevelemphasis, glszm_largearealowgraylevelemphasis,
    glszm_largeareahighgraylevelemphasis,
    nihss, nihss_was_missing, aspects, aspects_was_missing,
    evolution_minutes, evolution_hours, is_over_window, loaded_at
)
SELECT
    p.patient_sk,
    s.severity_sk,
    stg.dataset_origin, stg.slice_order,
    stg.shape2d_meshsurface, stg.shape2d_perimeter, stg.shape2d_sphericity, stg.shape2d_sphericaldisproportion,
    stg.shape2d_maximumdiameter, stg.shape2d_majoraxislength, stg.shape2d_minoraxislength, stg.shape2d_elongation,
    stg.firstorder_entropy, stg.firstorder_minimum, stg.firstorder_10percentile, stg.firstorder_90percentile,
    stg.firstorder_maximum, stg.firstorder_mean, stg.firstorder_median, stg.firstorder_interquartilerange,
    stg.firstorder_range, stg.firstorder_meanabsolutedeviation, stg.firstorder_robustmeanabsolutedeviation,
    stg.firstorder_rootmeansquared, stg.firstorder_standarddeviation, stg.firstorder_skewness,
    stg.firstorder_kurtosis, stg.firstorder_variance, stg.firstorder_uniformity,
    stg.glcm_autocorrelation, stg.glcm_jointaverage, stg.glcm_clusterprominence, stg.glcm_clustershade,
    stg.glcm_clustertendency, stg.glcm_contrast, stg.glcm_correlation, stg.glcm_differenceaverage,
    stg.glcm_differenceentropy, stg.glcm_differencevariance, stg.glcm_jointenergy, stg.glcm_jointentropy,
    stg.glcm_imc1, stg.glcm_imc2, stg.glcm_idm, stg.glcm_mcc, stg.glcm_idmn, stg.glcm_id, stg.glcm_idn,
    stg.glcm_maximumprobability, stg.glcm_sumaverage, stg.glcm_sumentropy, stg.glcm_sumsquares,
    stg.glszm_smallareaemphasis, stg.glszm_largeareaemphasis, stg.glszm_graylevelnonuniformity,
    stg.glszm_graylevelnonuniformitynormalized, stg.glszm_sizezonenonuniformity,
    stg.glszm_sizezonenonuniformitynormalized, stg.glszm_zonepercentage, stg.glszm_graylevelvariance,
    stg.glszm_zonevariance, stg.glszm_zoneentropy, stg.glszm_lowgraylevelzoneemphasis,
    stg.glszm_highgraylevelzoneemphasis, stg.glszm_smallarealowgraylevelemphasis,
    stg.glszm_smallareahighgraylevelemphasis, stg.glszm_largearealowgraylevelemphasis,
    stg.glszm_largeareahighgraylevelemphasis,
    stg.nihss, stg.nihss_was_missing, stg.aspects, stg.aspects_was_missing,
    stg.evolution_minutes, stg.evolution_hours, stg.is_over_window, stg.loaded_at
FROM staging.fct_slice_stg stg
JOIN analytics.dim_patient  p
  ON p.patient_id = stg.patient_id AND p.dataset_origin = stg.dataset_origin
LEFT JOIN analytics.dim_severity s
  ON s.nihss_band = CASE
        WHEN stg.nihss IS NULL THEN 'unknown'
        WHEN stg.nihss = 0     THEN 'none'
        WHEN stg.nihss <= 4    THEN 'minor'
        WHEN stg.nihss <= 15   THEN 'moderate'
        WHEN stg.nihss <= 20   THEN 'mod-severe'
        ELSE 'severe'
     END
 AND s.aspects_band = CASE
        WHEN stg.aspects IS NULL THEN 'unknown'
        WHEN stg.aspects <= 7    THEN 'significant'
        ELSE 'preserved'
     END;

REFRESH MATERIALIZED VIEW analytics.agg_patient;
"""


def rebuild_analytics() -> tuple[int, int, int]:
    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql(REBUILD_ANALYTICS_SQL)
        n_patients = conn.execute(text("SELECT COUNT(*) FROM analytics.dim_patient")).scalar_one()
        n_slices = conn.execute(text("SELECT COUNT(*) FROM analytics.fct_slice")).scalar_one()
        n_agg = conn.execute(text("SELECT COUNT(*) FROM analytics.agg_patient")).scalar_one()
    return int(n_patients), int(n_slices), int(n_agg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    skip_ddl: bool = typer.Option(False, "--skip-ddl", help="Don't reapply sql/*.sql before loading."),
    reset: bool = typer.Option(False, "--reset", help="Truncate everything before loading."),
) -> None:
    """Full pipeline: DDL -> raw -> staging -> analytics."""
    t0 = time.time()
    if not skip_ddl:
        apply_ddl()
    if reset:
        reset_all()

    load_id = _new_load_id()
    console.print(f"[cyan]==> Load id = {load_id}[/]")

    console.print("[cyan]==> Loading raw[/]")
    n_raw = load_raw(load_id)
    console.print(f"[green]   raw rows inserted: {n_raw}[/]")

    console.print("[cyan]==> Rebuilding staging[/]")
    n_stg = rebuild_staging(load_id)
    console.print(f"[green]   staging.fct_slice_stg rows: {n_stg}[/]")

    console.print("[cyan]==> Rebuilding analytics[/]")
    n_pat, n_fct, n_agg = rebuild_analytics()
    console.print(
        f"[green]   dim_patient={n_pat} | fct_slice={n_fct} | agg_patient={n_agg}[/]"
    )

    console.print(f"[bold green]Done in {time.time()-t0:.1f}s[/]")


if __name__ == "__main__":
    app()
