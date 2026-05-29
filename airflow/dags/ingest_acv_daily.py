"""Airflow DAG — ingest_acv_daily.

Pipeline:
    apply_ddl
        -> load_raw_csvs
        -> rebuild_staging
        -> ge_quality_gate
        -> rebuild_analytics

The quality gate is a HARD gate: if any expectation fails the
downstream task `rebuild_analytics` is skipped and the run goes red.

Each task is a thin Airflow wrapper around a function in
`acv.io.load_csv_to_raw` / `acv.quality.gate`. Business logic lives in
the project code, not in the DAG file — this keeps the orchestration
testable independently.

Schedule: daily at 03:00 (placeholder; the medical operation will be
triggered manually anyway during v1).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

from airflow.decorators import dag, task

DEFAULT_ARGS = {
    "owner": "data-platform",
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="ingest_acv_daily",
    description="ACV stroke window — raw -> staging -> GE gate -> analytics",
    schedule="0 3 * * *",
    start_date=datetime(2026, 5, 25),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["acv", "etl", "fvl"],
    doc_md=__doc__,
)
def ingest_acv_daily():
    @task(task_id="apply_ddl")
    def apply_ddl_task() -> int:
        from acv.io.load_csv_to_raw import apply_ddl

        apply_ddl()
        return int(time.time() * 1000)

    @task(task_id="load_raw_csvs")
    def load_raw_task(load_id: int) -> int:
        from acv.io.load_csv_to_raw import load_raw

        return load_raw(load_id)

    @task(task_id="rebuild_staging")
    def rebuild_staging_task(load_id: int, _n_raw: int) -> int:
        from acv.io.load_csv_to_raw import rebuild_staging

        return rebuild_staging(load_id)

    @task(task_id="ge_quality_gate")
    def quality_gate_task(load_id: int, _n_stg: int) -> dict:
        from acv.quality.gate import hard_gate

        results = hard_gate(load_id=load_id)
        return {name: r.success for name, r in results.items()}

    @task(task_id="rebuild_analytics")
    def rebuild_analytics_task(gate_results: dict) -> dict:
        from acv.io.load_csv_to_raw import rebuild_analytics

        if not all(gate_results.values()):
            raise RuntimeError(f"Quality gate did not pass: {gate_results}")
        n_pat, n_fct, n_agg = rebuild_analytics()
        return {"dim_patient": n_pat, "fct_slice": n_fct, "agg_patient": n_agg}

    load_id = apply_ddl_task()
    n_raw = load_raw_task(load_id)
    n_stg = rebuild_staging_task(load_id, n_raw)
    gate = quality_gate_task(load_id, n_stg)
    rebuild_analytics_task(gate)


ingest_acv_daily()
