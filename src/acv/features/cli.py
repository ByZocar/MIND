"""CLI: `python -m acv.features.cli`.

Pipeline:
    1. Build wide patient-level feature matrix from analytics.fct_slice.
    2. Drop zero-variance columns (decided on train, applied to all).
    3. Drop highly-correlated pairs (|Spearman| > threshold) on train,
       keep higher-MI feature.
    4. Compute MI ranking over survivors on train only.
    5. Persist:
       - analytics.features_v1 (Postgres, wide)
       - data/processed/features_v1.parquet
       - data/processed/features_v1_mi_rank.parquet

We use stdlib argparse instead of typer to avoid the well-known
click+bool-default flakiness across versions.
"""
from __future__ import annotations

import argparse
import time

from rich.console import Console

from acv.features.build import build_features_v1
from acv.features.inventory import GROUP_KEY, TARGET_COL
from acv.features.persist import (
    PARQUET_PATH,
    PARQUET_RANK_PATH,
    save_mi_rank,
    save_to_parquet,
    save_to_postgres,
)
from acv.features.select import (
    drop_high_correlation,
    drop_zero_variance,
    rank_by_mutual_info,
)

console = Console()


def main(corr_threshold: float = 0.95, skip_db: bool = False) -> None:
    """Run the full feature-engineering pipeline end-to-end."""
    t0 = time.time()

    console.print("[cyan]==> Building wide feature matrix[/]")
    df_full = build_features_v1()
    console.print(f"   rows={len(df_full)}  cols={df_full.shape[1]}")

    train_mask = df_full[GROUP_KEY] == "train"
    console.print(
        f"   train patients: {int(train_mask.sum())}  test patients: {int((~train_mask).sum())}"
    )

    console.print("[cyan]==> Dropping zero-variance columns (decided on train)[/]")
    train_only = df_full.loc[train_mask].reset_index(drop=True)
    _, dropped_zv = drop_zero_variance(train_only)
    df_full = df_full.drop(columns=dropped_zv, errors="ignore")
    console.print(f"   dropped: {len(dropped_zv)}")

    console.print(f"[cyan]==> Dropping highly correlated pairs (|rho| > {corr_threshold}) on train[/]")
    train_only = df_full.loc[train_mask].reset_index(drop=True)
    _, dropped_pairs = drop_high_correlation(
        train_only, train_only[TARGET_COL], threshold=corr_threshold
    )
    dropped_cols = [loser for _, loser, _ in dropped_pairs]
    df_full = df_full.drop(columns=dropped_cols, errors="ignore")
    console.print(
        f"   pairs pruned: {len(dropped_pairs)} | surviving cols: {df_full.shape[1]}"
    )

    console.print("[cyan]==> Ranking surviving features by MI (train only)[/]")
    train_only = df_full.loc[df_full[GROUP_KEY] == "train"].reset_index(drop=True)
    rank = rank_by_mutual_info(train_only, train_only[TARGET_COL])
    console.print("   top-5 by MI:")
    for _, r in rank.head(5).iterrows():
        console.print(f"      {r.feature:60s}  MI={r.mi:.4f}")

    console.print(f"[cyan]==> Saving parquet: {PARQUET_PATH}[/]")
    save_to_parquet(df_full)
    console.print(f"[cyan]==> Saving MI rank: {PARQUET_RANK_PATH}[/]")
    save_mi_rank(rank)

    if not skip_db:
        console.print("[cyan]==> Persisting to Postgres analytics.features_v1[/]")
        n = save_to_postgres(df_full)
        console.print(f"   rows in DW: {n}")
    else:
        console.print("[yellow]   --skip-db given: NOT writing to Postgres[/]")

    console.print(f"[bold green]Done in {time.time()-t0:.1f}s[/]")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ACV patient-level features.")
    p.add_argument(
        "--corr-threshold",
        type=float,
        default=0.95,
        help="|Spearman| above this is pruned (default: 0.95).",
    )
    p.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip writing to Postgres analytics.features_v1.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    main(corr_threshold=args.corr_threshold, skip_db=args.skip_db)
