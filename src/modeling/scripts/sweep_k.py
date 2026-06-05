from __future__ import annotations

import argparse
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.db import get_engine, smoke_test
from baseline.sweep import run_sweep_k
from config_store.best_config import upsert_best_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--limit-rows-each", type=int, default=None)
    ap.add_argument("--k-min", type=int, default=3)
    args = ap.parse_args()

    engine = get_engine()
    print("DB:", smoke_test(engine))

    best_config, df_ab = run_sweep_k(
        engine,
        horizon=int(args.horizon),
        limit_rows_each=args.limit_rows_each,
        k_min=int(args.k_min),
    )

    upsert_best_config(engine, best_config)
    print("Saved best_config:", best_config)

    # Optional: print top-10
    print("\nTOP-10 sweep results:")
    cols = [
        c for c in [
            "K", "use_static", "val_month",
            "f1", "precision", "recall", "PR_AUC_val", "ROC_AUC_val",
            "val_prevalence", "best_threshold", "spw_used",
        ] if c in df_ab.columns
    ]
    print(df_ab[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
