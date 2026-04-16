
from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from preprocess.feature_tables import list_k_available, max_window_end_for_k
from infra.yymm import shift_yymm
from preprocess.static_features import load_cus_lifetime
from baseline.runner import eval_one_k_train_val

def run_sweep_k(
    engine: Engine,
    *,
    horizon: int,
    limit_rows_each: int | None = None,
    k_min: int = 3,
) -> tuple[dict, pd.DataFrame]:
    """
    Always sweep K to find the best K for current data (picked by F1 then PR_AUC).

    Returns:
      (best_config_candidate, df_ablation_sorted)
    """
    ks = [int(k) for k in list_k_available(engine) if int(k) >= int(k_min)]
    if not ks:
        raise ValueError("No K available in feature tables.")

    df_static = load_cus_lifetime(engine)

    ablation = []
    for k in ks:
        for use_static in [False, True]:
            out = eval_one_k_train_val(
                engine,
                k=int(k),
                horizon=int(horizon),
                use_static=bool(use_static),
                df_static=df_static,
                limit_rows_each=limit_rows_each
            )
            if out is None:
                continue
            ablation.append(out)
            print(f"K={k} | use_static={use_static} | val={out.get('val_month')} | "
                  f"F1={out['f1']:.4f} | PR_AUC={out['PR_AUC_val']:.4f}")

    if not ablation:
        raise ValueError("Ablation produced no result.")

    df_ab = pd.DataFrame(ablation).sort_values(["f1", "PR_AUC_val"], ascending=False).reset_index(drop=True)

    best_k = int(df_ab.iloc[0]["K"])
    use_static_best = bool(df_ab.iloc[0]["use_static"])
    best_f1_final = float(df_ab.iloc[0]["f1"])
    best_thr_final = float(df_ab.iloc[0]["best_threshold"])
    best_spw_final = float(df_ab.iloc[0]["spw_used"])

    as_of_month = int(max_window_end_for_k(engine, best_k))
    target_month = int(shift_yymm(str(as_of_month), int(horizon)))

    best_config = {
        "as_of_month": as_of_month,
        "target_month": target_month,
        "horizon": int(horizon),
        "best_k": best_k,
        "use_static": use_static_best,
        "best_threshold": best_thr_final,
        "best_spw": best_spw_final,
        "metric_f1_val": best_f1_final,
        "metric_pr_auc_val": float(df_ab.iloc[0]["PR_AUC_val"]),
        "val_month": int(df_ab.iloc[0]["val_month"]),
        "notes": "picked by F1 then PR_AUC; sweep K window_only then static ablation",
    }
    return best_config, df_ab
