from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

def resolve_now_cols(df: pd.DataFrame) -> Dict[str, str]:
    """Find current-month item/revenue columns in feature table.

    Prefer item_last/revenue_last, fallback item_t/revenue_t.
    """
    cols = {c.lower(): c for c in df.columns}

    if "item_last" in cols:
        item_col = cols["item_last"]
    elif "item_t" in cols:
        item_col = cols["item_t"]
    else:
        raise KeyError("Không tìm thấy cột item hiện tại (item_last hoặc item_t)")

    if "revenue_last" in cols:
        rev_col = cols["revenue_last"]
    elif "revenue_t" in cols:
        rev_col = cols["revenue_t"]
    else:
        raise KeyError("Không tìm thấy cột revenue hiện tại (revenue_last hoặc revenue_t)")

    return {"item_now": item_col, "rev_now": rev_col}

def apply_gate(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    cols = resolve_now_cols(d)
    item = pd.to_numeric(d[cols["item_now"]], errors="coerce").fillna(0)
    rev  = pd.to_numeric(d[cols["rev_now"]],  errors="coerce").fillna(0)

    d["is_churned_now"] = ((item == 0) & (rev == 0)).astype(int)
    d["is_active_now"]  = 1 - d["is_churned_now"]
    d["gate_group"]     = np.where(d["is_active_now"] == 1, "active_now", "churned_now")
    return d
