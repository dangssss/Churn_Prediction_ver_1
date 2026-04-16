from __future__ import annotations

"""
Step 24: Enrich Dossier (Drill-down mạnh)
----------------------------------------
Mục tiêu: tạo 2 bảng monthly dossier tách riêng:
- data_static.cus_dossier_active_<YYMM>  : khách active được đưa vào danh sách risk (lọc theo churn_rate %)
- data_static.cus_dossier_churned_<YYMM> : khách churned_now (post-mortem list)

Nguồn enrich (tối đa):
- public.cas_info
- public.cas_customer (monthly metrics + ser_* + complaintXXX)
- public.cms_complaint (chi tiết complaint)
- bccp_orderitem (hoặc partition bccp_orderitem_YYMM): drilldown theo service_code/region/province + SLA/delay/quality

Output lưu TEXT JSON (để an toàn với DB), kèm 1 số cột phẳng quan trọng.
"""

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sqlalchemy import text, bindparam
from sqlalchemy.engine import Engine

from infra.yymm import shift_yymm

SERVICE_CODES = ["C", "E", "M", "P", "R", "U", "L", "Q"]
COMPLAINT_CODES = [114, 115, 116, 134, 194, 554, 595, 314, 594, 274, 614, 654, 234, 174]

# -------------------------
# Date helpers
# -------------------------

def yymm_to_date(yymm: int) -> date:
    """2510 -> 2025-10-01"""
    y = 2000 + (yymm // 100)
    m = yymm % 100
    return date(y, m, 1)

def yymm_str(yymm: int) -> str:
    return f"{int(yymm):04d}"

def next_month_yymm(yymm: int) -> int:
    return int(shift_yymm(yymm_str(yymm), 1))

def month_range(start_yymm: int, end_yymm: int) -> List[int]:
    cur = start_yymm
    out = []
    while cur <= end_yymm:
        out.append(cur)
        cur = int(shift_yymm(yymm_str(cur), 1))
    return out

# -------------------------
# SQL helpers
# -------------------------

def list_tables(engine: Engine, schema: str, like_prefix: str) -> List[str]:
    q = text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema AND table_name LIKE :pat
        ORDER BY table_name
    """)
    df = pd.read_sql(q, engine, params={"schema": schema, "pat": f"{like_prefix}%"})
    return df["table_name"].tolist() if not df.empty else []

def table_exists(engine: Engine, schema: str, table: str) -> bool:
    q = text("""
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = :schema AND table_name = :table
        LIMIT 1
    """)
    df = pd.read_sql(q, engine, params={"schema": schema, "table": table})
    return not df.empty

def _chunks(xs: List[str], n: int = 500) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

# -------------------------
# Fetchers
# -------------------------

def fetch_cas_info(engine: Engine, cms_codes: List[str]) -> pd.DataFrame:
    """Fetch cas_info for given customers, pick latest record per customer if multiple exist."""
    if not cms_codes:
        return pd.DataFrame()
    # use expanding bind param for IN
    q = text("""
        SELECT
            cms_code_enc, crm_code_enc, cus_province,
            contract_service, tenure, custype,
            contract_classify, contract_sig_first,
            contract_mgr_org, cus_poscode,
            customer_update_date
        FROM public.cas_info
        WHERE cms_code_enc IN :codes
    """).bindparams(bindparam("codes", expanding=True))
    frames = []
    for ch in _chunks(cms_codes, 500):
        frames.append(pd.read_sql(q, engine, params={"codes": ch}))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    # pick latest per customer by customer_update_date if exists, else keep first
    if "customer_update_date" in df.columns:
        df["customer_update_date"] = pd.to_datetime(df["customer_update_date"], errors="coerce")
        df = df.sort_values(["cms_code_enc", "customer_update_date"], ascending=[True, False])
        df = df.drop_duplicates("cms_code_enc", keep="first")
    return df

def fetch_cas_customer(engine: Engine, cms_codes: List[str], start_yymm: int, end_yymm: int) -> pd.DataFrame:
    """Fetch cas_customer monthly series for customers in [start_yymm, end_yymm]."""
    if not cms_codes:
        return pd.DataFrame()
    start_date = yymm_to_date(start_yymm)
    end_date = yymm_to_date(end_yymm)
    q = text("""
        SELECT
            cms_code_enc, report_month,
            item_count, weight_kg, total_fee,
            intra_province, international,
            ser_c, ser_e, ser_m, ser_p, ser_r, ser_u, ser_l, ser_q,
            delay_day, delay_count,
            nodone, refunded, noaccepted, lost_order,
            lastday, noservice, dev_item,
            order_score, satisfaction_score,
            total_complaint,
            complaint114, complaint115, complaint116, complaint134, complaint194,
            complaint554, complaint595, complaint314, complaint594, complaint274,
            complaint614, complaint654, complaint234, complaint174,
            update_at
        FROM public.cas_customer
        WHERE cms_code_enc IN :codes
          AND report_month >= :start_date AND report_month <= :end_date
    """).bindparams(bindparam("codes", expanding=True))
    frames = []
    for ch in _chunks(cms_codes, 400):
        frames.append(pd.read_sql(q, engine, params={"codes": ch, "start_date": start_date, "end_date": end_date}))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty and "report_month" in df.columns:
        df["report_month"] = pd.to_datetime(df["report_month"], errors="coerce")
    return df

def fetch_cms_complaint(engine: Engine, cms_codes: List[str], start_dt: date, end_dt: date) -> pd.DataFrame:
    """Fetch complaint detail rows in [start_dt, end_dt)."""
    if not cms_codes:
        return pd.DataFrame()
    q = text("""
        SELECT
            cms_code_enc, item_code,
            create_complaint_date, exp_complaint_date, close_complaint_date,
            delay_complaint, complaint_code,
            complaint_content, complaint_content_bit,
            complaint_update_date, etl_date
        FROM public.cms_complaint
        WHERE cms_code_enc IN :codes
          AND create_complaint_date >= :start_dt
          AND create_complaint_date < :end_dt
    """).bindparams(bindparam("codes", expanding=True))
    frames = []
    for ch in _chunks(cms_codes, 300):
        frames.append(pd.read_sql(q, engine, params={"codes": ch, "start_dt": start_dt, "end_dt": end_dt}))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    for c in ["create_complaint_date", "exp_complaint_date", "close_complaint_date", "complaint_update_date", "etl_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _resolve_orderitem_tables(engine: Engine, schema: str, start_yymm: int, end_yymm: int) -> List[str]:
    """Return list of orderitem tables to scan (single table if exists else partitions)."""
    if table_exists(engine, schema, "bccp_orderitem"):
        return ["bccp_orderitem"]
    # partitions
    cand = list_tables(engine, schema, "bccp_orderitem_")
    out = []
    for t in cand:
        m = re.match(r"^bccp_orderitem_(\d{4})$", t)
        if not m:
            continue
        yy = int(m.group(1))
        if start_yymm <= yy <= end_yymm:
            out.append(t)
    return out

def fetch_orderitem_aggs(engine: Engine, cms_codes: List[str], start_dt: date, end_dt: date, start_yymm: int, end_yymm: int) -> Dict[str, pd.DataFrame]:
    """
    Drilldown aggregates from bccp_orderitem / partitions:
    - overall per customer
    - by service_code
    - by region
    - by rec_province_code
    """
    if not cms_codes:
        return {"overall": pd.DataFrame(), "by_service": pd.DataFrame(), "by_region": pd.DataFrame(), "by_rec_province": pd.DataFrame()}
    schema = "public"
    tables = _resolve_orderitem_tables(engine, schema, start_yymm, end_yymm)
    if not tables:
        return {"overall": pd.DataFrame(), "by_service": pd.DataFrame(), "by_region": pd.DataFrame(), "by_rec_province": pd.DataFrame()}

    def _query_for_table(tbl: str, group_cols: List[str]) -> pd.DataFrame:
        group_sql = ", ".join(group_cols) if group_cols else ""
        sel_group = (group_sql + ",") if group_sql else ""
        grp = f"GROUP BY {group_sql}" if group_sql else ""
        q = text(f"""
            SELECT
                {sel_group}
                COUNT(*) AS n_items,
                SUM(COALESCE(total_fee,0)) AS total_fee,
                SUM(CASE WHEN COALESCE(delay_day,0) > 0 THEN 1 ELSE 0 END) AS delay_cnt,
                AVG(COALESCE(delay_day,0)) AS avg_delay_day_all,
                AVG(CASE WHEN COALESCE(delay_day,0) > 0 THEN delay_day ELSE NULL END) AS avg_delay_day_on_delay,
                SUM(CASE WHEN COALESCE(done,0) = 1 THEN 1 ELSE 0 END) AS done_cnt,
                SUM(CASE WHEN COALESCE(refunded,0) = 1 THEN 1 ELSE 0 END) AS refunded_cnt,
                SUM(CASE WHEN COALESCE(no_accepted,0) = 1 THEN 1 ELSE 0 END) AS noaccepted_cnt,
                SUM(CASE WHEN COALESCE(lost_order,0) = 1 THEN 1 ELSE 0 END) AS lost_cnt,
                SUM(COALESCE(total_complaint,0)) AS complaint_cnt
            FROM {schema}.{tbl}
            WHERE cms_code_enc IN :codes
              AND sending_time >= :start_dt
              AND sending_time < :end_dt
            {grp}
        """).bindparams(bindparam("codes", expanding=True))
        frames = []
        for ch in _chunks(cms_codes, 250):
            try:
                frames.append(pd.read_sql(q, engine, params={"codes": ch, "start_dt": start_dt, "end_dt": end_dt}))
            except Exception:
                # table or columns mismatch; skip
                continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # aggregate by scanning tables and concat then re-aggregate in pandas (safer across partitions)
    overall_parts = []
    by_service_parts = []
    by_region_parts = []
    by_rec_parts = []
    for tbl in tables:
        overall_parts.append(_query_for_table(tbl, ["cms_code_enc"]))
        by_service_parts.append(_query_for_table(tbl, ["cms_code_enc", "service_code"]))
        by_region_parts.append(_query_for_table(tbl, ["cms_code_enc", "region"]))
        by_rec_parts.append(_query_for_table(tbl, ["cms_code_enc", "rec_province_code"]))

    def _sum_merge(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        if df.empty:
            return df
        # numeric sum for counts/fee, avg recompute weighted
        num_sum_cols = ["n_items","total_fee","delay_cnt","done_cnt","refunded_cnt","noaccepted_cnt","lost_cnt","complaint_cnt"]
        # keep avg fields; weighted by n_items or delay_cnt for avg_on_delay
        df[num_sum_cols] = df[num_sum_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["avg_delay_day_all"] = pd.to_numeric(df["avg_delay_day_all"], errors="coerce")
        df["avg_delay_day_on_delay"] = pd.to_numeric(df["avg_delay_day_on_delay"], errors="coerce")
        # weighted sums
        df["_w_all"] = df["avg_delay_day_all"].fillna(0) * df["n_items"]
        df["_w_delay"] = df["avg_delay_day_on_delay"].fillna(0) * df["delay_cnt"].replace({0: 0})
        g = df.groupby(group_cols, dropna=False, as_index=False).agg({
            **{c:"sum" for c in num_sum_cols},
            "_w_all":"sum",
            "_w_delay":"sum"
        })
        g["avg_delay_day_all"] = g["_w_all"] / g["n_items"].replace({0: pd.NA})
        g["avg_delay_day_on_delay"] = g["_w_delay"] / g["delay_cnt"].replace({0: pd.NA})
        g = g.drop(columns=["_w_all","_w_delay"])
        return g

    overall = _sum_merge(pd.concat(overall_parts, ignore_index=True) if overall_parts else pd.DataFrame(), ["cms_code_enc"])
    by_service = _sum_merge(pd.concat(by_service_parts, ignore_index=True) if by_service_parts else pd.DataFrame(), ["cms_code_enc","service_code"])
    by_region = _sum_merge(pd.concat(by_region_parts, ignore_index=True) if by_region_parts else pd.DataFrame(), ["cms_code_enc","region"])
    by_rec = _sum_merge(pd.concat(by_rec_parts, ignore_index=True) if by_rec_parts else pd.DataFrame(), ["cms_code_enc","rec_province_code"])

    return {"overall": overall, "by_service": by_service, "by_region": by_region, "by_rec_province": by_rec}

# -------------------------
# Dossier builders
# -------------------------

def _topk_json(df: pd.DataFrame, key_col: str, value_col: str, k: int = 5) -> str:
    if df is None or df.empty:
        return "[]"
    tmp = df[[key_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce").fillna(0)
    tmp = tmp.sort_values(value_col, ascending=False).head(k)
    rows = tmp.to_dict(orient="records")
    return json.dumps(rows, ensure_ascii=False)

def _series_json(df: pd.DataFrame, cols: List[str]) -> str:
    if df is None or df.empty:
        return "[]"
    d = df.copy()
    if "report_month" in d.columns:
        d["report_month"] = pd.to_datetime(d["report_month"], errors="coerce")
        d["report_month"] = d["report_month"].dt.strftime("%Y-%m-%d")
    keep = [c for c in ["report_month"] + cols if c in d.columns]
    return json.dumps(d[keep].to_dict(orient="records"), ensure_ascii=False)

def build_dossier(
    *,
    df_base: pd.DataFrame,
    group_name: str,
    month_yymm: int,
    best_k: int,
    engine: Engine,
    drill_months: int = 6,
    for_churned: bool = False,
) -> pd.DataFrame:
    """
    df_base: MUST contain cms_code_enc, window_end, proactive/reactive flags, churn_type, reasons.
    - active dossier: include churn_probability/churn_rate
    - churned dossier: no scoring required (churn_probability null)
    """
    if df_base is None or df_base.empty:
        return pd.DataFrame()

    d = df_base.copy()
    d["cms_code_enc"] = d["cms_code_enc"].astype(str)

    # Determine time window for drilldown
    end_yymm = int(month_yymm)
    if for_churned:
        # end at churn month start (exclude churn month)
        end_dt = yymm_to_date(end_yymm)
        end_yymm_for_series = int(shift_yymm(yymm_str(end_yymm), -1))
    else:
        # include current month
        end_dt = yymm_to_date(next_month_yymm(end_yymm))
        end_yymm_for_series = end_yymm

    start_yymm = int(shift_yymm(yymm_str(end_yymm_for_series), -(drill_months - 1)))
    start_dt = yymm_to_date(start_yymm)

    cms_codes = d["cms_code_enc"].unique().tolist()

    # Fetch sources
    info = fetch_cas_info(engine, cms_codes)
    cust = fetch_cas_customer(engine, cms_codes, start_yymm=start_yymm, end_yymm=end_yymm_for_series)
    comp = fetch_cms_complaint(engine, cms_codes, start_dt=start_dt, end_dt=end_dt)
    order_aggs = fetch_orderitem_aggs(engine, cms_codes, start_dt=start_dt, end_dt=end_dt, start_yymm=start_yymm, end_yymm=end_yymm_for_series)

    # cas_info merge
    if not info.empty:
        d = d.merge(info, on="cms_code_enc", how="left", suffixes=("", "_info"))

    # cas_customer summarise per customer
    if not cust.empty:
        # service totals across drill window
        ser_cols = [f"ser_{c.lower()}" for c in SERVICE_CODES if f"ser_{c.lower()}" in cust.columns]
        comp_cols = [f"complaint{c}" for c in COMPLAINT_CODES if f"complaint{c}" in cust.columns]
        metric_cols = ["item_count","total_fee","delay_count","nodone","refunded","noaccepted","lost_order","total_complaint","noservice","dev_item","order_score","satisfaction_score"]
        keep_cols = [c for c in metric_cols + ser_cols + comp_cols if c in cust.columns]

        cust_keep = cust[["cms_code_enc","report_month"] + keep_cols].copy()
        # series json (last N months)
        series_cols = [c for c in ["item_count","total_fee","delay_count","nodone","total_complaint","noservice","order_score","satisfaction_score"] + ser_cols + comp_cols if c in cust_keep.columns]
        series_json_map = cust_keep.sort_values(["cms_code_enc","report_month"]).groupby("cms_code_enc").apply(lambda g: _series_json(g, series_cols)).to_dict()

        # aggregate totals
        agg = cust_keep.groupby("cms_code_enc", as_index=False).agg({c:"sum" for c in ["item_count","total_fee","delay_count","nodone","refunded","noaccepted","lost_order","total_complaint"] if c in cust_keep.columns})
        # service mix
        if ser_cols:
            ser_sum = cust_keep.groupby("cms_code_enc", as_index=False)[ser_cols].sum()
            # make json list for service counts
            def _service_json(row):
                items=[]
                total=0
                for c in ser_cols:
                    v=float(row.get(c,0) or 0)
                    total+=v
                    items.append({"service": c.replace("ser_","").upper(), "count": int(v)})
                items.sort(key=lambda x: x["count"], reverse=True)
                # add share
                for it in items:
                    it["share"] = (it["count"]/total) if total>0 else None
                return json.dumps(items, ensure_ascii=False)

            ser_sum["service_mix_json"] = ser_sum.apply(_service_json, axis=1)
            # top service
            def _top_service(row):
                best=None; bestv=-1
                total=0
                for c in ser_cols:
                    v=float(row.get(c,0) or 0); total+=v
                    if v>bestv:
                        bestv=v; best=c
                return pd.Series({
                    "top_service": best.replace("ser_","").upper() if best else None,
                    "top_service_cnt": int(bestv) if bestv>=0 else None,
                    "top_service_share": (bestv/total) if total>0 else None
                })
            top_ser = ser_sum.apply(_top_service, axis=1)
            ser_sum = pd.concat([ser_sum[["cms_code_enc","service_mix_json"]], top_ser], axis=1)
        else:
            ser_sum = pd.DataFrame(columns=["cms_code_enc","service_mix_json","top_service","top_service_cnt","top_service_share"])

        # complaint code mix from cas_customer
        if comp_cols:
            comp_sum = cust_keep.groupby("cms_code_enc", as_index=False)[comp_cols].sum()
            def _compcode_json(row):
                items=[]
                total=0
                for c in comp_cols:
                    v=float(row.get(c,0) or 0)
                    total+=v
                    items.append({"complaint_code": int(c.replace("complaint","")), "count": int(v)})
                items.sort(key=lambda x: x["count"], reverse=True)
                for it in items:
                    it["share"] = (it["count"]/total) if total>0 else None
                return json.dumps(items, ensure_ascii=False)
            comp_sum["complaint_code_mix_json"] = comp_sum.apply(_compcode_json, axis=1)
            def _top_comp(row):
                best=None; bestv=-1
                total=0
                for c in comp_cols:
                    v=float(row.get(c,0) or 0); total+=v
                    if v>bestv:
                        bestv=v; best=c
                return pd.Series({
                    "top_complaint_code": int(best.replace("complaint","")) if best else None,
                    "top_complaint_cnt": int(bestv) if bestv>=0 else None,
                    "top_complaint_share": (bestv/total) if total>0 else None
                })
            top_comp = comp_sum.apply(_top_comp, axis=1)
            comp_sum = pd.concat([comp_sum[["cms_code_enc","complaint_code_mix_json"]], top_comp], axis=1)
        else:
            comp_sum = pd.DataFrame(columns=["cms_code_enc","complaint_code_mix_json","top_complaint_code","top_complaint_cnt","top_complaint_share"])

        d["cas_customer_series_json"] = d["cms_code_enc"].map(series_json_map)
        d = d.merge(agg, on="cms_code_enc", how="left", suffixes=("", "_6m"))
        d = d.merge(ser_sum, on="cms_code_enc", how="left")
        d = d.merge(comp_sum, on="cms_code_enc", how="left")

    # cms_complaint summarise
    if not comp.empty:
        # summary by code
        comp["complaint_code"] = pd.to_numeric(comp["complaint_code"], errors="coerce")
        comp_sum = comp.groupby("cms_code_enc", as_index=False).agg(
            complaints_90d=("complaint_code","count"),
            open_complaints_90d=("close_complaint_date", lambda s: int(pd.isna(s).sum())),
            last_complaint_date=("create_complaint_date","max"),
        )
        # top complaint codes json
        by_code = comp.groupby(["cms_code_enc","complaint_code"], as_index=False).size().rename(columns={"size":"count"})
        top_code_json = by_code.groupby("cms_code_enc").apply(lambda g: _topk_json(g, "complaint_code", "count", k=5)).to_dict()
        # sample latest 3 complaint contents (truncated)
        comp_sorted = comp.sort_values(["cms_code_enc","create_complaint_date"], ascending=[True, False])
        def _sample_json(g):
            rows=[]
            for _,r in g.head(3).iterrows():
                content = r.get("complaint_content")
                if isinstance(content,str) and len(content) > 180:
                    content = content[:180] + "..."
                rows.append({
                    "complaint_code": int(r["complaint_code"]) if pd.notna(r["complaint_code"]) else None,
                    "create_date": r["create_complaint_date"].strftime("%Y-%m-%d") if pd.notna(r["create_complaint_date"]) else None,
                    "delay_complaint": r.get("delay_complaint"),
                    "content": content
                })
            return json.dumps(rows, ensure_ascii=False)
        sample_json = comp_sorted.groupby("cms_code_enc").apply(_sample_json).to_dict()

        comp_sum["last_complaint_date"] = pd.to_datetime(comp_sum["last_complaint_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        d = d.merge(comp_sum, on="cms_code_enc", how="left")
        d["cms_complaint_top_codes_json"] = d["cms_code_enc"].map(top_code_json)
        d["cms_complaint_samples_json"] = d["cms_code_enc"].map(sample_json)

    # orderitem drilldown
    overall = order_aggs.get("overall", pd.DataFrame())
    by_service = order_aggs.get("by_service", pd.DataFrame())
    by_region = order_aggs.get("by_region", pd.DataFrame())
    by_rec = order_aggs.get("by_rec_province", pd.DataFrame())

    if not overall.empty:
        overall = overall.rename(columns={"n_items":"oi_n_items","total_fee":"oi_total_fee","delay_cnt":"oi_delay_cnt","avg_delay_day_all":"oi_avg_delay_day","avg_delay_day_on_delay":"oi_avg_delay_day_on_delay"})
        # add rates
        overall["oi_delay_rate"] = overall["oi_delay_cnt"] / overall["oi_n_items"].replace({0: pd.NA})
        overall["oi_done_rate"] = overall["done_cnt"] / overall["oi_n_items"].replace({0: pd.NA})
        d = d.merge(overall, on="cms_code_enc", how="left")

    if not by_service.empty:
        # top services from orderitem
        by_service["n_items"] = pd.to_numeric(by_service["n_items"], errors="coerce").fillna(0)
        svc_json = by_service.groupby("cms_code_enc").apply(lambda g: _topk_json(g, "service_code", "n_items", k=5)).to_dict()
        d["orderitem_top_services_json"] = d["cms_code_enc"].map(svc_json)

    if not by_region.empty:
        by_region["n_items"] = pd.to_numeric(by_region["n_items"], errors="coerce").fillna(0)
        reg_json = by_region.groupby("cms_code_enc").apply(lambda g: _topk_json(g, "region", "n_items", k=5)).to_dict()
        d["orderitem_top_regions_json"] = d["cms_code_enc"].map(reg_json)

    if not by_rec.empty:
        by_rec["n_items"] = pd.to_numeric(by_rec["n_items"], errors="coerce").fillna(0)
        rec_json = by_rec.groupby("cms_code_enc").apply(lambda g: _topk_json(g, "rec_province_code", "n_items", k=5)).to_dict()
        d["orderitem_top_rec_province_json"] = d["cms_code_enc"].map(rec_json)

    d["dossier_group"] = group_name
    d["dossier_month"] = int(month_yymm)
    d["dossier_start_yymm"] = int(start_yymm)
    d["dossier_end_yymm"] = int(end_yymm_for_series)
    d["generated_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # ensure json text columns exist
    json_cols = [
        "cas_customer_series_json",
        "service_mix_json",
        "complaint_code_mix_json",
        "cms_complaint_top_codes_json",
        "cms_complaint_samples_json",
        "orderitem_top_services_json",
        "orderitem_top_regions_json",
        "orderitem_top_rec_province_json",
    ]
    for c in json_cols:
        if c not in d.columns:
            d[c] = None

    return d

# -------------------------
# Persist to DB
# -------------------------

def ensure_dossier_table_schema(engine: Engine, *, month_yymm: int, group_name: str) -> str:
    month = int(month_yymm)
    table_name = f"cus_dossier_{group_name}_{month}"

    ddl = f"""
    DROP TABLE IF EXISTS data_static.{table_name} CASCADE;

    CREATE TABLE data_static.{table_name} (
        cms_code_enc VARCHAR(64) NOT NULL,
        window_end INT,
        dossier_group VARCHAR(20),
        dossier_month INT,
        dossier_start_yymm INT,
        dossier_end_yymm INT,
        generated_at TIMESTAMP,

        -- scoring
        churn_probability DOUBLE PRECISION,
        churn_rate DOUBLE PRECISION,
        risk_score DOUBLE PRECISION,
        risk_flag INT,

        -- churn-type analysis
        proactive_flag INT,
        reactive_flag INT,
        proactive_pure_flag INT,
        churn_type VARCHAR(20),
        proactive_reasons TEXT,
        reactive_reasons TEXT,
        reason_1 TEXT,
        reason_2 TEXT,
        reason_3 TEXT,

        -- cas_info
        crm_code_enc VARCHAR(64),
        cus_province INT,
        contract_service INT,
        contract_classify INT,
        custype INT,
        tenure INT,
        contract_sig_first TIMESTAMP,
        contract_mgr_org INT,
        cus_poscode INT,

        -- cas_customer aggregates (drill_months window)
        item_count BIGINT,
        total_fee BIGINT,
        delay_count BIGINT,
        nodone BIGINT,
        refunded BIGINT,
        noaccepted BIGINT,
        lost_order BIGINT,
        total_complaint BIGINT,

        top_service VARCHAR(5),
        top_service_cnt BIGINT,
        top_service_share DOUBLE PRECISION,
        top_complaint_code INT,
        top_complaint_cnt BIGINT,
        top_complaint_share DOUBLE PRECISION,

        -- cms_complaint summary
        complaints_90d BIGINT,
        open_complaints_90d BIGINT,
        last_complaint_date DATE,

        -- orderitem overall
        oi_n_items BIGINT,
        oi_total_fee BIGINT,
        oi_delay_cnt BIGINT,
        oi_avg_delay_day DOUBLE PRECISION,
        oi_avg_delay_day_on_delay DOUBLE PRECISION,
        oi_delay_rate DOUBLE PRECISION,
        oi_done_rate DOUBLE PRECISION,
        refunded_cnt BIGINT,
        noaccepted_cnt BIGINT,
        lost_cnt BIGINT,
        complaint_cnt BIGINT,

        -- JSON text payloads
        cas_customer_series_json TEXT,
        service_mix_json TEXT,
        complaint_code_mix_json TEXT,
        cms_complaint_top_codes_json TEXT,
        cms_complaint_samples_json TEXT,
        orderitem_top_services_json TEXT,
        orderitem_top_regions_json TEXT,
        orderitem_top_rec_province_json TEXT
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    return table_name

def insert_dossier(engine: Engine, table_name: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    # align columns
    cols = [
        "cms_code_enc","window_end","dossier_group","dossier_month","dossier_start_yymm","dossier_end_yymm","generated_at",
        "churn_probability","churn_rate","risk_score","risk_flag",
        "proactive_flag","reactive_flag","proactive_pure_flag","churn_type","proactive_reasons","reactive_reasons","reason_1","reason_2","reason_3",
        "crm_code_enc","cus_province","contract_service","contract_classify","custype","tenure","contract_sig_first","contract_mgr_org","cus_poscode",
        "item_count","total_fee","delay_count","nodone","refunded","noaccepted","lost_order","total_complaint",
        "top_service","top_service_cnt","top_service_share","top_complaint_code","top_complaint_cnt","top_complaint_share",
        "complaints_90d","open_complaints_90d","last_complaint_date",
        "oi_n_items","oi_total_fee","oi_delay_cnt","oi_avg_delay_day","oi_avg_delay_day_on_delay","oi_delay_rate","oi_done_rate",
        "refunded_cnt","noaccepted_cnt","lost_cnt","complaint_cnt",
        "cas_customer_series_json","service_mix_json","complaint_code_mix_json",
        "cms_complaint_top_codes_json","cms_complaint_samples_json",
        "orderitem_top_services_json","orderitem_top_regions_json","orderitem_top_rec_province_json"
    ]
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = None
    d = d[cols].copy()
    # normalize types lightly
    d["cms_code_enc"] = d["cms_code_enc"].astype(str)

    insert_sql = f"""
    INSERT INTO data_static.{table_name} (
        {", ".join(cols)}
    ) VALUES (
        {", ".join([f":{c}" for c in cols])}
    );
    """
    with engine.begin() as conn:
        conn.execute(text(insert_sql), d.to_dict(orient="records"))
    return int(len(d))
