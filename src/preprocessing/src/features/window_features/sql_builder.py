import pandas as pd

from src.features.template_engine import render_template


def build_relative_suffix(month_offset: int) -> str:
    if month_offset == 0:
        return 'last'
    return f"{month_offset}m_ago"


def render_window_sqls(
    table_name: str,
    start_date: str,
    end_date: str,
    start_ym: str,
    end_ym: str,
    window_size: int,
    window_source_table: str,
    complaint_source_table: str,
    bccp_source_table: str,
):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    months = pd.date_range(start, end, freq='MS')
    month_keys = [m.strftime('%y%m') for m in months]
    rel_suffixes = [build_relative_suffix(len(month_keys) - 1 - idx) for idx in range(len(month_keys))]

    case_parts = []
    cols_parts = []
    select_parts = []
    insert_parts = []

    for month_key, rel_suffix in zip(month_keys, rel_suffixes):
        case_parts.append(
            f'MAX(CASE WHEN month_key = \'{month_key}\' THEN item_sum END) AS "item_{rel_suffix}", '
            f'MAX(CASE WHEN month_key = \'{month_key}\' THEN revenue_sum END) AS "revenue_{rel_suffix}", '
            f'MAX(CASE WHEN month_key = \'{month_key}\' THEN complaint_sum END) AS "complaint_{rel_suffix}", '
            f'MAX(CASE WHEN month_key = \'{month_key}\' THEN delay_sum END) AS "delay_{rel_suffix}", '
            f'MAX(CASE WHEN month_key = \'{month_key}\' THEN nodone_sum END) AS "nodone_{rel_suffix}", '
            f'MAX(CASE WHEN month_key = \'{month_key}\' THEN order_score_avg END) AS "order_score_{rel_suffix}", '
            f'MAX(CASE WHEN month_key = \'{month_key}\' THEN satisfaction_avg END) AS "satisfaction_{rel_suffix}"'
        )

        cols_parts.append(
            f'    "item_{rel_suffix}" BIGINT,\n'
            f'    "revenue_{rel_suffix}" BIGINT,\n'
            f'    "complaint_{rel_suffix}" BIGINT,\n'
            f'    "delay_{rel_suffix}" BIGINT,\n'
            f'    "nodone_{rel_suffix}" BIGINT,\n'
            f'    "order_score_{rel_suffix}" DOUBLE PRECISION,\n'
            f'    "satisfaction_{rel_suffix}" DOUBLE PRECISION'
        )

        select_parts.append(
            f'COALESCE(mp."item_{rel_suffix}", 0) AS "item_{rel_suffix}", '
            f'COALESCE(mp."revenue_{rel_suffix}", 0) AS "revenue_{rel_suffix}", '
            f'COALESCE(mp."complaint_{rel_suffix}", 0) AS "complaint_{rel_suffix}", '
            f'COALESCE(mp."delay_{rel_suffix}", 0) AS "delay_{rel_suffix}", '
            f'COALESCE(mp."nodone_{rel_suffix}", 0) AS "nodone_{rel_suffix}", '
            f'COALESCE(mp."order_score_{rel_suffix}", 0) AS "order_score_{rel_suffix}", '
            f'COALESCE(mp."satisfaction_{rel_suffix}", 0) AS "satisfaction_{rel_suffix}"'
        )

        insert_parts.append(
            f'"item_{rel_suffix}", "revenue_{rel_suffix}", "complaint_{rel_suffix}", '
            f'"delay_{rel_suffix}", "nodone_{rel_suffix}", "order_score_{rel_suffix}", '
            f'"satisfaction_{rel_suffix}"'
        )

    table_safe = table_name.replace('.', '_')
    create_sql = render_template(
        'sliding_table',
        TABLE_NAME=table_name,
        TABLE_SAFE=table_safe,
        MONTHLY_COLUMNS=',\n'.join(cols_parts),
    )

    insert_sql = render_template(
        'sliding_aggregate',
        TABLE_NAME=table_name,
        START_DATE=start_date,
        END_DATE=end_date,
        WINDOW_SIZE=window_size,
        START_YM=start_ym,
        END_YM=end_ym,
        WINDOW_SOURCE_TABLE=window_source_table,
        COMPLAINT_SOURCE_TABLE=complaint_source_table,
        BCCP_SOURCE_TABLE=bccp_source_table,
        MONTHLY_CASE_STATEMENTS=',\n        '.join(case_parts),
        MONTHLY_SELECT_COLUMNS=', '.join(select_parts),
        MONTHLY_COLUMNS_LIST=', '.join(insert_parts),
    )

    return create_sql, insert_sql
