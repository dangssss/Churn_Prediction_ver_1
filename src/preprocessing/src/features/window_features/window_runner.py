import pandas as pd
from logging_config import get_logger

from .chunk_checkpoint import execute_sql_pairs_in_chunks
from .chunk_checkpoint import load_checkpoint
from .sql_builder import render_window_sqls
from .generate_window_table_names import generate_window_table_names
from .stage_tables import WINDOW_BCCP_TABLE
from .stage_tables import WINDOW_COMPLAINTS_TABLE
from .stage_tables import WINDOW_SOURCE_TABLE
from .stage_tables import cleanup_stage_tables
from .stage_tables import prepare_stage_tables
from .table_planner import list_existing_window_tables
from .table_planner import list_empty_tables
from .table_planner import parse_window_size
from .table_planner import split_tables_to_keep_and_recompute
from .table_planner import truncate_tables

logger = get_logger('window_runner')


def render_and_run_all(
    engine,
    months,
    window_sizes,
    enable_optimization: bool = False,
    recompute_last_n: int = 0,
    batch_size: int = 5,
    max_workers: int = 4,
    month_chunk_size: int = 2,
    window_group_size: int = 2,
    checkpoint_path: str | None = None,
    resume_from_checkpoint: bool = False,
):
    logger.info(f"Starting window feature aggregation ({len(window_sizes)} sizes x {len(months)} months)")

    months_list = list(months)
    if not months_list:
        logger.info('No months supplied for window aggregation')
        return {'total_possible': 0, 'to_compute': 0, 'recomputed': 0, 'new_tables': 0, 'kept_tables': 0}

    specs_by_size = generate_window_table_names(months_list, window_sizes, pd.Timestamp('2025-01-01'))
    all_window_specs = [spec for specs in specs_by_size.values() for spec in specs]
    if not all_window_specs:
        logger.info('No window tables need computation')
        return {'total_possible': 0, 'to_compute': 0, 'recomputed': 0, 'new_tables': 0, 'kept_tables': 0}

    existing_tables = list_existing_window_tables(engine)
    existing_by_size = {}
    for table_name in existing_tables:
        window_size = parse_window_size(table_name)
        if window_size is None:
            continue
        existing_by_size.setdefault(window_size, []).append(table_name)

    specs_to_compute = []
    kept_tables = 0
    recompute_tables = 0
    new_tables = 0

    for window_size in window_sizes:
        size_specs = specs_by_size.get(window_size, [])
        existing_for_size = sorted(existing_by_size.get(window_size, []))

        empty_for_size = set(list_empty_tables(engine, 'data_window', existing_for_size, logger))
        if empty_for_size:
            logger.info(
                f"Window {window_size}m: found {len(empty_for_size)} empty existing table(s), forcing recompute"
            )

        _, to_recompute = split_tables_to_keep_and_recompute(existing_for_size, recompute_last_n)
        recompute_set = set(to_recompute) | empty_for_size

        if recompute_set:
            truncated = truncate_tables(engine, [f"data_window.{name}" for name in recompute_set], logger)
            logger.info(
                f"Window {window_size}m: truncated {truncated}/{len(recompute_set)} recompute table(s)"
            )

        for spec in size_specs:
            short_name = spec['short_name']
            if short_name in recompute_set:
                specs_to_compute.append(spec)
                recompute_tables += 1
            elif short_name in existing_for_size:
                kept_tables += 1
            else:
                specs_to_compute.append(spec)
                new_tables += 1

    total = len(specs_to_compute)
    logger.info(f"Window plan: keep={kept_tables}, recompute={recompute_tables}, new={new_tables}, to_compute={total}")

    if total == 0:
        logger.info('No window tables need computation after incremental planning')
        return {
            'total_possible': len(all_window_specs),
            'to_compute': 0,
            'recomputed': recompute_tables,
            'new_tables': new_tables,
            'kept_tables': kept_tables,
        }

    full_start_date = months_list[0].strftime('%Y-%m-%d')
    full_end_date = (months_list[-1] + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')

    all_sql_pairs = []
    checkpoint = load_checkpoint(checkpoint_path, logger) if resume_from_checkpoint else {"completed_chunks": [], "chunks": {}}
    completed_chunk_ids = set(checkpoint.get("completed_chunks", []))
    logger.info(f"Chunking resume mode: {resume_from_checkpoint}")

    try:
        prepare_stage_tables(engine, full_start_date, full_end_date)
        for spec in specs_to_compute:
            create_sql, insert_sql = render_window_sqls(
                spec['table_name'],
                spec['start_date'],
                spec['end_date'],
                spec['start_ym'],
                spec['end_ym'],
                spec['window_size'],
                WINDOW_SOURCE_TABLE,
                WINDOW_COMPLAINTS_TABLE,
                WINDOW_BCCP_TABLE,
            )
            all_sql_pairs.append((spec['table_name'], create_sql, insert_sql, spec))

        worker_count = max(1, min(max_workers, batch_size))
        execute_sql_pairs_in_chunks(
            engine=engine,
            all_sql_pairs=all_sql_pairs,
            months_list=months_list,
            window_sizes=window_sizes,
            month_chunk_size=month_chunk_size,
            window_group_size=window_group_size,
            batch_size=batch_size,
            worker_count=worker_count,
            total=total,
            completed_chunk_ids=completed_chunk_ids,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            logger=logger,
        )

        logger.info(f"Window feature aggregation complete ({total} windows)")
        return {
            'total_possible': len(all_window_specs),
            'to_compute': total,
            'recomputed': recompute_tables,
            'new_tables': new_tables,
            'kept_tables': kept_tables,
            'completed_chunks': len(completed_chunk_ids),
        }
    finally:
        cleanup_stage_tables(engine)


def render_and_run_optimized(
    engine,
    months,
    window_sizes,
    enable_optimization: bool = True,
    recompute_last_n: int = 2,
    batch_size: int = 5,
):
    return render_and_run_all(
        engine,
        months,
        window_sizes,
        enable_optimization=enable_optimization,
        recompute_last_n=recompute_last_n,
        batch_size=batch_size,
    )
