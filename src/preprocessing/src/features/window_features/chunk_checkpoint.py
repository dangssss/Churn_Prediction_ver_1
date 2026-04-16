from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path


def chunk_items(items, chunk_size: int):
    if not items:
        return []
    if chunk_size <= 0 or chunk_size >= len(items):
        return [list(items)]
    return [list(items[idx : idx + chunk_size]) for idx in range(0, len(items), chunk_size)]


def load_checkpoint(checkpoint_path: str | None, logger):
    if not checkpoint_path:
        return {"completed_chunks": [], "chunks": {}}

    path = Path(checkpoint_path)
    if not path.exists():
        return {"completed_chunks": [], "chunks": {}}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        completed_chunks = data.get("completed_chunks", [])
        chunks = data.get("chunks", {})
        if not isinstance(completed_chunks, list) or not isinstance(chunks, dict):
            return {"completed_chunks": [], "chunks": {}}
        return {"completed_chunks": completed_chunks, "chunks": chunks}
    except Exception as exc:
        logger.warning(f"Checkpoint load failed, starting fresh: {exc}")
        return {"completed_chunks": [], "chunks": {}}


def save_checkpoint(checkpoint_path: str | None, checkpoint_data):
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint_data, indent=2), encoding="utf-8")


def execute_sql_pairs_in_chunks(
    engine,
    all_sql_pairs,
    months_list,
    window_sizes,
    month_chunk_size: int,
    window_group_size: int,
    batch_size: int,
    worker_count: int,
    total: int,
    completed_chunk_ids,
    checkpoint,
    checkpoint_path,
    logger,
):
    month_groups = chunk_items(months_list, month_chunk_size)
    size_groups = chunk_items(list(window_sizes), window_group_size)
    total_chunks = len(month_groups) * len(size_groups)
    chunk_position = 0
    processed_tables = 0

    logger.info(
        f"Chunking plan: {len(month_groups)} month group(s) x {len(size_groups)} window-size group(s)"
    )

    def execute_insert(pair):
        table_name, _, insert_sql, _ = pair
        with engine.begin() as conn:
            conn.exec_driver_sql(insert_sql)
        return table_name

    for month_group in month_groups:
        month_tokens = {month.strftime('%y%m') for month in month_group}
        month_label = f"{month_group[0].strftime('%Y-%m')}..{month_group[-1].strftime('%Y-%m')}"

        for size_group in size_groups:
            chunk_position += 1
            size_label = f"{size_group[0]}..{size_group[-1]}"
            chunk_id = f"months:{month_label}|sizes:{size_label}"

            chunk_pairs = [
                pair for pair in all_sql_pairs
                if pair[3]['window_size'] in size_group and pair[3]['end_ym'] in month_tokens
            ]

            if not chunk_pairs:
                logger.info(f"[CHUNK {chunk_position}/{total_chunks}] {chunk_id} -> no tables, skip")
                continue

            if chunk_id in completed_chunk_ids:
                processed_tables += len(chunk_pairs)
                logger.info(
                    f"[CHUNK {chunk_position}/{total_chunks}] {chunk_id} -> already completed in checkpoint, skip"
                )
                continue

            logger.info(
                f"[CHUNK {chunk_position}/{total_chunks}] {chunk_id} -> {len(chunk_pairs)} table(s), "
                f"workers={worker_count}, batch_size={batch_size}"
            )

            with engine.begin() as conn:
                for table_name, create_sql, _, _ in chunk_pairs:
                    try:
                        conn.exec_driver_sql(create_sql)
                    except Exception as exc:
                        logger.warning(f"Table {table_name} creation failed: {exc}")

            for batch_idx in range(0, len(chunk_pairs), batch_size):
                batch = chunk_pairs[batch_idx : batch_idx + batch_size]
                logger.info(f"  [CHUNK-BATCH {batch_idx // batch_size + 1}] inserting {len(batch)} table(s)")
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [executor.submit(execute_insert, pair) for pair in batch]
                    for offset, future in enumerate(futures, start=1):
                        table_name = batch[offset - 1][0]
                        global_idx = processed_tables + offset
                        logger.info(f"    [{global_idx}/{total}] {table_name}...")
                        try:
                            future.result()
                        except Exception as exc:
                            logger.error(f"Insert to {table_name} failed: {exc}")
                            raise

                processed_tables += len(batch)

            completed_chunk_ids.add(chunk_id)
            checkpoint["completed_chunks"] = sorted(completed_chunk_ids)
            checkpoint["chunks"][chunk_id] = {"tables": len(chunk_pairs)}
            save_checkpoint(checkpoint_path, checkpoint)
            logger.info(f"[CHUNK DONE] {chunk_id}")
