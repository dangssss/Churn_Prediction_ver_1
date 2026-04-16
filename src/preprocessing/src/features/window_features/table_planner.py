from sqlalchemy import inspect


def split_tables_to_keep_and_recompute(existing_tables, recompute_last_n: int):
    if recompute_last_n <= 0:
        return existing_tables, []
    if len(existing_tables) <= recompute_last_n:
        return [], existing_tables
    return existing_tables[:-recompute_last_n], existing_tables[-recompute_last_n:]


def truncate_tables(engine, table_names, logger):
    if not table_names:
        return 0

    truncated = 0
    with engine.begin() as conn:
        for table_name in table_names:
            try:
                conn.exec_driver_sql(f"TRUNCATE TABLE {table_name};")
                truncated += 1
            except Exception as exc:
                logger.warning(f"Failed to truncate {table_name}: {exc}")
    return truncated


def list_empty_tables(engine, schema_name: str, table_names, logger):
    if not table_names:
        return []

    empty_tables = []
    with engine.begin() as conn:
        for table_name in table_names:
            qualified_name = f'{schema_name}.{table_name}'
            try:
                result = conn.exec_driver_sql(
                    f"SELECT EXISTS (SELECT 1 FROM {qualified_name} LIMIT 1);"
                )
                has_rows = bool(result.scalar())
                if not has_rows:
                    empty_tables.append(table_name)
            except Exception as exc:
                logger.warning(f"Failed to inspect row existence for {qualified_name}: {exc}")

    return empty_tables


def list_existing_window_tables(engine):
    inspector = inspect(engine)
    table_names = inspector.get_table_names(schema='data_window')
    return sorted([name for name in table_names if name.startswith('cus_feature_')])


def parse_window_size(table_name: str):
    parts = table_name.split('_')
    if len(parts) < 4:
        return None
    window_part = parts[2]
    if not window_part.endswith('m'):
        return None
    try:
        return int(window_part[:-1])
    except ValueError:
        return None
