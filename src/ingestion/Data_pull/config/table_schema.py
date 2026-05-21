# Data_pull/config/table_schema.py
"""
Định nghĩa schema chính thức cho tất cả bảng production.
Bao gồm:
  - Tên cột chính thức (canonical)
  - Kiểu dữ liệu đúng
  - Vị trí cột (position)
  - Transform logic cho từng cột
"""

from typing import Dict, List, Tuple

# ============================================================
# BCCP_ORDERITEM Schema
# ============================================================
BCCP_ORDERITEM_COLUMNS: List[Tuple[str, str, int]] = [
    # (column_name, data_type, position)
    ("crm_code_enc", "VARCHAR(100)", 1),
    ("cms_code_enc", "VARCHAR(100)", 2),
    ("item_code", "VARCHAR(100)", 3),
    ("service_code", "VARCHAR(10)", 4),
    ("weight_kg", "DECIMAL(10,3)", 5),
    ("length_size", "BIGINT", 6),
    ("width_size", "BIGINT", 7),
    ("height_size", "BIGINT", 8),
    ("total_fee", "BIGINT", 9),
    ("is_domestic", "BIGINT", 10),
    ("country_code", "VARCHAR(20)", 11),
    ("send_province_code", "BIGINT", 12),
    ("send_district_code", "BIGINT", 13),
    ("send_commune_code", "BIGINT", 14),
    ("rec_province_code", "BIGINT", 15),
    ("rec_district_code", "BIGINT", 16),
    ("rec_commune_code", "BIGINT", 17),
    ("region", "VARCHAR(20)", 18),
    ("sending_time", "TIMESTAMPTZ", 19),
    ("ending_time", "TIMESTAMPTZ", 20),
    ("rec_success", "BIGINT", 21),
    ("refunded", "BIGINT", 22),
    ("no_accepted", "BIGINT", 23),
    ("lost_order", "BIGINT", 24),
    ("delay_day", "BIGINT", 25),
    ("done", "BIGINT", 26),
    ("total_complaint", "BIGINT", 27),
    ("complaint114", "BIGINT", 28),
    ("complaint115", "BIGINT", 29),
    ("complaint116", "BIGINT", 30),
    ("complaint134", "BIGINT", 31),
    ("complaint194", "BIGINT", 32),
    ("complaint554", "BIGINT", 33),
    ("complaint595", "BIGINT", 34),
    ("complaint314", "BIGINT", 35),
    ("complaint594", "BIGINT", 36),
    ("complaint274", "BIGINT", 37),
    ("complaint614", "BIGINT", 38),
    ("complaint654", "BIGINT", 39),
    ("complaint234", "BIGINT", 40),
    ("complaint174", "BIGINT", 41),
    ("order_score", "DECIMAL(10,3)", 42),
    ("bccp_update_date", "TIMESTAMPTZ", 43),
]

# ============================================================
# CMS_COMPLAINT Schema
# ============================================================
CMS_COMPLAINT_COLUMNS: List[Tuple[str, str, int]] = [
    ("cms_code_enc", "VARCHAR(100)", 1),
    ("item_code", "VARCHAR(100)", 2),
    ("create_complaint_date", "TIMESTAMPTZ", 3),
    ("exp_complaint_date", "TIMESTAMPTZ", 4),
    ("close_complaint_date", "TIMESTAMPTZ", 5),
    ("delay_complaint", "BIGINT", 6),
    ("complaint_code", "BIGINT", 7),
    ("complaint_content", "TEXT", 8),
    ("complaint_content_bit", "BIGINT", 9),
    ("complaint_update_date", "TIMESTAMPTZ", 10),
    ("etl_date", "TIMESTAMPTZ", 11),
]

# ============================================================
# CAS_CUSTOMER Schema
# ============================================================
CAS_CUSTOMER_COLUMNS: List[Tuple[str, str, int]] = [
    ("cms_code_enc", "VARCHAR(100)", 1),
    ("report_month", "DATE", 2),
    ("item_count", "BIGINT", 3),
    ("weight_kg", "DECIMAL(12,3)", 4),
    ("total_fee", "BIGINT", 5),
    ("intra_province", "BIGINT", 6),
    ("international", "BIGINT", 7),
    ("ser_c", "BIGINT", 8),
    ("ser_e", "BIGINT", 9),
    ("ser_m", "BIGINT", 10),
    ("ser_p", "BIGINT", 11),
    ("ser_r", "BIGINT", 12),
    ("ser_u", "BIGINT", 13),
    ("ser_l", "BIGINT", 14),
    ("ser_q", "BIGINT", 15),
    ("delay_day", "BIGINT", 16),
    ("delay_count", "BIGINT", 17),
    ("nodone", "BIGINT", 18),
    ("refunded", "BIGINT", 19),
    ("noaccepted", "BIGINT", 20),
    ("lost_order", "BIGINT", 21),
    ("lastday", "BIGINT", 22),
    ("noservice", "BIGINT", 23),
    ("dev_item", "DECIMAL(10,3)", 24),
    ("order_score", "DECIMAL(10,3)", 25),
    ("satisfaction_score", "DECIMAL(10,3)", 26),
    ("total_complaint", "BIGINT", 27),
    ("complaint114", "BIGINT", 28),
    ("complaint115", "BIGINT", 29),
    ("complaint116", "BIGINT", 30),
    ("complaint134", "BIGINT", 31),
    ("complaint194", "BIGINT", 32),
    ("complaint554", "BIGINT", 33),
    ("complaint595", "BIGINT", 34),
    ("complaint314", "BIGINT", 35),
    ("complaint594", "BIGINT", 36),
    ("complaint274", "BIGINT", 37),
    ("complaint614", "BIGINT", 38),
    ("complaint654", "BIGINT", 39),
    ("complaint234", "BIGINT", 40),
    ("complaint174", "BIGINT", 41),
    ("updated_at", "TIMESTAMPTZ", 42),
]

# ============================================================
# CAS_INFO Schema
# ============================================================
CAS_INFO_COLUMNS: List[Tuple[str, str, int]] = [
    ("cms_code_enc", "VARCHAR(100)", 1),
    ("crm_code_enc", "VARCHAR(100)", 2),
    ("cus_province", "BIGINT", 3),
    ("contract_service", "BIGINT", 4),
    ("tenure", "BIGINT", 5),
    ("custype", "BIGINT", 6),
    ("customer_update_date", "TIMESTAMPTZ", 7),
    ("contract_classify", "BIGINT", 8),
    ("contract_sig_first", "TIMESTAMPTZ", 9),
    ("contract_mgr_org", "BIGINT", 10),
    ("cus_poscode", "BIGINT", 11),
]

# ============================================================
# Mapping function
# ============================================================
TABLE_SCHEMAS: Dict[str, List[Tuple[str, str, int]]] = {
    "bccp_orderitem": BCCP_ORDERITEM_COLUMNS,
    "cms_complaint": CMS_COMPLAINT_COLUMNS,
    "cas_customer": CAS_CUSTOMER_COLUMNS,
    "cas_info": CAS_INFO_COLUMNS,
}


def get_table_schema(table_base: str) -> List[Tuple[str, str, int]]:
    """
    Lấy schema của bảng.
    
    Args:
        table_base: "bccp_orderitem", "cms_complaint", "cas_customer", "cas_info"
    
    Returns:
        List của (column_name, data_type, position)
    """
    if table_base not in TABLE_SCHEMAS:
        raise ValueError(f"Unknown table: {table_base}")
    return TABLE_SCHEMAS[table_base]


def get_canonical_column_names(table_base: str) -> List[str]:
    """Lấy danh sách tên cột chính thức theo thứ tự."""
    schema = get_table_schema(table_base)
    return [col_name for col_name, _, _ in schema]


def get_column_datatype(table_base: str, column_name: str) -> str:
    """Lấy kiểu dữ liệu của 1 cột."""
    schema = get_table_schema(table_base)
    for col, dtype, _ in schema:
        if col.lower() == column_name.lower():
            return dtype
    raise ValueError(f"Column {column_name} not found in {table_base}")


def get_column_by_position(table_base: str, position: int) -> Tuple[str, str]:
    """Lấy tên cột và kiểu theo vị trí (1-indexed)."""
    schema = get_table_schema(table_base)
    for col_name, dtype, pos in schema:
        if pos == position:
            return (col_name, dtype)
    raise ValueError(f"Position {position} not found in {table_base}")


def get_prod_table_ddl(table_base: str, table_name: str, prod_schema: str = "public") -> str:
    """
    Generate CREATE TABLE statement cho production table với đúng data types.
    
    Args:
        table_base: "bccp_orderitem", "cms_complaint", "cas_customer", "cas_info"
        table_name: Full table name (e.g., "public.bccp_orderitem")
        prod_schema: Schema name (default: "public")
    
    Returns:
        CREATE TABLE SQL statement
    """
    schema = get_table_schema(table_base)
    
    # Build column definitions
    cols = [f'"{col_name}" {dtype}' for col_name, dtype, _ in schema]
    
    col_list = ",\n    ".join(cols)
    
    return f"""CREATE TABLE IF NOT EXISTS {table_name} (
    {col_list}
);"""
