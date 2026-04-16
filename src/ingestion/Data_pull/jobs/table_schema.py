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
    ("crm_code_enc", "VARCHAR(20)", 1),
    ("cms_code_enc", "VARCHAR(20)", 2),
    ("item_code", "VARCHAR(20)", 3),
    ("service_code", "VARCHAR(10)", 4),
    ("weight_kg", "DECIMAL(10,3)", 5),
    ("length_size", "INT", 6),
    ("width_size", "INT", 7),
    ("height_size", "INT", 8),
    ("total_fee", "INT", 9),
    ("is_domestic", "INT", 10),
    ("country_code", "VARCHAR(20)", 11),
    ("send_province_code", "INT", 12),
    ("send_district_code", "INT", 13),
    ("send_commune_code", "INT", 14),
    ("rec_province_code", "INT", 15),
    ("rec_district_code", "INT", 16),
    ("rec_commune_code", "INT", 17),
    ("region", "VARCHAR(20)", 18),
    ("sending_time", "DATETIME", 19),
    ("ending_time", "DATETIME", 20),
    ("rec_success", "INT", 21),
    ("refunded", "INT", 22),
    ("no_accepted", "INT", 23),
    ("lost_order", "INT", 24),
    ("delay_day", "INT", 25),
    ("done", "INT", 26),
    ("total_complaint", "INT", 27),
    ("complaint114", "INT", 28),
    ("complaint115", "INT", 29),
    ("complaint116", "INT", 30),
    ("complaint134", "INT", 31),
    ("complaint194", "INT", 32),
    ("complaint554", "INT", 33),
    ("complaint595", "INT", 34),
    ("complaint314", "INT", 35),
    ("complaint594", "INT", 36),
    ("complaint274", "INT", 37),
    ("complaint614", "INT", 38),
    ("complaint654", "INT", 39),
    ("complaint234", "INT", 40),
    ("complaint174", "INT", 41),
    ("order_score", "DECIMAL(10,3)", 42),
    ("bccp_update_date", "DATETIME", 43),
]

# ============================================================
# CMS_COMPLAINT Schema
# ============================================================
CMS_COMPLAINT_COLUMNS: List[Tuple[str, str, int]] = [
    ("cms_code_enc", "VARCHAR(20)", 1),
    ("item_code", "VARCHAR(20)", 2),
    ("create_complaint_date", "DATETIME", 3),
    ("exp_complaint_date", "DATETIME", 4),
    ("close_complaint_date", "DATETIME", 5),
    ("delay_complaint", "INT", 6),
    ("complaint_code", "INT", 7),
    ("complaint_content", "TEXT", 8),
    ("complaint_content_bit", "INT", 9),
    ("complaint_update_date", "DATETIME", 10),
]

# ============================================================
# CAS_CUSTOMER Schema
# ============================================================
CAS_CUSTOMER_COLUMNS: List[Tuple[str, str, int]] = [
    ("cms_code_enc", "VARCHAR(20)", 1),
    ("report_month", "DATE", 2),  # YYYY-MM-DD format
    ("item_count", "BIGINT", 3),
    ("weight_kg", "DECIMAL(12,3)", 4),
    ("total_fee", "BIGINT", 5),
    ("intra_province", "INT", 6),
    ("international", "INT", 7),
    ("ser_c", "INT", 8),
    ("ser_e", "INT", 9),
    ("ser_m", "INT", 10),
    ("ser_p", "INT", 11),
    ("ser_r", "INT", 12),
    ("ser_u", "INT", 13),
    ("ser_l", "INT", 14),
    ("ser_q", "INT", 15),
    ("delay_day", "INT", 16),
    ("delay_count", "INT", 17),
    ("nodone", "INT", 18),
    ("refunded", "INT", 19),
    ("noaccepted", "INT", 20),
    ("lost_order", "INT", 21),
    ("lastday", "INT", 22),
    ("noservice", "INT", 23),
    ("dev_item", "DECIMAL(10,3)", 24),
    ("order_score", "DECIMAL(10,3)", 25),
    ("satisfaction_score", "DECIMAL(10,3)", 26),
    ("total_complaint", "INT", 27),
    ("complaint114", "INT", 28),
    ("complaint115", "INT", 29),
    ("complaint116", "INT", 30),
    ("complaint134", "INT", 31),
    ("complaint194", "INT", 32),
    ("complaint554", "INT", 33),
    ("complaint595", "INT", 34),
    ("complaint314", "INT", 35),
    ("complaint594", "INT", 36),
    ("complaint274", "INT", 37),
    ("complaint614", "INT", 38),
    ("complaint654", "INT", 39),
    ("complaint234", "INT", 40),
    ("complaint174", "INT", 41),
    ("updated_at", "DATETIME", 42),
]

# ============================================================
# CAS_INFO Schema
# ============================================================
CAS_INFO_COLUMNS: List[Tuple[str, str, int]] = [
    ("cms_code_enc", "VARCHAR(20)", 1),
    ("crm_code_enc", "VARCHAR(20)", 2),
    ("cus_province", "INT", 3),
    ("contract_service", "DATETIME", 4),
    ("contract_sig_first", "DATETIME", 5),
    ("tenure", "INT", 6),
    ("custype", "INT", 7),
    ("contract_classify", "INT", 8),
    ("contract_mgr_org", "INT", 9),
    ("cus_poscode", "INT", 10),
    ("customer_update_date", "DATETIME", 11),
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
    
    Example:
        >>> ddl = get_prod_table_ddl("cms_complaint", "cms_complaint", "public")
        >>> print(ddl)
        CREATE TABLE IF NOT EXISTS public.cms_complaint (
            "cms_code_enc" VARCHAR(20),
            "item_code" VARCHAR(20),
            ...
        );
    """
    schema = get_table_schema(table_base)
    
    # Build column definitions
    cols = [f'"{col_name}" {dtype}' for col_name, dtype, _ in schema]
    col_list = ",\n    ".join(cols)
    
    return f"""CREATE TABLE IF NOT EXISTS {table_name} (
    {col_list}
);"""
