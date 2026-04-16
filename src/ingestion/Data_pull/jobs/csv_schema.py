# config/csv_schema.py
from typing import Set, Dict

# ============================================================
# SCHEMA DEFINITIONS FOR 4 TABLES (Dynamic Header Support)
# ============================================================
# Note: Headers are now read directly from CSV files (first row)
# These configs only define TEXT_COLS, DATETIME_COLS and MODE for each table

# === TABLE CONFIGURATION ===
TABLE_CONFIG: Dict[str, Dict] = {
    "bccp_orderitem": {
        "text_cols": {"crm_code", "cms_code_enc", "item_code", "service_code", "country_code", "region"},
        "mode": "snapshot",  # truncate on every load
        "datetime_cols": {"sending_time", "ending_time", "casreport_update_date"},
    },
    "cms_complaint": {
        "text_cols": {"cms_code", "crm_code", "item_code", "complaint_content"},
        "mode": "snapshot",  # truncate on every load
        "datetime_cols": {"create_complaint_date", "exp_complaint_date", "close_complaint_date", "complaint_update_date"},
    },
    "cas_customer": {
        "text_cols": {"cms_code_enc"},
        "mode": "snapshot",  # truncate on every load
        "datetime_cols": {"customer_update_date"},
    },
    "cas_info": {
        "text_cols": {"cms_code_enc", "crm_code"},
        "mode": "snapshot",  # truncate on every load
        "datetime_cols": {"contract_service", "contract_sig_first"},
    },
}

# === CSV OPTIONS ===
# CSV source CÓ HEADER (từ đầu data warehouse)
SOURCE_HAS_HEADER: bool = True

# Guard CSV injection (sanitize mode: prepend ' to dangerous cells)
CSV_INJECTION_GUARD: str = "sanitize"

# chunk size đọc CSV
BATCH_ROWS: int = 50_000


def get_table_config(table_base: str) -> Dict:
    """
    Lấy config cho table_base.
    table_base: "bccp_orderitem", "cms_complaint", "cas_customer", "cas_info"
    """
    if table_base not in TABLE_CONFIG:
        raise ValueError(f"Unknown table: {table_base}")
    return TABLE_CONFIG[table_base]


def get_text_cols(table_base: str) -> Set[str]:
    """Lấy text columns cần sanitize cho bảng."""
    return get_table_config(table_base).get("text_cols", set())


def get_datetime_cols(table_base: str) -> Set[str]:
    """Lấy datetime columns cần normalize cho bảng."""
    return get_table_config(table_base).get("datetime_cols", set())


def get_mode(table_base: str) -> str:
    """Lấy mode (monthly/snapshot) cho bảng."""
    return get_table_config(table_base).get("mode", "monthly")


