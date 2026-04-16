"""
- Bảng cus_risk_X là SNAPSHOT (mỗi lần chạy sẽ upsert theo cms_code_enc).
"""

from __future__ import annotations

from pathlib import Path
from sqlalchemy import text
from sqlalchemy.engine import Engine


def ensure_risk_table_schema(engine: Engine, risk_threshold: float = 90) -> str:
    risk_pct = int(risk_threshold)
    table_name = f"cus_risk_{risk_pct}"

    # Load SQL template from file
    sql_dir = Path(__file__).parent / "sql"
    sql_file = sql_dir / "cus_risk_template.sql"
    
    if not sql_file.exists():
        raise FileNotFoundError(f"SQL template not found: {sql_file}")
    
    ddl = sql_file.read_text()
    ddl = ddl.replace("{THRESHOLD}", str(risk_pct))
    ddl = ddl.replace("{TABLE_NAME}", table_name)

    with engine.begin() as conn:
        for stmt in ddl.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))

    return table_name
