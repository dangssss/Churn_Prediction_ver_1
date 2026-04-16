from __future__ import annotations

import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

def get_engine(database_url: str | None = None, pool_pre_ping: bool = True) -> Engine:
    # Prioritize PG_ variables to match Ingestion exactly
    pg_host = os.getenv("PG_HOST")
    if pg_host:
        pg_port = os.getenv("PG_PORT", "25432")
        pg_user = os.getenv("PG_USER", "cpuser")
        pg_password = os.getenv("PG_PW", "cp123456")
        pg_db = os.getenv("PG_DB", "churn_prediction")
        url = f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    else:
        # Fallback to DATABASE_URL or hardcoded IP
        url = database_url or os.getenv("DATABASE_URL")
        if not url:
            url = "postgresql+psycopg2://cpuser:cp123456@172.16.2.142:25432/churn_prediction"
            
    print(f"DEBUG: SQLAlchemy engine connecting to target DB: {url.replace('cp123456', '***')}")
    return create_engine(url, pool_pre_ping=pool_pre_ping, pool_size=10, max_overflow=20)

def smoke_test(engine: Engine) -> tuple:
    """Return (current_database, current_user, version)."""
    with engine.connect() as conn:
        return conn.execute(text("select current_database(), current_user, version()")).fetchone()
