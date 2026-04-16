"""Database connection and source validation helpers for feature generation."""

import os

from sqlalchemy import create_engine
from sqlalchemy import text

from libs.database import PostgresConfig
from libs.db_utils import ensure_public_table_columns_exist
from libs.db_utils import ensure_public_tables_exist
from logging_config import get_logger

logger = get_logger("feature_generation_database")

REQUIRED_COLUMNS = {
    "cas_customer": ["cms_code_enc", "item_count"],
    "cms_complaint": ["cms_code_enc"],
    "cas_info": ["cms_code_enc"],
}


def build_database_url(database_url: str | None = None) -> str:
    if database_url:
        logger.info(f"Using provided database URL")
        return database_url

    env_database_url = os.environ.get("DATABASE_URL")
    if env_database_url:
        logger.info(f"Using DATABASE_URL from environment")
        return env_database_url

    cfg = PostgresConfig.from_env()
    url = f"postgresql+psycopg2://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.dbname}"
    logger.info(f"Built database URL: postgresql+psycopg2://{cfg.user}:***@{cfg.host}:{cfg.port}/{cfg.dbname}")
    return url


def create_pipeline_engine(database_url: str | None = None):
    url = build_database_url(database_url)
    logger.info("Establishing database connection...")
    try:
        engine = create_engine(url, echo=False)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("[OK] Database connection successful")
        return engine
    except Exception as e:
        logger.error(f"[FAILED] Failed to connect to database: {e}")
        raise


def validate_source_tables(engine) -> None:
    logger.info("Checking required source tables...")
    try:
        ensure_public_tables_exist(engine)
        logger.info("[OK] All required source tables found")
        
        logger.info(f"Validating columns in {len(REQUIRED_COLUMNS)} tables...")
        ensure_public_table_columns_exist(engine, REQUIRED_COLUMNS)
        logger.info("[OK] All required columns validated")
    except Exception as e:
        logger.error(f"[FAILED] Source table validation failed: {e}")
        raise
