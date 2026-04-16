# resources/db.py
from dataclasses import dataclass
from config.env_loader import parse_int
from config.env_loader import require_env
from config.env_loader import load_project_env_files
import psycopg2

load_project_env_files()


@dataclass
class PostgresConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        return cls(
            host=require_env("PG_HOST"),
            port=parse_int("PG_PORT"),
            dbname=require_env("PG_DB"),
            user=require_env("PG_USER"),
            password=require_env("PG_PW"),
        )

    def dsn(self) -> str:
        return (
            f"host={self.host} "
            f"port={self.port} "
            f"dbname={self.dbname} "
            f"user={self.user} "
            f"password={self.password}"
        )


def get_pg_conn(cfg: PostgresConfig, *, autocommit: bool = False):
    """
    Mở kết nối psycopg2 từ PostgresConfig.
        pg_cfg = PostgresConfig.from_env()
        conn = get_pg_conn(pg_cfg)
        cur = conn.cursor()
    """
    conn = psycopg2.connect(cfg.dsn())
    conn.autocommit = autocommit
    return conn
