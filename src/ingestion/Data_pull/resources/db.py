import os
from dataclasses import dataclass
import psycopg2

@dataclass
class PostgresConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        """
        Read DB config from environment variables.
        Defaults match the Docker Compose setup.
        """
        return cls(
            host=os.getenv("PG_HOST", "172.20.0.65"),
            port=int(os.getenv("PG_PORT", "5433")),
            dbname=os.getenv("PG_DB", "cpuser"),
            user=os.getenv("PG_USER", "cpuser"),
            password=os.getenv("PG_PW", "db12345"),
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
    Get a psycopg2 connection from PostgresConfig.
    """
    conn = psycopg2.connect(cfg.dsn())
    conn.autocommit = autocommit
    return conn
