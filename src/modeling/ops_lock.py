from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import psycopg2

LOCK_NAME = "ds_churn_pipeline_exclusive"


def _connect():
    return psycopg2.connect(
        host=os.environ["PG_HOST"],
        port=os.getenv("PG_PORT", "25432"),
        dbname=os.getenv("PG_DB", "churn_prediction"),
        user=os.getenv("PG_USER", "cpuser"),
        password=os.environ["PG_PW"],
    )


def _try_lock(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (LOCK_NAME,))
        return bool(cur.fetchone()[0])


def _unlock(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_unlock(hashtext(%s))", (LOCK_NAME,))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a churn pipeline command while holding a shared Postgres advisory lock."
    )
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--skip-if-busy", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command after --")

    conn = _connect()
    conn.autocommit = True
    deadline = time.monotonic() + max(int(args.wait_seconds), 0)
    acquired = _try_lock(conn)
    while not acquired and time.monotonic() < deadline:
        time.sleep(5)
        acquired = _try_lock(conn)

    if not acquired:
        message = f"Pipeline lock {LOCK_NAME!r} is busy."
        if args.skip_if_busy:
            print(f"SKIP: {message}")
            return 0
        print(f"ERROR: {message}", file=sys.stderr)
        return 75

    try:
        print(f"Acquired pipeline lock {LOCK_NAME!r}; running: {command}")
        return subprocess.run(command, check=False).returncode
    finally:
        try:
            _unlock(conn)
        finally:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
