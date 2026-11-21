"""
SQLite connection helpers (Phase 1 foundation).

Design choice: a NEW short-lived connection per operation. SQLite opens in
microseconds, and this avoids cross-thread problems inside FastAPI.
All queries live in app/db/repository.py (Member 2).
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

from app.config import settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
PathLike = Union[str, Path, None]


def get_connection(db_path: PathLike = None) -> sqlite3.Connection:
    """Open a connection with dict-like rows and foreign keys enabled."""
    path = Path(db_path) if db_path else Path(settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")   # readers don't block the writer
    return conn


@contextmanager
def db_session(db_path: PathLike = None) -> Iterator[sqlite3.Connection]:
    """Usage:
        with db_session() as conn:
            conn.execute(...)
    Commits on success, rolls back on error, always closes."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: PathLike = None) -> None:
    """Create all tables (idempotent)."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with db_session(db_path) as conn:
        conn.executescript(sql)


def reset_db(db_path: PathLike = None) -> None:
    """Delete the database file and recreate an empty schema (dev/tests only)."""
    path = Path(db_path) if db_path else Path(settings.db_path)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()
    init_db(path)


def fts5_available() -> bool:
    """True if this Python's SQLite supports FTS5 keyword search."""
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


def table_counts(db_path: PathLike = None) -> Dict[str, int]:
    """Row count per main table - used by /health and setup checks."""
    tables = ["products", "customers", "orders", "sessions", "audit_log"]
    counts: Dict[str, int] = {}
    with db_session(db_path) as conn:
        for table in tables:
            try:
                counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                counts[table] = -1   # table missing -> run init_db()
    return counts
