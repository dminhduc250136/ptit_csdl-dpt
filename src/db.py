"""Connection layer cho Supabase PostgreSQL.

Dùng psycopg2 + context manager để đảm bảo connection luôn được close,
kể cả khi có exception. Load DATABASE_URL từ .env.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2.extensions import connection as PgConnection
from dotenv import load_dotenv

load_dotenv()

_DATABASE_URL = os.getenv("DATABASE_URL")
if not _DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL chưa được set. Copy .env.example sang .env và điền connection string."
    )


@contextmanager
def get_conn() -> Iterator[PgConnection]:
    """Mở connection mới, commit khi thoát bình thường, rollback khi có exception."""
    conn = psycopg2.connect(_DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
