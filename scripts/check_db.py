"""Test connection đến Supabase và liệt kê các bảng đã tạo.

Chạy: python -m scripts.check_db
"""
from __future__ import annotations

import io
import sys

from src.db import get_conn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


EXPECTED_TABLES = {"videos", "frames", "frame_features", "search_logs"}


def main() -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"[OK] Connected: {version.split(',')[0]}")

        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
            """
        )
        tables = {row[0] for row in cur.fetchall()}
        print(f"[OK] Public tables: {sorted(tables)}")

        missing = EXPECTED_TABLES - tables
        if missing:
            print(f"[WARN] Thieu bang: {sorted(missing)}")
            return 1

        for t in sorted(EXPECTED_TABLES):
            cur.execute(f"SELECT count(*) FROM {t};")
            print(f"  {t:18s} {cur.fetchone()[0]:>6} rows")

    print("[OK] Schema khop voi schema.sql")
    return 0


if __name__ == "__main__":
    sys.exit(main())
