from __future__ import annotations

from pathlib import Path

import psycopg


def apply_migrations(postgres_dsn: str, migrations_dir: Path | None = None) -> list[str]:
    base_dir = migrations_dir or (Path(__file__).resolve().parent / "migrations")
    migration_files = sorted(base_dir.glob("*.sql"))
    applied: list[str] = []

    with psycopg.connect(postgres_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    migration_id TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute("SELECT migration_id FROM schema_migrations")
            already_applied = {row[0] for row in cur.fetchall()}

            for file_path in migration_files:
                migration_id = file_path.name
                if migration_id in already_applied:
                    continue
                sql = file_path.read_text(encoding="utf-8")
                cur.execute(sql)
                cur.execute("INSERT INTO schema_migrations (migration_id) VALUES (%s)", (migration_id,))
                applied.append(migration_id)
        conn.commit()

    return applied

