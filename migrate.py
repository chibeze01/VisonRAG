from __future__ import annotations

from visionrag.config import Settings
from visionrag.db.migrate import apply_migrations


def main() -> None:
    settings = Settings.from_env()
    applied = apply_migrations(settings.postgres_dsn)
    if not applied:
        print("No new migrations.")
        return
    print("Applied migrations:")
    for item in applied:
        print(f"- {item}")


if __name__ == "__main__":
    main()

