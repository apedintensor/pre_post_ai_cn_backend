"""Add users.country_code (VARCHAR(2)) if missing.

Run locally:
  python -m scripts.add_country_column
"""
from sqlalchemy import inspect, text
from app.db.session import sync_engine as engine


def column_exists(conn, table: str, column: str) -> bool:
    insp = inspect(conn)
    cols = [c["name"] for c in insp.get_columns(table)]
    return column in cols


def main():
    with engine.begin() as conn:
        if not column_exists(conn, "users", "country_code"):
            conn.execute(text("ALTER TABLE users ADD COLUMN country_code VARCHAR(2)"))
            print("Added users.country_code")
        else:
            print("users.country_code already exists")


if __name__ == "__main__":
    main()
