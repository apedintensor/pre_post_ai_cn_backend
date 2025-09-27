"""Add investigation_action and next_step_action columns to assessments if missing.

Run locally:
  python -m scripts.add_assessment_actions_columns
"""
from sqlalchemy import inspect, text
from app.db.session import sync_engine as engine


def column_exists(conn, table: str, column: str) -> bool:
    insp = inspect(conn)
    cols = [c["name"] for c in insp.get_columns(table)]
    return column in cols


def main():
    with engine.begin() as conn:
        if not column_exists(conn, "assessments", "investigation_action"):
            conn.execute(text("ALTER TABLE assessments ADD COLUMN investigation_action VARCHAR"))
            print("Added assessments.investigation_action")
        else:
            print("assessments.investigation_action already exists")

        if not column_exists(conn, "assessments", "next_step_action"):
            conn.execute(text("ALTER TABLE assessments ADD COLUMN next_step_action VARCHAR"))
            print("Added assessments.next_step_action")
        else:
            print("assessments.next_step_action already exists")


if __name__ == "__main__":
    main()
