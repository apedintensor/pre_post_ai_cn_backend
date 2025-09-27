"""Smoke script: seed minimal mock data, play through one block, generate block report.
Run: python scripts/generate_report_smoke.py
"""
from fastapi.testclient import TestClient
from datetime import datetime
import time
from app.main import app
from app.core.config import settings
from app.db.session import SessionLocal
from app.models.models import DiagnosisTerm, Case

client = TestClient(app)

def ensure_terms_and_cases():
    db = SessionLocal()
    try:
        # Ensure at least 3 diagnosis terms
        terms = db.query(DiagnosisTerm).limit(3).all()
        missing = 3 - len(terms)
        for i in range(missing):
            t = DiagnosisTerm(name=f"MockTerm{i+1}_{int(time.time())}")
            db.add(t)
        if missing:
            db.commit()
            terms = db.query(DiagnosisTerm).limit(3).all()
        assert len(terms) >= 3
        ground_id = terms[0].id
        block_size = max(1, settings.GAME_BLOCK_SIZE)
        # Ensure enough cases referencing ground truth term
        existing_cases = db.query(Case).count()
        needed = block_size - existing_cases
        for i in range(needed):
            c = Case(ground_truth_diagnosis_id=ground_id, created_at=datetime.utcnow())
            db.add(c)
        if needed > 0:
            db.commit()
    finally:
        db.close()


def register_and_login():
    email = f"mock_report_{int(time.time())}@example.com"
    password = "pass1234"
    r = client.post("/auth/register", json={"email": email, "password": password})
    # 400 if already exists (unlikely with timestamped email) is fine
    assert r.status_code in (200,201,400), r.text
    r = client.post("/auth/jwt/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def play_block(headers):
    # Start or resume
    r = client.post("/game/start", headers=headers)
    assert r.status_code == 200, r.text
    block_index = r.json()["block_index"]
    assignments = r.json()["assignments"]
    # Fetch 3 suggestion terms (fallback use existing terms ids 1..3)
    sug = client.get("/diagnosis_terms/suggest", params={"q": "m"}, headers=headers)
    suggestion_ids = []
    if sug.status_code == 200:
        for s in sug.json():
            if s.get("id") not in suggestion_ids:
                suggestion_ids.append(s["id"])
            if len(suggestion_ids) == 3:
                break
    if len(suggestion_ids) < 3:
        # fallback brute fetch via IDs
        suggestion_ids = [1,2,3][:3]
    suggestion_ids = [sid for sid in suggestion_ids if isinstance(sid,int)]
    while len(suggestion_ids) < 3:
        suggestion_ids.append(suggestion_ids[0])

    for a in assignments:
        aid = a["id"]
        # PRE
        pre_entries = [
            {"rank": i+1, "raw_text": f"t{i}", "diagnosis_term_id": suggestion_ids[i]}
            for i in range(3)
        ]
        pre_payload = {
            "assignment_id": aid,
            "phase": "PRE",
            "diagnostic_confidence": 3,
            "management_confidence": 3,
            "investigation_action": "NONE",
            "next_step_action": "REASSURE",
            "diagnosis_entries": pre_entries,
        }
        r = client.post("/assessment/", headers=headers, json=pre_payload)
        assert r.status_code in (200,201), r.text
        # POST (reverse order)
        post_entries = list(reversed(pre_entries))
        for i,e in enumerate(post_entries):
            e["rank"] = i+1
        post_payload = {
            "assignment_id": aid,
            "phase": "POST",
            "diagnostic_confidence": 4,
            "management_confidence": 4,
            "investigation_action": "BIOPSY",
            "next_step_action": "REFER",
            "changed_primary_diagnosis": True,
            "changed_management_plan": True,
            "ai_usefulness": "Useful",
            "diagnosis_entries": post_entries,
        }
        r = client.post("/assessment/", headers=headers, json=post_payload)
        assert r.status_code in (200,201), r.text
    return block_index


def fetch_report(block_index, headers):
    r = client.get(f"/game/report/{block_index}", headers=headers)
    return r


def main():
    ensure_terms_and_cases()
    headers = register_and_login()
    block_index = play_block(headers)
    report_resp = fetch_report(block_index, headers)
    if report_resp.status_code != 200:
        print("Report fetch failed:", report_resp.status_code, report_resp.text)
    else:
        print("Report JSON:\n", report_resp.json())
    # Also list all reports
    all_r = client.get("/game/reports", headers=headers)
    print("/game/reports status", all_r.status_code)
    if all_r.status_code == 200:
        print("Reports:", all_r.json())

if __name__ == "__main__":
    main()
