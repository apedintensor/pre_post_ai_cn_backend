import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.models.models import BlockFeedback


@pytest.mark.asyncio
async def test_end_to_end_workflow(client: AsyncClient):
    """Minimal end-to-end smoke test covering register, login, start block, suggestions, PRE/POST assessments, and report generation."""
    # Register (idempotent)
    resp = await client.post("/auth/register", json={"email": "dan@123.com", "password": "123456"})
    assert resp.status_code in (200, 201, 400)
    # Login
    resp = await client.post("/auth/jwt/login", data={"username": "dan@123.com", "password": "123456"})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Start game
    resp = await client.post("/game/start", headers=headers)
    assert resp.status_code == 200, resp.text
    assignment = resp.json()["assignments"][0]

    # Suggestions
    resp = await client.get("/diagnosis_terms/suggest", params={"q": "ca"}, headers=headers)
    assert resp.status_code == 200, resp.text
    suggestions = resp.json()

    chosen = []
    seen = set()
    for s in suggestions:
        if s["id"] not in seen:
            seen.add(s["id"])
            chosen.append(s)
        if len(chosen) == 3:
            break
    # Create extra terms if needed
    extra_names = ["SmokeTermA", "SmokeTermB", "SmokeTermC"]
    idx = 0
    while len(chosen) < 3 and idx < len(extra_names):
        create = await client.post("/diagnosis_terms/", json={"name": extra_names[idx]})
        if create.status_code in (200, 201):
            new = create.json()
            if new["id"] not in seen:
                seen.add(new["id"])
                chosen.append({"id": new["id"], "label": new["name"], "match": new["name"], "source": "created"})
        idx += 1
    assert chosen, "No terms available"
    while len(chosen) < 3:
        chosen.append(chosen[0])  # allow duplicates for smoke test only

    # PRE assessment
    pre_payload = {
        "assignment_id": assignment["id"],
        "phase": "PRE",
        "diagnostic_confidence": 3,
        "management_confidence": 3,
        "investigation_action": "NONE",
        "next_step_action": "REASSURE",
        "diagnosis_entries": [
            {"rank": i + 1, "raw_text": c["match"], "diagnosis_term_id": c["id"]}
            for i, c in enumerate(chosen)
        ],
    }
    resp = await client.post("/assessment/", headers=headers, json=pre_payload, follow_redirects=True)
    assert resp.status_code in (200, 201), resp.text

    # POST assessment (reverse order)
    post_entries = list(reversed(pre_payload["diagnosis_entries"]))
    for i, e in enumerate(post_entries):
        e["rank"] = i + 1
    post_payload = {
        "assignment_id": assignment["id"],
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
    resp = await client.post("/assessment/", headers=headers, json=post_payload, follow_redirects=True)
    assert resp.status_code in (200, 201), resp.text

    # Report (finalize if necessary)
    report = await client.get("/game/report/0", headers=headers)
    assert report.status_code == 200, report.text
    data = report.json()
    # Ensure new peer average fields exist
    assert "peer_avg_top1_pre" in data
    assert data.get("block_index", 0) == 0
