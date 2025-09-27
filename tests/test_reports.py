import pytest
from httpx import AsyncClient, ASGITransport

@pytest.mark.asyncio
async def test_block_report_listing():
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Register/login
        r = await client.post("/auth/register", json={"email": "report@test.com", "password": "pw123456"})
        assert r.status_code in (200,201,400)
        r = await client.post("/auth/jwt/login", data={"username": "report@test.com", "password": "pw123456"})
        assert r.status_code == 200
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        # Start block
        r = await client.post("/game/start", headers=headers)
        assert r.status_code == 200, r.text
        data = r.json()
        assignments = data["assignments"]
        # Ensure at least one diagnosis term exists (try suggestion first)
        term_id = None
        s = await client.get("/diagnosis_terms/suggest", params={"q": "a"}, headers=headers)
        if s.status_code == 200 and s.json():
            term_id = s.json()[0]["id"]
        if term_id is None:
            created = await client.post("/diagnosis_terms/", json={"name": "TempTermX"})
            assert created.status_code in (200,201), created.text
            term_id = created.json()["id"]

        # For each assignment do PRE+POST quickly
        for a in assignments:
            aid = a["id"]
            # minimal 3 diagnosis entries using same id repeated (test purpose)
            entries = [
                {"rank": 1, "raw_text": "x", "diagnosis_term_id": term_id},
                {"rank": 2, "raw_text": "x", "diagnosis_term_id": term_id},
                {"rank": 3, "raw_text": "x", "diagnosis_term_id": term_id},
            ]
            pre_payload = {
                "assignment_id": aid,
                "phase": "PRE",
                "diagnostic_confidence": 3,
                "management_confidence": 3,
                "investigation_action": "NONE",
                "next_step_action": "REASSURE",
                "diagnosis_entries": entries,
            }
            r = await client.post("/assessment/", headers=headers, json=pre_payload)
            assert r.status_code in (200,201), r.text
            post_entries = list(reversed(entries))
            for i,e in enumerate(post_entries):
                e["rank"] = i+1
            post_payload = {
                "assignment_id": aid,
                "phase": "POST",
                "diagnostic_confidence": 4,
                "management_confidence": 4,
                "investigation_action": "BIOPSY",
                "next_step_action": "REFER",
                "changed_primary_diagnosis": False,
                "changed_management_plan": False,
                "ai_usefulness": "Useful",
                "diagnosis_entries": post_entries,
            }
            r = await client.post("/assessment/", headers=headers, json=post_payload)
            assert r.status_code in (200,201), r.text
        # Request report listing (should finalize lazily without 500)
        r = await client.get("/game/reports", headers=headers)
        assert r.status_code == 200, r.text
        rows = r.json()
        assert isinstance(rows, list)
        # At least one block feedback row
        assert len(rows) >= 1
        row = rows[0]
        assert "peer_avg_top1_pre" in row
        assert row.get("block_index") == 0
