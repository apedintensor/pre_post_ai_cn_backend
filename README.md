# Reader Study Web Backend

FastAPI + SQLAlchemy service powering the Reader Study application (gameplay blocks, assessments, AI output surfacing, role/user auth & reporting).

---
## Key Features

- FastAPI with async & sync SQLAlchemy sessions
- fastapi-users JWT auth (registration, login, reset, verify, user management)
- Role-based access model
- Game workflow: block assignments, PRE/POST assessments, block feedback
- AI model probability ingestion + top‑K extraction
- Idempotent data seeding & full import scripts
- Dual route mounting (supports both `/roles/` and `/api/roles/` during transition)
- Containerized deploy on Fly.io (Uvicorn entrypoint)

---
## Project Layout

```
app/
   api/            # Aggregated API routers + endpoint modules
   auth/           # fastapi-users integration (models, routes, manager)
   core/           # Settings, exceptions, rate limiter
   crud/           # CRUD helper classes
   db/             # Engines, sessions, init_db
   models/         # SQLAlchemy mapped classes
   schemas/        # Pydantic models
   services/       # Domain logic (assessment/game/metrics)
   main.py         # App factory & router mounting
scripts/          # Data import & maintenance scripts
data/             # CSV / JSON reference data (terms, AI predictions)
docs/             # Extended design & workflow docs
Dockerfile        # Multi-stage build
pyproject.toml    # Project + deps (managed by uv / poetry style lock)
```

---
## Environment & Configuration

Configuration is centralized in `app/core/config.py` (Pydantic Settings v2). Important variables:

| Variable | Purpose | Example |
|----------|---------|---------|
| `DATABASE_URL` | Primary DB URL (PostgreSQL or SQLite fallback) | `postgresql://user:pass@host/db` |
| `SECRET_KEY` | JWT & signing secret | (generate 32+ chars) |
| `SUPERUSER_EMAIL` | Bootstrap superuser email | `admin@example.com` |
| `SUPERUSER_PASSWORD` | Bootstrap superuser password | `change-me` |
| `IMAGE_BASE_URL` | Optional base path/URL for image hosting | `/images/` |
| `GAME_BLOCK_SIZE` | Cases per game block | `10` |

If `DATABASE_URL` starts with `postgres://` or `postgresql://` it is adapted to async/sync URLs automatically (`asyncpg` + `psycopg2`). Otherwise SQLite file `test.db` is used.

---
## Running Locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .            # or `uv sync` if using uv
uvicorn app.main:app --reload
```

Docs: http://localhost:8000/docs

---
## Dual Route Mounting (Transition Aid)

Routers are mounted twice to allow both legacy and prefixed calls:

| Path Form | Example | Notes |
|-----------|---------|-------|
| Unprefixed | `/roles/` `/auth/register` `/game/active` | Shown in OpenAPI docs |
| Prefixed   | `/api/roles/` `/api/auth/register` `/api/game/active` | Hidden from docs (`include_in_schema=False`) |

Future: introduce versioning (`/api/v1`) then deprecate unprefixed routes.

---
## Authentication Endpoints

| Action | Method / Path |
|--------|---------------|
| Register | `POST /auth/register` (or `/api/auth/register`) |
| Login (JWT) | `POST /auth/jwt/login` |
| Current User | `GET /auth/me` |
| Reset Password | `POST /auth/reset-password/request` etc. |
| Verify Email | `POST /auth/verify` |

fastapi-users standard responses apply. Include Authorization header: `Bearer <token>` after login.

---
## Game & Assessment (High Level)

See `docs/game_workflow.md` and `docs/frontend_update.md` for detailed contracts.

Typical sequence:
1. Start or fetch active block (`/game/start` or `/game/active`).
2. For each assignment: submit PRE, reveal AI, submit POST.
3. Fetch report card (`/game/report/{block_index}`).

---
## Data Import & Seeding

Scripts live in `scripts/`:

| Script | Purpose |
|--------|---------|
| `import_initial_data.py` | Argument-driven full import (roles, terms, synonyms, cases, images, AI outputs). |
| `reset_and_import_postgres.py` | Drops & recreates schema then runs import (Postgres). |
| `seed_basic.py` | Zero-argument idempotent seed (uses JSON + CSV defaults). |

Primary data files:
* `data/derm_dictionary.json` – term ids + canonical + synonyms/abbreviations.
* `data/ai_prediction_by_id.csv` – cases with probability vector columns.

See `docs/data_import.md` for full rules (top‑3 extraction, idempotency, edge cases).

---
## Fly.io Deployment

Dockerfile uses multi-stage build and pinned Python 3.13. Deployment basics:

```bash
fly launch            # one-time (already done)
fly secrets set DATABASE_URL=postgresql://... SECRET_KEY=... SUPERUSER_EMAIL=... SUPERUSER_PASSWORD=...
fly deploy            # builds & releases new image
```

Updating secrets (`fly secrets set ...`) triggers a new release & machine restart automatically; no manual restart required.

### Health / Troubleshooting
Common startup issues:
| Symptom | Cause | Fix |
|---------|-------|-----|
| `TypeError: connect() got an unexpected keyword argument 'check_same_thread'` | SQLite-only arg passed to Postgres | Resolved in `app/db/session.py` by conditional connect args. |
| Repeated restarts with PydanticUserError | Dynamic Settings fields not annotated | Fixed by restructuring `config.py`. |
| 404 on `/api/roles/` | Router lacked prefix | Dual mount added (`/roles/` + `/api/roles/`). |

To force a recycle without code change: `fly machines restart <id>` (or redeploy).

---
## Testing

Run pytest (ensure `.env` or SQLite fallback):
```bash
pytest -q
```
Tests may construct a SQLite in-memory/file DB; see `tests/conftest.py`.

---
## Extensibility Roadmap

- Add `/healthz` lightweight endpoint for external monitors
- Introduce `/api/v1` versioned prefix and deprecate unprefixed routes
- Alembic migrations instead of metadata create_all
- Sentry (add `sentry-sdk[fastapi]` + init early in `main.py` if DSN provided)
- Background percentile computation job populating `peer_percentile_*`

---
## Quick Reference

| Task | Command |
|------|---------|
| Local dev | `uvicorn app.main:app --reload` |
| Basic seed | `python scripts/seed_basic.py` |
| Full import | `python scripts/import_initial_data.py --terms data/derm_dictionary.csv --cases data/ai_prediction_by_id.csv` |
| Reset + import (danger) | `python scripts/reset_and_import_postgres.py --terms ... --cases ...` |
| Run tests | `pytest -q` |
| Deploy | `fly deploy` |

---
## License

MIT

---
## Change Log (Recent)

| Date | Change |
|------|--------|
| 2025-09 | Dual route mounting, session engine fix, config refactor for Pydantic v2. |
| 2025-09 | Added `seed_basic.py` zero-arg seeding. |
| 2025-09 | Dockerfile optimized (multi-stage + uvicorn entry). |

---
Feel free to open issues or extend docs under `docs/` for new workflows.