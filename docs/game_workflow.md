Game workflow (frontend ↔ backend)

Purpose
- Document the linear 10-case game flow, resume behavior, single-active-game rule, and where/reporting metrics are computed and stored.
- Provide concrete endpoint sketches and state transitions for the frontend and backend implementers.

Update (Implemented Endpoints)
- The codebase now uses the prefix `/game` (not `/api/games`) and a unified assessment endpoint `/assessment/`.
- Added convenience & reporting endpoints (see `game_report_endpoints.md` for full schema details):
  - `GET /game/can_view_report/{block_index}` – lightweight availability probe.
  - `GET /game/report/latest` – most recent finalized block.
  - `GET /game/reports` – list all finalized reports (no pagination yet).
- The finalized per-block metrics are returned as `ReportCardResponse` objects.
- Start/resume logic is idempotent: `POST /game/start` returns the existing active block if one is in progress (no longer a 409).

Quick mapping (original sketch → implemented):
| Original Sketch | Implemented |
|-----------------|-------------|
| POST /api/games/start | POST /game/start |
| GET /api/games/active | GET /game/active |
| POST /api/assessments | POST /assessment/ |
| GET /api/games/{block_index}/report | GET /game/report/{block_index} |
| (none) | GET /game/can_view_report/{block_index} |
| (none) | GET /game/report/latest |
| (none) | GET /game/reports |

Checklist (requirements)
- Start a new game that assigns a block of 10 cases to a user.
- Allow resume: if the user exits mid-block, resume where they left off.
- Prevent starting a new game while an unfinished game exists for that user.
- When 10 cases are finished, compute per-block Top‑1/Top‑3 accuracy and peer percentile, store results in `block_feedback`, and make a report-card available.

High-level flow
1. Frontend: user clicks Start Game.
2. Frontend: send POST /api/games/start to backend.
3. Backend: checks for active (unfinished) block for user. If exists, respond 409 with active block info (frontend should offer Resume). Otherwise create a new block assignment (10 `reader_case_assignments` rows with same `block_index`), mark `started_at`, return the list of cases and assignment ids.
4. Frontend: present cases one-by-one in `display_order`. For each case the user submits their assessment, frontend POSTs assessment (or PATCH if editing) to `/api/assessments`.
5. Backend: on assessment create/update, compute `top1_correct`, `top3_correct`, and `rank_of_truth` and persist them on the `assessments` row.
6. Backend: when the 10th assessment for the user+block_index+phase is saved, run the block-finalizer to compute block metrics and persist a `block_feedback` row. Trigger peer percentile update asynchronously (background job) and return the finalized block id in the response.
7. Frontend: user sees a report card (GET /api/games/{block_index}/report or GET /api/block_feedback/{id}) with Top‑1/Top‑3 accuracy pre/post and peer percentiles.

State machine (per user)
- No active block -> user may start a new block.
- Active block (0..9 completed) -> user can resume; cannot start new block until finished.
- Completed block (10 completed) -> block is finalized and report available; user may start next block (block_index incremented).

API sketches
Note: these are implementation sketches; adapt to your routing and auth.

POST /game/start
- Purpose: start a new 10-case block for the authenticated user.
- Request: { "phase": "PRE" } (optional; default PRE)
- Responses:
  - 201 Created: { "block_index": 3, "assignments": [{"assignment_id": 1001, "case_id": 42, "display_order": 0}, ...] }
  - 409 Conflict: { "error": "active_block_exists", "block_index": 2, "incomplete": 4 }

GET /game/active
- Purpose: return current active block for user, if any.
- Response: 200: { "block_index": 2, "assignments": [{"assignment_id":..., "case_id":..., "completed":true/false, "assessment_id":...}, ...] }

POST /assessment/
- Purpose: save one assessment (pre or post) for a case/assignment.
- Body example:
  {
    "assignment_id": 1001,
    "phase": "PRE",
    "diagnosis_entries": [
      { "rank": 1, "diagnosis_term_id": 23, "raw_text": "..." },
      { "rank": 2, "diagnosis_term_id": 7 },
      { "rank": 3, "diagnosis_term_id": null }
    ],
    "diagnostic_confidence": 4,
    "biopsy_recommended": false
  }
- Response: 200: { "assessment_id": 4001, "top1_correct": true, "top3_correct": true, "rank_of_truth": 1 }
- Side effects: server computes the correctness flags, persists `assessments` and `diagnosis_entries`, and if this was the final assessment for the block, triggers finalization.

GET /game/report/{block_index}
- Purpose: return the block report-card for the given user+block_index.
- Response: {
    "block_index": 2,
    "top1_accuracy_pre": 0.7,
    "top3_accuracy_pre": 0.9,
    "top1_accuracy_post": 0.8,
    "top3_accuracy_post": 0.92,
    "peer_percentile_top1": 84.3,
    "peer_percentile_top3": 90.1,
    "stats_json": { ... per-case breakdown ... }
  }

GET /game/report/latest (new) – if you just need the last one
GET /game/reports (new) – list all
GET /game/can_view_report/{block_index} (new) – check availability before fetching full report
GET /api/block_feedback/{id} (legacy style not implemented under /game yet)
- Fetch specific block_feedback row.

Resume behaviour (frontend)
- On app load or when user opens the game page, call GET /api/games/active.
- If the response shows an active block, the frontend should display "Resume game" and fetch the incomplete assignment(s). The UI should restore the last unfinished `display_order`.
- If user clicks Start but server returned 409 because an active block exists, show Resume prompt that links to the existing block.
 - Current implementation: no 409; `POST /game/start` simply returns existing active block. Frontend can always call it safely.

Enforcing single active game
- Server rule: before creating a new block, check for `reader_case_assignments` for user where block_index = N and `completed_pre_at` (or `completed_post_at`) is NULL for the requested phase. If any assignments in the block are incomplete, reject new starts with 409 and return existing block info.

Block finalization and peer percentile
- Finalizer triggered automatically by the server when it detects the Nth (10th) completed assessment for user+block_index+phase. Finalizer:
  - read the 10 assessments, compute counts and accuracies, store per-case breakdown in `block_feedback.stats_json`, set `top1_accuracy_pre` / `top3_accuracy_pre` or post equivalents, compute `delta_top1/delta_top3`.
  - enqueue or schedule a background job to recompute peer percentiles (so finalizer stays quick). The background job recomputes percentiles for all `block_feedback` rows with the same `block_index` and updates `peer_percentile_top1`/`_top3` and `peer_percentile_updated_at`.

Storage mapping (where things go)
- Per-case correctness and rank saved to `assessments.top1_correct`, `assessments.top3_correct`, `assessments.rank_of_truth`.
- Per-block aggregate + per-case breakdown saved to `block_feedback.stats_json` and scalar columns `top1_accuracy_pre/post`, `top3_accuracy_pre/post`, `delta_top1`, `delta_top3`.
- Peer percentile cached on `block_feedback.peer_percentile_top1` / `_top3` and `peer_percentile_updated_at` (recommended).

Edge cases and notes
- Partial blocks: do not finalize until the desired number of cases is completed (default 10). If you allow dynamic block sizes or allow resuming across multiple sessions, make sure the finalizer checks the actual count.
- Editing assessments after finalize: either block edits after finalization or re-run finalizer + percentile job when edits occur.
- Concurrent finishes: run finalizer inside a transaction and use idempotent insert/update to avoid double finalization.
- Small-population percentile noise: consider a minimum N for percentiles (e.g., require >= 10 blocks before showing percentile or mark percentile as provisional).

Implementation hints (server)
- Compute per-assessment correctness when saving assessments; keep that logic in one place (service or repo layer) so finalizer and other consumers reuse it.
- Finalizer should be fast and non-blocking: commit the `block_feedback` synchronously, and compute percentiles in a background worker (Celery/RQ or simple cron job).
- Keep `stats_json` small: include assignment_id, case_id, top1/top3 correctness, and rank_of_truth. The UI can fetch associated case/image data separately for the detail view.

Next steps I can do for you
- Add server-side endpoint skeletons in `app/api` and wire a stub `finalize_block` call when the 10th assessment is saved.
- Patch `scripts/import_ai_predictions.py` to store Top‑5 in `ai_outputs` and full vector in `cases.ai_predictions_json`, then run a sample import against a backup.

Tell me which next step you want me to perform and I'll implement it.
