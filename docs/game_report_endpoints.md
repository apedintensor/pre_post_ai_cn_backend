# Game Report & Progress Endpoints (Frontend Integration Guide)

This document describes the endpoints your frontend should use to manage game progress, determine report availability, and fetch block performance summaries.

## Summary of Endpoints

Base prefix (already mounted): `/game`

| Purpose | Method & Path | Auth | Response (200) |
|---------|---------------|------|----------------|
| Start or resume current block | POST `/game/start` | Bearer | `{ block_index, assignments: [...] }` |
| Active block state | GET `/game/active` | Bearer | `{ block_index, assignments, remaining }` or `{ block_index: -1, ...}` when none |
| Submit assessment (PRE/POST) | POST `/assessment/` | Bearer | Assessment JSON (created/updated) |
| Report for specific block | GET `/game/report/{block_index}` | Bearer | `ReportCardResponse` |
| Check if report can be viewed (without fetching full report) | GET `/game/can_view_report/{block_index}` | Bearer | `{ available: bool, block_index, reason? }` |
| Latest completed report | GET `/game/report/latest` | Bearer | `ReportCardResponse` |
| List all completed reports | GET `/game/reports` | Bearer | `ReportCardResponse[]` |

## Data Shapes

`ReaderCaseAssignment` (subset used by UI):
```
{
  id: number,
  case_id: number,
  display_order: number,
  block_index: number,
  started_at: string (ISO),
  completed_pre_at: string | null,
  completed_post_at: string | null
}
```

`ReportCardResponse`:
```
{
  id: number,
  user_id: number,
  block_index: number,
  top1_accuracy_pre: number | null,
  top1_accuracy_post: number | null,
  top3_accuracy_pre: number | null,
  top3_accuracy_post: number | null,
  delta_top1: number | null,
  delta_top3: number | null,
  peer_percentile_top1: number | null,
  peer_percentile_top3: number | null,
  stats_json: object | null,
  created_at: string (ISO)
}
```

`can_view_report` response (lightweight):
```
// Available
{ "available": true,  "block_index": 2 }

// Not yet available
{ "available": false, "block_index": 2, "reason": "3 cases pending" }

// Invalid block index
{ "available": false, "block_index": 99, "reason": "Block not found" }
```

## Typical Frontend Flow

1. User opens the Game dashboard.
2. Call `GET /game/active`.
   - If `block_index == -1`: show a CTA button: "Start New Block".
   - Else: show progress (e.g., `remaining` cases) and button: "Resume Block".
3. When user clicks Start/Resume:
   - Call `POST /game/start` (idempotent: will return existing unfinished block instead of error).
   - Render list or linear workflow of `assignments` sorted by `display_order`.
4. For each case interaction:
   - Submit PRE assessment via `POST /assessment/`.
   - (After AI reveal) submit POST assessment via the same endpoint.
5. After posting POST assessment for the final assignment of the block:
   - Backend finalizes block (creates `BlockFeedback`).
   - Frontend can poll `GET /game/can_view_report/{block_index}` every few seconds until `available: true` (usually instant unless peer metrics recomputation is asynchronous).
6. Once available, fetch `GET /game/report/{block_index}` (or `GET /game/report/latest`).
7. Display performance metrics. Provide navigation to prior reports using `GET /game/reports`.

## Determining Report Availability (Fast Path)
Use `GET /game/can_view_report/{block_index}` before requesting full report to:
- Avoid a 404 -> try finalize -> 404 loop.
- Provide immediate user messaging (e.g., "3 cases pending").

## Handling Edge Cases
- Starting a new block while one is in progress: server returns existing block assignments (no error). Always treat `POST /game/start` as "safe to call".
- User abandons mid-way: `GET /game/active` will reflect remaining count. Continue from first assignment with `completed_post_at == null`.
- Report not found / missing: If `can_view_report` says `Block not found`, remove stale local references.
- Peer percentile fields may be null if not computed or insufficient cohort size yet. UI should fallback to placeholder (e.g., "–" or tooltip: "Percentile pending").

## Minimal UI State Model
```
interface GameState {
  activeBlockIndex: number | null; // -1 or null means none
  assignments: ReaderCaseAssignment[];
  remaining: number; // derived
  reports: ReportCardResponse[]; // cached list from /game/reports
  latestReport?: ReportCardResponse;
}
```

## Polling Strategy Example
```
async function waitForReport(blockIndex, token) {
  for (let attempt = 0; attempt < 10; attempt++) {
    const r = await fetch(`/game/can_view_report/${blockIndex}`, { headers: { Authorization: `Bearer ${token}` }}).then(res => res.json());
    if (r.available) return true;
    if (r.reason?.includes('not found')) break;
    await new Promise(res => setTimeout(res, 1500));
  }
  return false;
}
```

## Performance Considerations
- `can_view_report` executes only lightweight queries; safe to poll.
- `latest_report` and `reports` both use simple ordered selects; add pagination later if block counts become large.

## Future Enhancements (Optional)
- Add `GET /game/progress` summarizing: total blocks started, completed, cumulative accuracy trends.
- WebSocket or Server-Sent Events for real-time report readiness instead of polling.
- Include per-case image thumbnails in `stats_json` to avoid extra round trips.

## Testing Notes
- Existing automated smoke test (`tests/test_workflow.py`) covers start → PRE/POST → report path.
- Add a small test for `can_view_report` + `latest_report` if you expand test coverage.

---
Questions or adjustments needed for the frontend? Extend this doc or request additional aggregation endpoints.
