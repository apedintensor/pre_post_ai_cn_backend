# Frontend Update Guide (Refactored Game & Assessment API)

This document summarizes all backend data model and API changes required for the updated reader study workflow (block-based PRE/POST assessments with AI impact + feedback reports).

---
## High-Level Changes

1. Assessments no longer use a composite key `(user_id, case_id, is_post_ai)`.
2. New entity `ReaderCaseAssignment` drives game blocks (10 cases per block) and links user ↔ case.
3. Each assignment can have up to 2 assessments distinguished by `phase` = `PRE` or `POST`.
4. Diagnoses are now stored as ranked `DiagnosisEntry` (1..3) per assessment.
5. Block completion triggers creation of `BlockFeedback` summarizing accuracy deltas and (later) percentiles.
6. Legacy endpoints for assessments (with composite params) and diagnosis/management plan flows are deprecated.

---
## New / Updated API Endpoints

Base prefix: `/api/v1`

### Game Flow
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/game/start` | Start a new block (or return existing active block if unfinished). |
| GET  | `/game/active` | Get the current active (incomplete) block, or empty if none. |
| GET  | `/game/report/{block_index}` | Get (or finalize & then return) feedback for a completed block. |

### Assessments
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/assessment/` | Create/replace PRE or POST assessment (idempotent per (assignment_id, phase)). |
| GET  | `/assessment/assignment/{assignment_id}` | List assessments (0–2) for an assignment. |
| GET  | `/assessment/user/{user_id}/block/{block_index}` | All assessments for a user+block (audit/report use). |

### Existing (Unchanged) Reference Data (examples)
You can continue to use existing endpoints for cases, images, AI outputs, diagnosis terms, users, etc., if already present in the frontend. (Verify actual paths already implemented in frontend.)

---
## Data Contracts (TypeScript Interfaces)

```ts
// Core reference
export interface DiagnosisTermRead {
  id: number;
  name: string;
}

export interface ImageRead {
  id: number;
  case_id: number;
  image_url: string;
}

export interface AIOutputRead {
  id: number;
  case_id: number;
  rank: number;          // 1 = top prediction
  prediction_id: number; // FK to DiagnosisTerm
  confidence_score: number | null;
  prediction?: DiagnosisTermRead | null; // included if backend eager loads
}

export interface CaseRead {
  id: number;
  ground_truth_diagnosis_id?: number | null;
  ground_truth_diagnosis?: DiagnosisTermRead | null;
  created_at: string; // ISO
  ai_predictions_json?: Record<string, number> | null; // full vector (term_id -> score)
  images: ImageRead[];
  ai_outputs: AIOutputRead[]; // usually top-k
}

// Game assignment
export interface ReaderCaseAssignmentRead {
  id: number;
  user_id: number;
  case_id: number;
  arm: string;          // e.g. "A" or future HUMAN / HUMAN_AI
  display_order: number; // 0..9 within block
  block_index: number;   // sequential block identifier
  started_at: string;    // ISO timestamp
  completed_pre_at?: string | null;
  completed_post_at?: string | null;
  // (Optional) You may separately fetch the case details (not automatically embedded)
}

// Diagnosis entries (ranked list user provides)
export interface DiagnosisEntryCreate {
  rank: 1 | 2 | 3;
  raw_text?: string;             // free text if user typed
  diagnosis_term_id?: number;    // canonical term selection
}

export interface DiagnosisEntryRead extends DiagnosisEntryCreate {
  id: number;
}

// Assessment create payload
export interface AssessmentCreate {
  assignment_id: number;
  phase: 'PRE' | 'POST';
  diagnostic_confidence?: number | null;    // scale defined by UI (e.g., 1-5)
  management_confidence?: number | null;    // scale defined by UI
  investigation_action?: 'NONE' | 'BIOPSY' | 'OTHER' | null;
  next_step_action?: 'REASSURE' | 'MANAGE_MYSELF' | 'REFER' | null;
  // POST-only (send null/omit for PRE)
  changed_primary_diagnosis?: boolean | null;
  changed_management_plan?: boolean | null;
  ai_usefulness?: string | null;            // free text or categorical label
  diagnosis_entries: DiagnosisEntryCreate[]; // rank must be unique & start at 1
}

// Assessment read response
export interface AssessmentRead {
  id: number;
  assignment_id: number;
  phase: 'PRE' | 'POST';
  diagnostic_confidence?: number | null;
  management_confidence?: number | null;
  investigation_action?: 'NONE' | 'BIOPSY' | 'OTHER' | null;
  next_step_action?: 'REASSURE' | 'MANAGE_MYSELF' | 'REFER' | null;
  changed_primary_diagnosis?: boolean | null;
  changed_management_plan?: boolean | null;
  ai_usefulness?: string | null;
  top1_correct?: boolean | null;
  top3_correct?: boolean | null;
  rank_of_truth?: number | null; // if user listed the ground truth
  created_at: string;
  diagnosis_entries: DiagnosisEntryRead[];
}

// Block feedback (report card)
export interface BlockFeedbackRead {
  id: number;
  user_id: number;
  block_index: number;
  top1_accuracy_pre?: number | null;  // 0-1
  top1_accuracy_post?: number | null; // 0-1
  top3_accuracy_pre?: number | null;
  top3_accuracy_post?: number | null;
  delta_top1?: number | null;         // post - pre
  delta_top3?: number | null;
  peer_percentile_top1?: number | null; // may be null until batch job runs
  peer_percentile_top3?: number | null;
  stats_json?: Record<string, any> | null; // reserved for future richer stats
  created_at: string;
}

// Game endpoint responses
export interface StartGameResponse {
  block_index: number;
  assignments: ReaderCaseAssignmentRead[];
}

export interface ActiveGameResponse {
  block_index: number;  // -1 if none active
  assignments: ReaderCaseAssignmentRead[];
  remaining: number;     // count of assignments missing POST assessment
}

export type ReportCardResponse = BlockFeedbackRead;
```

---
## Endpoint Request / Response Examples

### 1. Start a Block
POST `/api/v1/game/start`
```json
{
  "block_index": 3,
  "assignments": [
    {"id": 91, "user_id": 5, "case_id": 240, "arm": "A", "display_order": 0, "block_index": 3, "started_at": "2025-09-03T18:20:01Z", "completed_pre_at": null, "completed_post_at": null},
    {"id": 92, "user_id": 5, "case_id": 17,  "arm": "A", "display_order": 1, "block_index": 3, "started_at": "2025-09-03T18:20:01Z", "completed_pre_at": null, "completed_post_at": null}
    // ... up to 10
  ]
}
```

### 2. Submit PRE Assessment
POST `/api/v1/assessment/`
```json
{
  "assignment_id": 91,
  "phase": "PRE",
  "diagnostic_confidence": 4,
  "management_confidence": 3,
  "investigation_action": "NONE",
  "next_step_action": "REFER",
  "diagnosis_entries": [
    {"rank": 1, "diagnosis_term_id": 12},
    {"rank": 2, "diagnosis_term_id": 55},
    {"rank": 3, "raw_text": "Rare dermatosis"}
  ]
}
```
Response (simplified):
```json
{
  "id": 3001,
  "assignment_id": 91,
  "phase": "PRE",
  "top1_correct": false,
  "top3_correct": true,
  "rank_of_truth": 2,
  "diagnosis_entries": [ {"id": 9001, "rank": 1, "diagnosis_term_id": 12}, ... ]
}
```

### 3. Submit POST Assessment
Same endpoint; include POST-only fields if relevant:
```json
{
  "assignment_id": 91,
  "phase": "POST",
  "diagnostic_confidence": 5,
  "management_confidence": 4,
  "changed_primary_diagnosis": true,
  "changed_management_plan": false,
  "ai_usefulness": "helpful",
  "diagnosis_entries": [
    {"rank": 1, "diagnosis_term_id": 12},
    {"rank": 2, "diagnosis_term_id": 77},
    {"rank": 3, "diagnosis_term_id": 55}
  ]
}
```

### 4. Poll Active Block
GET `/api/v1/game/active`
```json
{
  "block_index": 3,
  "assignments": [
    {"id": 91, "completed_pre_at": "2025-09-03T18:25:00Z", "completed_post_at": "2025-09-03T18:30:15Z", ...},
    {"id": 92, "completed_pre_at": "2025-09-03T18:27:33Z", "completed_post_at": null, ...}
  ],
  "remaining": 1
}
```

### 5. Fetch Report Card
GET `/api/v1/game/report/3`
```json
{
  "id": 12,
  "user_id": 5,
  "block_index": 3,
  "top1_accuracy_pre": 0.3,
  "top1_accuracy_post": 0.6,
  "delta_top1": 0.3,
  "top3_accuracy_pre": 0.7,
  "top3_accuracy_post": 0.9,
  "delta_top3": 0.2,
  "peer_percentile_top1": null,
  "peer_percentile_top3": null,
  "created_at": "2025-09-03T18:45:00Z"
}
```

---
## Workflow Sequence Diagram (Textual)

1. User clicks "Start" → POST `/game/start` → receives block assignments.
2. For each assignment in ascending `display_order`:
   a. Display case (fetch case + images + AI outputs if/when needed).
   b. User submits PRE via POST `/assessment/`.
   c. (Optionally reveal AI outputs / probability vector.)
   d. User submits POST via POST `/assessment/`.
3. Frontend polls `/game/active` to update remaining count.
4. When all assignments have POST submitted, GET `/game/report/{block_index}` to obtain feedback (auto-finalizes if not yet).
5. Show report card; allow user to start next block.

---
## Validation Rules (Client-Side Recommendations)

| Field | Rule |
|-------|------|
| `diagnosis_entries` | Must contain unique ranks (1..3). Rank 1 required; others optional until submitted. |
| `phase` | Only 'PRE' or 'POST'. POST should only be allowed after successful PRE submission for that assignment (enforce in UI). |
| `changed_primary_diagnosis` | Only send for POST (omit or null in PRE). |
| `top1_correct` etc. | Read-only (returned by server). |

---
## Migration Impact on Frontend Code

| Old Concept | New Concept | Action |
|-------------|------------|--------|
| Composite assessment key `(user_id, case_id, is_post_ai)` | `assessment.id` + `(assignment_id, phase)` | Replace key logic, store `assignment_id` per case. |
| `Diagnosis` table (per assessment) | `DiagnosisEntry` | Rename model & update field names (`diagnosis_term_id`, `raw_text`). |
| Separate PRE/POST endpoints | Single POST `/assessment/` with `phase` | Consolidate form submission function. |
| Block feedback absent | `BlockFeedbackRead` via `/game/report/{block_index}` | Add report view & fetch logic. |
| Determining correctness client-side | Server computes correctness | Remove client correctness logic; rely on response fields. |

---
## Edge Cases to Handle in UI

1. Starting a new block while one is active returns the existing active block (do not assume fresh list length = 10 every time if data changes). 
2. A user may refresh mid-block; use `/game/active` to restore state. 
3. Report request before all POST assessments → 404 (show "Finish remaining cases"). 
4. Peer percentile fields may remain `null` until a background job populates them. 
5. Cases might lack ground truth (correctness fields will be null). Hide correctness-specific UI gracefully. 
6. AI outputs or full vector may not be loaded initially—defer fetch until after PRE submission to avoid leakage. 

---
## Suggested Frontend Store Shape (Example)

```ts
interface GameState {
  activeBlockIndex: number | null;
  assignments: Record<number, ReaderCaseAssignmentRead>; // key = assignment.id
  cases: Record<number, CaseRead>;                        // lazy-loaded
  assessments: Record<number, AssessmentRead[]>;          // key = assignment_id -> [PRE?, POST?]
  reportCards: Record<number, BlockFeedbackRead>;         // key = block_index
  loading: boolean;
  error?: string | null;
}
```

---
## Minimal Client Submission Helper (Pseudo-Code)

```ts
async function submitAssessment(payload: AssessmentCreate): Promise<AssessmentRead> {
  const res = await fetch('/api/v1/assessment/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
```

---
## Open / Future Fields

| Field | Purpose | Status |
|-------|---------|--------|
| `ai_predictions_json` | Full probability map for research downloads | Present, may be large (lazy fetch). |
| `peer_percentile_*` | Cohort percentile metrics | Populated by future batch job. |
| `stats_json` | Extended block stats (e.g., confusion patterns) | Reserved. |

---
## QA Checklist for Frontend Integration

1. Can start a block and receive 10 assignments.
2. Submitting PRE twice replaces previous PRE (idempotent per phase).
3. POST refusal before PRE (UI prevents; server would overwrite if allowed—but rely on UI gating).
4. Correctness flags visible only after submission; null when ground truth missing.
5. Block report only available when all POST completed.
6. Starting next block after report returns new `block_index`.

---
## Questions to Clarify (If Needed)

1. Should AI outputs be hidden until PRE is saved? (Current assumption: yes.)
2. Confidence scale (1–5? 0–100?) – make sure consistent across UI & server.
3. Are partial (less than 3) diagnosis entries allowed? (Assumed allowed; rank1 required.)
4. Should user be allowed to edit PRE after POST? (Current server implementation overwrites on new submission; UI decide policy.)

---
End of document.
