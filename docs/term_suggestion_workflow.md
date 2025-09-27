# Diagnosis Term Suggestion Workflow (Updated)

Purpose: Give the frontend a precise contract for mapping user-typed diagnosis text to canonical `diagnosis_terms` via the new suggestion + synonym endpoints while retaining original free text for analytics.

## 1. Goals
1. Minimize friction (sub‑300ms perceived latency for suggestions after typing).
2. Guarantee canonical mapping before assessment submission (no orphan free text).
3. Preserve original user wording (`raw_text`) for qualitative analysis.
4. Handle typos / partial tokens gracefully.
5. Avoid accidental selection (clear confirmation step).
6. Keep implementation simple (pure SQL + lightweight in‑memory cache) but extensible (future embedding or ML reranker).

## 2. Data Inputs
| Source | Table | Needed Fields | Notes |
|--------|-------|---------------|-------|
| Canonical terms | `diagnosis_terms` | id, name | You may preload once (GET `/api/v1/diagnosis_terms?limit=500`). |
| Synonyms | `diagnosis_synonyms` | id, diagnosis_term_id, synonym | Queried indirectly via suggestion endpoint; full preload optional for offline fallback. |
| Suggestions API | (derived) | id, label, match, source, score | Returned by GET `/api/v1/diagnosis_terms/suggest?q=...`. |
| User text | (front-end) | raw string | Saved as `raw_text` if it differs from canonical label chosen. |

## 3. UX States
1. Idle (empty input) – no dropdown.
2. User types ≥ 2 characters → trigger debounced suggestion fetch (150ms).
3. Show suggestion list (max 8–10 items): highlighted match segments.
4. Keyboard navigation (ArrowUp/Down), Enter to select, Escape to close.
5. On selection: populate input with canonical term label; retain original typed text separately (if different) for submission.
6. Allow clearing / retyping (resets mapping; submission disabled until re‑mapped).

## 4. Mapping Contract (Front-end → Back-end)
For each diagnosis rank the client must send:
```jsonc
{
  "rank": 1,
  "raw_text": "ptchiosis",          // what the user actually typed (may contain typo)
  "diagnosis_term_id": 42            // canonical selected term id
}
```
`raw_text` may equal the canonical term name if user clicked it directly or typed it perfectly.

## 5. Suggestion Retrieval Strategy
Now that synonyms + suggest endpoint exist, implement a hybrid:

Hybrid Recommendation:
1. Preload canonical terms at app bootstrap (fast, for instant local matches and label rendering).
2. For user query length ≥ 2, call server `/suggest` to include synonyms + canonical names with scoring.
3. Merge: If a canonical term already in local list also appears in server results, prefer server-provided score & match metadata.
4. Cache server responses by lowercased query substring (LRU of last ~30 queries) to reduce chatter when user navigates back.

Fallback (Offline / No Network): Use purely local filtering (see former Option A). Accept reduced synonym coverage.

### 5.1 Server Suggest Endpoint (Implemented)
Endpoint: `GET /api/v1/diagnosis_terms/suggest?q={fragment}&limit=10`

Query Params:
| Name | Type | Notes |
|------|------|-------|
| q | string (min 2) | User-typed fragment (case-insensitive). |
| limit | int (1..25) | Optional; default 10. |

Response (HTTP 200):
```
[
  {"id": 42, "label": "Basal Cell Carcinoma", "match": "BCC", "source": "synonym", "score": 219.5},
  {"id": 42, "label": "Basal Cell Carcinoma", "match": "Basal Cell Carcinoma", "source": "name", "score": 192.5}
]
```
Returned list is already deduplicated by term id (best scoring variant kept). Fields:
| Field | Meaning |
|-------|---------|
| id | Canonical diagnosis term id (use this in submissions). |
| label | Canonical diagnosis term (display value). |
| match | The text fragment that matched (synonym or canonical name). |
| source | 'synonym' or 'name'. |
| score | Heuristic for ordering (higher is better). |

Highlighting: Use `match` to emphasize substring inside `label` or show small pill with synonym if `source == 'synonym'`.

### 5.2 Local Assist (Optional)
Perform an immediate local filter (canonical names only) so UI feels instant; replace/merge with server results when they arrive (race pattern). If server returns before user moves on, reconcile ordering using server scores.

### 5.3 Synonym Management
- Create synonym (admin UI only): `POST /api/v1/diagnosis_terms/synonyms` body `{ "diagnosis_term_id": 42, "synonym": "BCC" }`.
- List synonyms: `GET /api/v1/diagnosis_terms/synonyms?term_id=42` (optional filter).
Front-end for study participants only needs `/suggest`.

## 6. Validation Rules
| Rule | Rationale |
|------|-----------|
| Rank 1 must be present before enabling PRE submit. | Ensures primary diagnosis always captured. |
| No duplicate `diagnosis_term_id` across ranks in same assessment. | Avoid redundant predictions; encourages diversity. |
| `diagnosis_term_id` required (cannot submit free text only). | Downstream analytics need canonical IDs. |
| If user edits text after selecting term, invalidate mapping until reselected. | Prevent mismatch between shown and stored concept. |

## 7. Error Handling / Edge Cases
| Case | Handling |
|------|----------|
| Empty input | Hide suggestions; disable add/select. |
| No matches found | Show "No matches – refine text" and keep submit disabled. |
| Fast paste of full correct term | Auto-select if exact case-insensitive match found; still allow manual override. |
| Mobile autocomplete adds trailing space | Trim before matching; don’t clear selection if label still matches. |
| User deletes selected term | Clear mapping (`diagnosis_term_id` null client-side), mark field invalid. |

## 8. Accessibility
* Use ARIA role="combobox" with aria-expanded, aria-controls pointing to list.
* Keyboard: Tab to move focus out, Enter to commit selection, Escape to close.
* Ensure visible focus ring on list items.

## 9. Performance Considerations
* Local array of 134 terms: O(n) filter per keystroke trivial (<1 ms in modern browsers).
* Debounce 120–180 ms to balance responsiveness vs churn.
* Memoize last query results to short‑circuit repeated same prefix.

## 10. Data Flow Diagram (Textual)
```
User types
  ├─ local quick filter (names only, optimistic UI)
  └─ debounce 150ms → GET /diagnosis_terms/suggest?q=frag
         ↓ merge & display top <= 10 unified suggestions
User selects item
  → store {diagnosis_term_id, canonical_label, raw_text(original)} in form state keyed by rank
User edits input after selection
  → clear diagnosis_term_id & mark field invalid until re-selected
Submit PRE/POST assessment
  → POST /assessment with diagnosis_entries[] (each has rank, raw_text, diagnosis_term_id)
Server persists + computes correctness flags
```

## 11. Analytics Enabled
| Metric | How Computed |
|--------|--------------|
| Term selection latency | Timestamp difference (input focus vs selection). (Optional; front-end only if needed.) |
| Correction rate | % where `raw_text.lower() != canonical_label.lower()` |
| Diversity | Count unique term IDs per assessment / block. |
| Pre→Post change | Compare term IDs rank-by-rank. |

## 12. Future Enhancements
| Idea | Description |
|------|-------------|
| Synonym ingestion | Add `diagnosis_synonyms` to widen match set. |
| Fuzzy scoring service | Use trigram similarity or embedding-based search when term count grows. |
| Popularity boost | Re-rank by historical frequency or AI top-k prior probabilities. |
| Inline AI hint | If AI top1 differs from user rank1 PRE, subtle prompt after PRE submission. |

## 13. Minimal Front-End Pseudocode (Hybrid)
```ts
interface CanonicalTerm { id: number; name: string }
interface Suggestion { id: number; label: string; match: string; source: 'name'|'synonym'; score: number }

let canonical: CanonicalTerm[] = [];
const serverCache = new Map<string, Suggestion[]>(); // key = query

async function initTerms() {
  const res = await fetch('/api/v1/diagnosis_terms?limit=500');
  canonical = await res.json();
}

function localFilter(q: string): Suggestion[] {
  const lc = q.toLowerCase();
  if (lc.length < 2) return [];
  return canonical.filter(t => t.name.toLowerCase().includes(lc))
    .map(t => ({ id: t.id, label: t.name, match: q, source: 'name' as const, score: heuristic(q, t.name) }))
    .sort((a,b)=> b.score - a.score).slice(0,10);
}

function heuristic(q: string, name: string): number {
  const ln = name.toLowerCase(), lq = q.toLowerCase();
  if (ln.startsWith(lq)) return 200 - (ln.length - lq.length);
  if (ln.includes(lq)) return 80 - Math.abs(ln.length - lq.length)*0.5;
  return 0;
}

async function serverSuggest(q: string): Promise<Suggestion[]> {
  const key = q.toLowerCase();
  if (serverCache.has(key)) return serverCache.get(key)!;
  const res = await fetch(`/api/v1/diagnosis_terms/suggest?q=${encodeURIComponent(q)}`);
  if (!res.ok) return [];
  const data: Suggestion[] = await res.json();
  serverCache.set(key, data);
  return data;
}

// Combined suggestion flow
async function getSuggestions(q: string, signal?: AbortSignal): Promise<Suggestion[]> {
  const local = localFilter(q);
  if (q.trim().length < 2) return [];
  // Fire server (do not await) to update later
  serverSuggest(q).then(remote => {
    if (signal?.aborted) return;
    if (!remote.length) return; // keep local if remote empty
    const merged = merge(local, remote);
    render(merged);
  });
  return local; // immediate optimistic result
}

function merge(local: Suggestion[], remote: Suggestion[]): Suggestion[] {
  const byId = new Map<number, Suggestion>();
  [...local, ...remote].forEach(s => {
    const existing = byId.get(s.id);
    if (!existing || s.score > existing.score) byId.set(s.id, s);
  });
  return [...byId.values()].sort((a,b)=> b.score - a.score).slice(0,10);
}

// Form state example
interface DiagnosisEntryDraft { rank: number; input: string; diagnosis_term_id?: number; canonical_label?: string; raw_text?: string; }

function onSelect(entry: DiagnosisEntryDraft, suggestion: Suggestion) {
  entry.diagnosis_term_id = suggestion.id;
  entry.canonical_label = suggestion.label;
  entry.raw_text = entry.input.trim(); // store what was typed
}

function onEdit(entry: DiagnosisEntryDraft) {
  // If user edits after selection and diverges from canonical label, clear mapping
  if (entry.diagnosis_term_id && entry.input.trim().toLowerCase() !== entry.canonical_label?.toLowerCase()) {
    entry.diagnosis_term_id = undefined;
  }
}

function buildAssessmentPayload(drafts: DiagnosisEntryDraft[]) {
  const diagnosis_entries = drafts
    .filter(d => d.diagnosis_term_id)
    .map(d => ({ rank: d.rank, raw_text: d.raw_text, diagnosis_term_id: d.diagnosis_term_id }));
  return { /* other assessment fields... */ diagnosis_entries };
}
```

## 14. Server Endpoint Summary (Current)
| Purpose | Method & Path | Notes |
|---------|---------------|-------|
| List canonical terms | GET `/api/v1/diagnosis_terms` | Use once at app start. Pagination via `skip`/`limit`. |
| Suggest (names + synonyms) | GET `/api/v1/diagnosis_terms/suggest?q=frag` | Primary runtime lookup. Returns array of suggestions. |
| Create synonym (admin) | POST `/api/v1/diagnosis_terms/synonyms` | Requires `diagnosis_term_id`, `synonym` unique. |
| List synonyms (admin) | GET `/api/v1/diagnosis_terms/synonyms?term_id=` | Optional management UI. |
| Submit assessment | POST `/api/v1/assessment` | Include `diagnosis_entries` array with mapped IDs. |

## 15. Submission Validation (Server-Side)
Current guarantees:
* DB NOT NULL on `diagnosis_entries.diagnosis_term_id` rejects unmapped entries.
* `(assessment_id, rank)` unique -> prevents duplicate rank numbers.

Frontend SHOULD also enforce:
1. Rank sequence starts at 1 (rank 1 required before enabling submit) and is contiguous for filled entries.
2. No duplicate `diagnosis_term_id` across ranks (prevent user selecting same term twice). Simple set check before enabling submit.
3. Clear `diagnosis_term_id` if input diverges from selected canonical label (ensures user intentionally reselects).

Optional future server hook: reject duplicates explicitly (add uniqueness constraint on `(assessment_id, diagnosis_term_id)` if needed).

---
End of document (frontend integration ready).
