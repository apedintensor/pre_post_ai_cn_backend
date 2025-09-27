from __future__ import annotations
from sqlalchemy.orm import Session
from sqlalchemy import select
from datetime import datetime
from app.models.models import Assessment, ReaderCaseAssignment, DiagnosisEntry, DiagnosisTerm
from app.schemas.schemas import AssessmentCreate

def create_or_replace_assessment(db: Session, payload: AssessmentCreate) -> Assessment:
    """Create or replace an assessment ensuring rank uniqueness without violating constraints.

    Strategy:
      * Fetch existing assessment by (assignment_id, phase).
      * If exists: clear diagnosis_entries list (cascade delete-orphan) instead of deleting assessment row.
      * Update fields, then recreate DiagnosisEntry rows.
    """
    # ensure assignment exists
    assignment = db.get(ReaderCaseAssignment, payload.assignment_id)
    if not assignment:
        raise ValueError("Assignment not found")

    phase = payload.phase.upper()
    existing = db.execute(
        select(Assessment).where(
            Assessment.assignment_id == payload.assignment_id,
            Assessment.phase == phase,
        )
    ).scalar_one_or_none()

    if existing:
        assessment = existing
        # Clear old entries safely
        if assessment.diagnosis_entries:
            for de in list(assessment.diagnosis_entries):
                db.delete(de)
            db.flush()
        # Update mutable fields
        assessment.diagnostic_confidence = payload.diagnostic_confidence
        assessment.management_confidence = payload.management_confidence
        assessment.investigation_action = payload.investigation_action
        assessment.next_step_action = payload.next_step_action
        assessment.changed_primary_diagnosis = payload.changed_primary_diagnosis
        assessment.changed_management_plan = payload.changed_management_plan
        assessment.ai_usefulness = payload.ai_usefulness
    else:
        assessment = Assessment(
            assignment_id=payload.assignment_id,
            phase=phase,
            diagnostic_confidence=payload.diagnostic_confidence,
            management_confidence=payload.management_confidence,
            investigation_action=payload.investigation_action,
            next_step_action=payload.next_step_action,
            changed_primary_diagnosis=payload.changed_primary_diagnosis,
            changed_management_plan=payload.changed_management_plan,
            ai_usefulness=payload.ai_usefulness,
        )
        db.add(assessment)
        db.flush()

    # Upsert diagnosis entries by rank (avoid UNIQUE constraint race)
    existing_by_rank = {de.rank: de for de in assessment.diagnosis_entries}
    incoming_ranks = {e.rank for e in payload.diagnosis_entries}

    # Update or create
    for entry in payload.diagnosis_entries:
        current = existing_by_rank.get(entry.rank)
        if current:
            current.raw_text = entry.raw_text
            current.diagnosis_term_id = entry.diagnosis_term_id
        else:
            db.add(
                DiagnosisEntry(
                    assessment_id=assessment.id,
                    rank=entry.rank,
                    raw_text=entry.raw_text,
                    diagnosis_term_id=entry.diagnosis_term_id,
                )
            )

    # Remove any old ranks (e.g., user reduced from 3 diagnoses to 1)
    for rank, obj in existing_by_rank.items():
        if rank not in incoming_ranks:
            db.delete(obj)

    db.flush()

    # compute correctness (top1, top3) if ground truth exists
    # IMPORTANT: use a fresh query for entries to avoid stale relationship cache
    if assignment.case.ground_truth_diagnosis_id:
        from app.models.models import DiagnosisEntry as _DE
        gt_id = assignment.case.ground_truth_diagnosis_id
        entries = db.execute(
            select(_DE).where(_DE.assessment_id == assessment.id).order_by(_DE.rank.asc())
        ).scalars().all()
        top_ids = [e.diagnosis_term_id for e in entries]
        if not top_ids:
            assessment.top1_correct = None
            assessment.top3_correct = None
            assessment.rank_of_truth = None
        else:
            try:
                found_index = top_ids.index(gt_id)
                found_rank = found_index + 1
                assessment.top1_correct = found_rank == 1
                assessment.top3_correct = found_rank <= 3
                assessment.rank_of_truth = found_rank
            except ValueError:
                assessment.top1_correct = False
                assessment.top3_correct = False
                assessment.rank_of_truth = None

    # mark completion times on assignment
    now = datetime.utcnow()
    if phase == 'PRE':
        assignment.completed_pre_at = now
    else:
        assignment.completed_post_at = now

    db.commit()
    db.refresh(assessment)
    return assessment
