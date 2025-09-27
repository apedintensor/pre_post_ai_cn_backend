from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.api import deps
from app.schemas.schemas import AssessmentCreate, AssessmentRead, AssessmentSubmitResponse
from app.services import game_service
from app.services.assessment_service import create_or_replace_assessment
from app.models.models import Assessment, ReaderCaseAssignment
from sqlalchemy import select

router = APIRouter()


@router.post("/", response_model=AssessmentSubmitResponse)
def submit_assessment(payload: AssessmentCreate, db: Session = Depends(deps.get_db)):
    try:
        assessment = create_or_replace_assessment(db, payload)
        # Determine block status
        assignment = db.get(ReaderCaseAssignment, assessment.assignment_id)
        block_index = assignment.block_index if assignment else -1
        # Fetch all assignments for block
        if assignment:
            from sqlalchemy import select
            block_assignments = db.execute(
                select(ReaderCaseAssignment).where(
                    ReaderCaseAssignment.user_id == assignment.user_id,
                    ReaderCaseAssignment.block_index == block_index
                )
            ).scalars().all()
            remaining = sum(1 for a in block_assignments if a.completed_post_at is None)
            block_complete = remaining == 0
        else:
            remaining = 0
            block_complete = False
        # Eager finalize if block just completed (idempotent)
        if block_complete and assignment:
            game_service.finalize_block_if_complete(db, assignment.user_id, block_index)
        # Build clean payload (avoid raw __dict__ which includes SA state and may omit relationships before refresh)
        resp = AssessmentSubmitResponse(
            id=assessment.id,
            assignment_id=assessment.assignment_id,
            phase=assessment.phase,
            diagnostic_confidence=assessment.diagnostic_confidence,
            management_confidence=assessment.management_confidence,
            investigation_action=assessment.investigation_action,
            next_step_action=assessment.next_step_action,
            changed_primary_diagnosis=assessment.changed_primary_diagnosis,
            changed_management_plan=assessment.changed_management_plan,
            ai_usefulness=assessment.ai_usefulness,
            top1_correct=assessment.top1_correct,
            top3_correct=assessment.top3_correct,
            rank_of_truth=assessment.rank_of_truth,
            created_at=assessment.created_at,
            diagnosis_entries=[{
                'id': de.id,
                'rank': de.rank,
                'raw_text': de.raw_text,
                'diagnosis_term_id': de.diagnosis_term_id,
            } for de in assessment.diagnosis_entries],
            block_index=block_index,
            block_complete=block_complete,
            report_available=block_complete,
            remaining_in_block=remaining,
        )
        return resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/assignment/{assignment_id}", response_model=List[AssessmentRead])
def get_assessments_for_assignment(assignment_id: int, db: Session = Depends(deps.get_db)):
    stmt = select(Assessment).where(Assessment.assignment_id == assignment_id).order_by(Assessment.phase.asc())
    return db.execute(stmt).scalars().all()


@router.get("/user/{user_id}/block/{block_index}", response_model=List[AssessmentRead])
def get_block_assessments(user_id: int, block_index: int, db: Session = Depends(deps.get_db)):
    stmt = select(Assessment).join(ReaderCaseAssignment).where(
        ReaderCaseAssignment.user_id == user_id,
        ReaderCaseAssignment.block_index == block_index
    )
    return db.execute(stmt).scalars().all()

