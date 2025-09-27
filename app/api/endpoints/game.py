from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from app.api import deps
from app.schemas.schemas import StartGameResponse, ActiveGameResponse, ReportCardResponse, GameProgressResponse, NextAssignmentResponse
from app.services import game_service
from app.models.models import ReaderCaseAssignment, BlockFeedback, Case, Assessment, DiagnosisEntry

router = APIRouter()


@router.post("/start", response_model=StartGameResponse)
def start_game(db: Session = Depends(deps.get_db), current_user=Depends(deps.get_current_user)):
    assignments = game_service.start_block(db, current_user.id)
    if not assignments:
        raise HTTPException(status_code=400, detail="No cases available for new block")
    return StartGameResponse(block_index=assignments[0].block_index, assignments=assignments)


@router.get("/active", response_model=ActiveGameResponse)
def active_game(db: Session = Depends(deps.get_db), current_user=Depends(deps.get_current_user)):
    block = game_service.get_active_block(db, current_user.id)
    if not block:
        return ActiveGameResponse(block_index=-1, assignments=[], remaining=0)
    # remaining = number of assignments without POST assessment
    remaining = sum(1 for a in block if a.completed_post_at is None)
    return ActiveGameResponse(block_index=block[0].block_index, assignments=block, remaining=remaining)


@router.get("/report/{block_index}", response_model=ReportCardResponse)
def report_block(block_index: int, db: Session = Depends(deps.get_db), current_user=Depends(deps.get_current_user)):
    feedback = db.execute(select(BlockFeedback).where(BlockFeedback.user_id == current_user.id, BlockFeedback.block_index == block_index)).scalar_one_or_none()
    if not feedback:
        # attempt finalize
        feedback = game_service.finalize_block_if_complete(db, current_user.id, block_index)
    if not feedback:
        # Improve diagnostics: is block absent or just incomplete?
        assignments = db.execute(select(ReaderCaseAssignment).where(ReaderCaseAssignment.user_id == current_user.id, ReaderCaseAssignment.block_index == block_index)).scalars().all()
        existing_block_indices = db.execute(select(ReaderCaseAssignment.block_index).where(ReaderCaseAssignment.user_id == current_user.id).distinct()).scalars().all()
        if not assignments:
            raise HTTPException(status_code=404, detail={
                "error": "block_not_found",
                "requested_block_index": block_index,
                "existing_block_indices": sorted(existing_block_indices),
                "message": "Requested block_index does not exist for this user."
            })
        remaining = sum(1 for a in assignments if a.completed_post_at is None)
        raise HTTPException(status_code=404, detail={
            "error": "block_incomplete",
            "requested_block_index": block_index,
            "remaining_cases": remaining,
            "message": "Block not yet complete; finish all POST assessments then retry."
        })
    # augment with case list + ground truths for this block
    assignments = db.execute(select(ReaderCaseAssignment).where(ReaderCaseAssignment.user_id == current_user.id, ReaderCaseAssignment.block_index == block_index)).scalars().all()
    case_ids = [a.case_id for a in assignments]
    gt_map = {}
    if case_ids:
        cases = db.execute(select(Case).where(Case.id.in_(case_ids))).scalars().all()
        gt_map = {c.id: c.ground_truth_diagnosis_id for c in cases}
    ordered = sorted(assignments, key=lambda a: a.display_order)
    # Build per-case user diagnosis info (pre/post top1 + top3 lists)
    case_summaries = []
    for a in ordered:
        assessments = db.execute(select(Assessment).where(Assessment.assignment_id == a.id)).scalars().all()
        pre = next((x for x in assessments if x.phase == 'PRE'), None)
        post = next((x for x in assessments if x.phase == 'POST'), None)
        def extract(ass: Assessment):
            if not ass:
                return None, []
            entries = db.execute(select(DiagnosisEntry).where(DiagnosisEntry.assessment_id == ass.id).order_by(DiagnosisEntry.rank.asc())).scalars().all()
            top_ids = [e.diagnosis_term_id for e in entries]
            top1 = top_ids[0] if top_ids else None
            return top1, top_ids
        pre_top1, pre_list = extract(pre)
        post_top1, post_list = extract(post)
        case_summaries.append({
            "case_id": a.case_id,
            "ground_truth_diagnosis_id": gt_map.get(a.case_id),
            "pre_top1_diagnosis_term_id": pre_top1,
            "post_top1_diagnosis_term_id": post_top1,
            "pre_top_diagnosis_term_ids": pre_list,
            "post_top_diagnosis_term_ids": post_list,
        })
    data = {**feedback.__dict__}
    data["total_cases"] = len(case_summaries)
    data["cases"] = case_summaries
    return data


@router.get("/can_view_report/{block_index}")
def can_view_report(block_index: int, db: Session = Depends(deps.get_db), current_user=Depends(deps.get_current_user)):
    """Lightweight check used by UI when user clicks game card to know if report is available.
    Returns {available: bool, reason?: str}
    """
    feedback = db.execute(select(BlockFeedback).where(BlockFeedback.user_id == current_user.id, BlockFeedback.block_index == block_index)).scalar_one_or_none()
    if feedback:
        return {"available": True, "block_index": block_index}
    # If not existing, see if block is fully complete (can be finalized)
    finalized = game_service.finalize_block_if_complete(db, current_user.id, block_index)
    if finalized:
        return {"available": True, "block_index": block_index}
    # Determine remaining cases if the block exists
    assignments = db.execute(select(ReaderCaseAssignment).where(ReaderCaseAssignment.user_id == current_user.id, ReaderCaseAssignment.block_index == block_index)).scalars().all()
    if not assignments:
        return {"available": False, "block_index": block_index, "reason": "Block not found"}
    remaining = sum(1 for a in assignments if a.completed_post_at is None)
    return {"available": False, "block_index": block_index, "reason": f"{remaining} cases pending"}


@router.get("/reports", response_model=list[ReportCardResponse])
def list_completed_reports(db: Session = Depends(deps.get_db), current_user=Depends(deps.get_current_user)):
    """Return all finalized BlockFeedback rows for the user ordered by block_index."""
    rows = db.execute(select(BlockFeedback).where(BlockFeedback.user_id == current_user.id).order_by(BlockFeedback.block_index.asc())).scalars().all()
    out = []
    for fb in rows:
        assignments = db.execute(select(ReaderCaseAssignment).where(ReaderCaseAssignment.user_id == current_user.id, ReaderCaseAssignment.block_index == fb.block_index)).scalars().all()
        case_ids = [a.case_id for a in assignments]
        gt_map = {}
        if case_ids:
            cases = db.execute(select(Case).where(Case.id.in_(case_ids))).scalars().all()
            gt_map = {c.id: c.ground_truth_diagnosis_id for c in cases}
        ordered = sorted(assignments, key=lambda a: a.display_order)
        case_summaries = []
        for a in ordered:
            assessments = db.execute(select(Assessment).where(Assessment.assignment_id == a.id)).scalars().all()
            pre = next((x for x in assessments if x.phase == 'PRE'), None)
            post = next((x for x in assessments if x.phase == 'POST'), None)
            def extract(ass: Assessment):
                if not ass:
                    return None, []
                entries = db.execute(select(DiagnosisEntry).where(DiagnosisEntry.assessment_id == ass.id).order_by(DiagnosisEntry.rank.asc())).scalars().all()
                top_ids = [e.diagnosis_term_id for e in entries]
                top1 = top_ids[0] if top_ids else None
                return top1, top_ids
            pre_top1, pre_list = extract(pre)
            post_top1, post_list = extract(post)
            case_summaries.append({
                "case_id": a.case_id,
                "ground_truth_diagnosis_id": gt_map.get(a.case_id),
                "pre_top1_diagnosis_term_id": pre_top1,
                "post_top1_diagnosis_term_id": post_top1,
                "pre_top_diagnosis_term_ids": pre_list,
                "post_top_diagnosis_term_ids": post_list,
            })
        d = {**fb.__dict__}
        d["total_cases"] = len(case_summaries)
        d["cases"] = case_summaries
        out.append(d)
    return out


@router.get("/report/latest", response_model=ReportCardResponse)
def latest_report(db: Session = Depends(deps.get_db), current_user=Depends(deps.get_current_user)):
    row = db.execute(select(BlockFeedback).where(BlockFeedback.user_id == current_user.id).order_by(BlockFeedback.block_index.desc()).limit(1)).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="No reports yet")
    assignments = db.execute(select(ReaderCaseAssignment).where(ReaderCaseAssignment.user_id == current_user.id, ReaderCaseAssignment.block_index == row.block_index)).scalars().all()
    case_ids = [a.case_id for a in assignments]
    gt_map = {}
    if case_ids:
        cases = db.execute(select(Case).where(Case.id.in_(case_ids))).scalars().all()
        gt_map = {c.id: c.ground_truth_diagnosis_id for c in cases}
    ordered = sorted(assignments, key=lambda a: a.display_order)
    case_summaries = []
    for a in ordered:
        assessments = db.execute(select(Assessment).where(Assessment.assignment_id == a.id)).scalars().all()
        pre = next((x for x in assessments if x.phase == 'PRE'), None)
        post = next((x for x in assessments if x.phase == 'POST'), None)
        def extract(ass: Assessment):
            if not ass:
                return None, []
            entries = db.execute(select(DiagnosisEntry).where(DiagnosisEntry.assessment_id == ass.id).order_by(DiagnosisEntry.rank.asc())).scalars().all()
            top_ids = [e.diagnosis_term_id for e in entries]
            top1 = top_ids[0] if top_ids else None
            return top1, top_ids
        pre_top1, pre_list = extract(pre)
        post_top1, post_list = extract(post)
        case_summaries.append({
            "case_id": a.case_id,
            "ground_truth_diagnosis_id": gt_map.get(a.case_id),
            "pre_top1_diagnosis_term_id": pre_top1,
            "post_top1_diagnosis_term_id": post_top1,
            "pre_top_diagnosis_term_ids": pre_list,
            "post_top_diagnosis_term_ids": post_list,
        })
    data = {**row.__dict__}
    data["total_cases"] = len(case_summaries)
    data["cases"] = case_summaries
    return data


@router.get("/progress", response_model=GameProgressResponse)
def game_progress(db: Session = Depends(deps.get_db), current_user=Depends(deps.get_current_user)):
    total_cases = db.query(func.count(Case.id)).scalar() or 0
    assigned_q = db.query(ReaderCaseAssignment).where(ReaderCaseAssignment.user_id == current_user.id)
    assigned_cases = assigned_q.count()
    completed_cases = assigned_q.filter(ReaderCaseAssignment.completed_post_at.isnot(None)).count()
    in_progress_cases = assigned_q.filter(
        ReaderCaseAssignment.completed_pre_at.isnot(None),
        ReaderCaseAssignment.completed_post_at.is_(None)
    ).count()
    remaining_cases = max(total_cases - completed_cases, 0)
    unassigned_cases = max(total_cases - assigned_cases, 0)
    return GameProgressResponse(
        total_cases=total_cases,
        completed_cases=completed_cases,
        remaining_cases=remaining_cases,
        assigned_cases=assigned_cases,
        unassigned_cases=unassigned_cases,
        in_progress_cases=in_progress_cases,
    )


@router.post("/next", response_model=NextAssignmentResponse)
def next_assignment(db: Session = Depends(deps.get_db), current_user=Depends(deps.get_current_user)):
    """Return the next incomplete assignment in the active block, or start a new block if finished.

    Response shape:
    {
      "status": "continuing" | "started" | "exhausted",
      "block_index": int | null,
      "assignment": ReaderCaseAssignment | null,
      "remaining": int  # remaining (including this one) in current block when continuing
    }
    """
    block_index, assignment = game_service.start_or_continue_game(db, current_user.id)
    if block_index is None or assignment is None:
        return NextAssignmentResponse(status="exhausted", block_index=None, assignment=None, remaining=0)
    # Determine remaining in block (unfinished including current)
    block = game_service.get_active_block(db, current_user.id)
    remaining = sum(1 for a in block if a.completed_post_at is None) if block else 0
    status = "continuing" if any(a.block_index == block_index for a in block if a.id != assignment.id) else "started"
    return NextAssignmentResponse(
        status=status,
        block_index=block_index,
        assignment=assignment,
        remaining=remaining,
    )
