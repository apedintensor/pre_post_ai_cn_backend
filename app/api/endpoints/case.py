# backend/app/api/endpoints/case.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app import crud, schemas as app_schemas
from app.api import deps
from app.models.models import User
from app.auth.manager import current_active_user

router = APIRouter()

@router.get("/", response_model=List[app_schemas.CaseRead])
def get_cases(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(current_active_user)  # Direct dependency on auth
):
    """Get list of cases."""
    return crud.case.get_multi(db=db, skip=skip, limit=limit)

@router.get("/{case_id}", response_model=app_schemas.CaseRead)
def get_case(
    case_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(current_active_user)  # Direct dependency on auth
):
    """Get specific case by ID."""
    db_case = crud.case.get(db=db, case_id=case_id)
    if db_case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return db_case

@router.post("/", response_model=app_schemas.CaseRead, status_code=201)
def create_case(
    *, # Enforces keyword-only arguments
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(current_active_user),  # Direct dependency on auth
    case_in: app_schemas.CaseCreate
):
    """Create new case."""
    # Add checks (e.g., diagnosis term exists)
    case = crud.case.create(db=db, case=case_in)
    return case
