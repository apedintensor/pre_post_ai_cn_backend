# backend/app/api/endpoints/diagnosis_term.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List

from app import crud
from app.schemas import schemas as app_schemas
from app.api import deps

router = APIRouter()

@router.get("/suggest", response_model=List[app_schemas.DiagnosisSuggestion])
def suggest_terms(
    q: str = Query(min_length=2, description="User typed fragment"),
    limit: int = Query(10, ge=1, le=25),
    db: Session = Depends(deps.get_db)
):
    """Return ranked diagnosis term suggestions (name + synonyms).
    Placed before /{term_id} route to avoid path param capture precedence issues.
    """
    return crud.crud_diagnosis_term.synonyms["suggest"](db, q, limit=limit)

@router.get("/", response_model=List[app_schemas.DiagnosisTermRead])
def read_diagnosis_terms(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100
):
    """Retrieve diagnosis terms."""
    terms = crud.diagnosis_term["get_multi"](db, skip=skip, limit=limit)
    return terms

@router.post("/", response_model=app_schemas.DiagnosisTermRead, status_code=201)
def create_diagnosis_term(
    *, # Enforces keyword-only arguments
    db: Session = Depends(deps.get_db),
    term_in: app_schemas.DiagnosisTermCreate
):
    """Create new diagnosis term."""
    # Check if term already exists by name
    existing_term = crud.diagnosis_term["get_by_name"](db=db, name=term_in.name)
    if existing_term:
        raise HTTPException(status_code=400, detail="Diagnosis term with this name already exists")
    term = crud.diagnosis_term["create"](db=db, term=term_in)
    return term

@router.get("/{term_id}", response_model=app_schemas.DiagnosisTermRead)
def read_diagnosis_term(
    term_id: int,
    db: Session = Depends(deps.get_db),
):
    """Get diagnosis term by ID."""
    db_term = crud.diagnosis_term["get"](db=db, term_id=term_id)
    if db_term is None:
        raise HTTPException(status_code=404, detail="Diagnosis term not found")
    return db_term

# Add PUT, DELETE if needed


@router.post("/synonyms", response_model=app_schemas.DiagnosisSynonymRead, status_code=201)
def create_synonym(
    *,
    db: Session = Depends(deps.get_db),
    synonym_in: app_schemas.DiagnosisSynonymCreate
):
    # uniqueness enforced at DB level (unique synonym)
    return crud.crud_diagnosis_term.synonyms["create"](db, synonym_in)


@router.get("/synonyms", response_model=List[app_schemas.DiagnosisSynonymRead])
def list_synonyms(
    term_id: int | None = None,
    db: Session = Depends(deps.get_db)
):
    return crud.crud_diagnosis_term.synonyms["list"](db, term_id=term_id)


