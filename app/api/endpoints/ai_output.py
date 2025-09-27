# backend/app/api/endpoints/ai_output.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

# Change the import style
from app import crud
# Import the schemas module directly and alias it
from app.schemas import schemas as app_schemas
from app.api import deps

router = APIRouter()

# Use the alias for the response_model
@router.get("/case/{case_id}", response_model=List[app_schemas.AIOutputRead])
def read_ai_outputs_by_case(
    case_id: int,
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 10
):
    """Retrieve AI outputs for a specific case."""
    ai_outputs = crud.ai_output["get_multi_by_case"](db=db, case_id=case_id, skip=skip, limit=limit)
    return ai_outputs

# Use the alias for the response_model and input schema
@router.post("/", response_model=app_schemas.AIOutputRead, status_code=201)
def create_ai_output(
    *, # Enforces keyword-only arguments
    db: Session = Depends(deps.get_db),
    ai_output_in: app_schemas.AIOutputCreate
):
    """Create new AI output."""
    # Add checks if needed (e.g., case exists)
    ai_output = crud.ai_output["create"](db=db, ai_output=ai_output_in)
    return ai_output

# Add GET by ID, PUT, DELETE if needed
