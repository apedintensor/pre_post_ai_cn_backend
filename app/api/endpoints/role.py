# backend/app/api/endpoints/role.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app import crud
from app.schemas import schemas as app_schemas
from app.api import deps

router = APIRouter()

@router.get("/", response_model=List[app_schemas.RoleRead])
def read_roles(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100
):
    """Retrieve roles."""
    roles = crud.role.get_multi(db, skip=skip, limit=limit)
    return roles

@router.post("/", response_model=app_schemas.RoleRead, status_code=201)
def create_role(
    *, # Enforces keyword-only arguments
    db: Session = Depends(deps.get_db),
    role_in: app_schemas.RoleCreate
):
    """Create new role."""
    # Check if role already exists by name
    existing_role = crud.role.get_by_name(db=db, name=role_in.name)
    if existing_role:
        raise HTTPException(status_code=400, detail="Role with this name already exists")
    role = crud.role.create(db=db, role=role_in)
    return role

@router.get("/{role_id}", response_model=app_schemas.RoleRead)
def read_role(
    role_id: int,
    db: Session = Depends(deps.get_db),
):
    """Get role by ID."""
    db_role = crud.role.get(db=db, role_id=role_id)
    if db_role is None:
        raise HTTPException(status_code=404, detail="Role not found")
    return db_role

# Add PUT, DELETE if needed
