# backend/app/api/endpoints/management_strategy.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app import crud
from app.schemas import schemas as app_schemas
from app.api import deps

router = APIRouter()

@router.get("/", response_model=List[app_schemas.ManagementStrategyRead])
def read_management_strategies(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100
):
    """Retrieve management strategies."""
    strategies = crud.management_strategy.get_multi(db, skip=skip, limit=limit)
    if not strategies:
        # Seed a minimal default set if empty
        defaults = ["Reassure", "Manage myself", "Refer"]
        for name in defaults:
            if not crud.management_strategy.get_by_name(db, name):
                crud.management_strategy.create(db, app_schemas.ManagementStrategyCreate(name=name))
        strategies = crud.management_strategy.get_multi(db, skip=skip, limit=limit)
    return strategies

@router.post("/", response_model=app_schemas.ManagementStrategyRead, status_code=201)
def create_management_strategy(
    *, # Enforces keyword-only arguments
    db: Session = Depends(deps.get_db),
    strategy_in: app_schemas.ManagementStrategyCreate
):
    """Create new management strategy."""
    # Check if strategy already exists by name
    existing_strategy = crud.management_strategy.get_by_name(db=db, name=strategy_in.name)
    if existing_strategy:
        raise HTTPException(status_code=400, detail="Management strategy with this name already exists")
    strategy = crud.management_strategy.create(db=db, strategy=strategy_in)
    return strategy

@router.get("/{strategy_id}", response_model=app_schemas.ManagementStrategyRead)
def read_management_strategy(
    strategy_id: int,
    db: Session = Depends(deps.get_db),
):
    """Get management strategy by ID."""
    db_strategy = crud.management_strategy.get(db=db, strategy_id=strategy_id)
    if db_strategy is None:
        raise HTTPException(status_code=404, detail="Management strategy not found")
    return db_strategy

# Add PUT, DELETE if needed
