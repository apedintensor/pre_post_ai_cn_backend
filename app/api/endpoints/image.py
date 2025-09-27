# backend/app/api/endpoints/image.py
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
@router.get("/case/{case_id}", response_model=List[app_schemas.ImageRead])
def read_images_by_case(
    case_id: int,
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100
):
    """Retrieve images for a specific case."""
    images = crud.image.get_multi_by_case(db=db, case_id=case_id, skip=skip, limit=limit)
    return images

# Use the alias for the response_model and input schema
@router.post("/", response_model=app_schemas.ImageRead, status_code=201)
def create_image(
    *, # Enforces keyword-only arguments
    db: Session = Depends(deps.get_db),
    image_in: app_schemas.ImageCreate
):
    """Create new image record."""
    # Add check if case exists
    db_case = crud.case.get(db=db, case_id=image_in.case_id)
    if not db_case:
        raise HTTPException(status_code=404, detail="Case not found")
    image = crud.image.create(db=db, image=image_in)
    return image

# Use the alias for the response_model
@router.get("/{image_id}", response_model=app_schemas.ImageRead)
def read_image(
    image_id: int,
    db: Session = Depends(deps.get_db),
):
    """Get image by ID."""
    db_image = crud.image.get(db=db, image_id=image_id)
    if db_image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return db_image

# Add PUT, DELETE if needed
