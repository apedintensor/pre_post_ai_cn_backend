from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud
from app.schemas import schemas as app_schemas
from app.api import deps

router = APIRouter()

@router.post("/case/{case_id}", response_model=app_schemas.CaseMetaDataRead, status_code=201)
def create_case_metadata(
	case_id: int,
	metadata_in: app_schemas.CaseMetaDataCreate,
	db: Session = Depends(deps.get_db),
):
	"""Create or replace metadata for a case."""
	existing = crud.case_metadata.get_by_case(db, case_id)
	if existing:
		# Overwrite existing for idempotency in tests
		db.delete(existing)
		db.commit()
	created = crud.case_metadata.create(db, metadata_in, case_id)
	return created


@router.get("/case/{case_id}", response_model=app_schemas.CaseMetaDataRead)
def get_case_metadata(
	case_id: int,
	db: Session = Depends(deps.get_db),
):
	meta = crud.case_metadata.get_by_case(db, case_id)
	if not meta:
		raise HTTPException(status_code=404, detail="Case metadata not found")
	return meta
