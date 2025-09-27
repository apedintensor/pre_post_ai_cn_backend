# backend/app/crud/crud_case_metadata.py
from sqlalchemy.orm import Session
from app.models import models
from app.schemas import schemas

class CaseMetadataCRUD:
    def get(self, db: Session, metadata_id: int):
        return db.query(models.CaseMetaData).filter(models.CaseMetaData.id == metadata_id).first()

    def get_by_case(self, db: Session, case_id: int):
        # Assuming one metadata per case due to unique constraint
        return db.query(models.CaseMetaData).filter(models.CaseMetaData.case_id == case_id).first()

    def create(self, db: Session, metadata: schemas.CaseMetaDataCreate, case_id: int):
        db_metadata = models.CaseMetaData(**metadata.model_dump(), case_id=case_id)
        db.add(db_metadata)
        db.commit()
        db.refresh(db_metadata)
        return db_metadata

case_metadata = CaseMetadataCRUD()
