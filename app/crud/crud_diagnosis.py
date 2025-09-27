# backend/app/crud/crud_diagnosis.py
from sqlalchemy.orm import Session
from app.models import models
from app.schemas import schemas
from typing import List

class DiagnosisCRUD:
    def get(self, db: Session, diagnosis_id: int):
        return db.query(models.Diagnosis).filter(models.Diagnosis.id == diagnosis_id).first()

    def get_multi_by_assessment(self, db: Session, assessment_id: int, skip: int = 0, limit: int = 100):
        return db.query(models.Diagnosis).filter(models.Diagnosis.assessment_id == assessment_id).offset(skip).limit(limit).all()

    def create(self, db: Session, diagnosis: schemas.DiagnosisCreate):
        db_diagnosis = models.Diagnosis(**diagnosis.model_dump()) # Pydantic v2
        db.add(db_diagnosis)
        db.commit()
        db.refresh(db_diagnosis)
        return db_diagnosis

diagnosis = DiagnosisCRUD()
