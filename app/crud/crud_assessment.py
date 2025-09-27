# backend/app/crud/crud_assessment.py
from sqlalchemy.orm import Session
from app.models import models
from app.schemas import schemas
from typing import List, Optional
from sqlalchemy import and_

class AssessmentCRUD:
    def get(self, db: Session, user_id: int, case_id: int, is_post_ai: bool):
        return db.query(models.Assessment).filter(
            and_(
                models.Assessment.user_id == user_id,
                models.Assessment.case_id == case_id,
                models.Assessment.is_post_ai == is_post_ai
            )
        ).first()

    def get_multi_by_user(self, db: Session, user_id: int, skip: int = 0, limit: int = 100):
        return db.query(models.Assessment).filter(models.Assessment.user_id == user_id).offset(skip).limit(limit).all()

    def get_multi_by_case(self, db: Session, case_id: int, skip: int = 0, limit: int = 100):
        return db.query(models.Assessment).filter(models.Assessment.case_id == case_id).offset(skip).limit(limit).all()

    def get_multi(self, db: Session, skip: int = 0, limit: int = 100):
        return db.query(models.Assessment).offset(skip).limit(limit).all()

    def create(self, db: Session, assessment: schemas.AssessmentCreate):
        db_assessment = models.Assessment(**assessment.model_dump())
        db.add(db_assessment)
        db.commit()
        db.refresh(db_assessment)
        return db_assessment

    def update(self, db: Session, user_id: int, case_id: int, is_post_ai: bool, assessment: schemas.AssessmentUpdate):
        db_assessment = self.get(db=db, user_id=user_id, case_id=case_id, is_post_ai=is_post_ai)
        if not db_assessment:
            return None
            
        for field, value in assessment.model_dump(exclude_unset=True).items():
            setattr(db_assessment, field, value)
            
        db.add(db_assessment)
        db.commit()
        db.refresh(db_assessment)
        return db_assessment

    def get_by_composite_key(self, db: Session, user_id: int, case_id: int, is_post_ai: bool):
        return db.query(models.Assessment).filter(
            models.Assessment.user_id == user_id,
            models.Assessment.case_id == case_id,
            models.Assessment.is_post_ai == is_post_ai
        ).first()

assessment = AssessmentCRUD()
