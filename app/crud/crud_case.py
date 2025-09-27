# backend/app/crud/crud_case.py
from sqlalchemy.orm import Session
from app.models import models
from app.schemas import schemas

class CaseCRUD:
    def get(self, db: Session, case_id: int):
        return db.query(models.Case).filter(models.Case.id == case_id).first()

    def get_multi(self, db: Session, skip: int = 0, limit: int = 100):
        return db.query(models.Case).offset(skip).limit(limit).all()

    def create(self, db: Session, case: schemas.CaseCreate):
        db_case = models.Case(**case.model_dump())
        db.add(db_case)
        db.commit()
        db.refresh(db_case)
        return db_case

case = CaseCRUD()
