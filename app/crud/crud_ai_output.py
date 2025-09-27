# backend/app/crud/crud_ai_output.py
from sqlalchemy.orm import Session
from app.models import models
from app.schemas import schemas
from typing import List

def get_ai_output(db: Session, ai_output_id: int):
    return db.query(models.AIOutput).filter(models.AIOutput.id == ai_output_id).first()

def get_ai_outputs_by_case(db: Session, case_id: int, skip: int = 0, limit: int = 10):
    # Usually limited number of outputs per case
    return db.query(models.AIOutput).filter(models.AIOutput.case_id == case_id).order_by(models.AIOutput.rank).offset(skip).limit(limit).all()

def create_ai_output(db: Session, ai_output: schemas.AIOutputCreate):
    db_ai_output = models.AIOutput(**ai_output.model_dump()) # Pydantic v2
    db.add(db_ai_output)
    db.commit()
    db.refresh(db_ai_output)
    return db_ai_output

# Update/Delete as needed

ai_output = {
    "get": get_ai_output,
    "get_multi_by_case": get_ai_outputs_by_case,
    "create": create_ai_output,
}
