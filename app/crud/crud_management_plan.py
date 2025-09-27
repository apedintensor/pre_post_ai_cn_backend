# backend/app/crud/crud_management_plan.py
from sqlalchemy.orm import Session
from app.models import models
from app.schemas import schemas

class ManagementPlanCRUD:
    def get(self, db: Session, plan_id: int):
        return db.query(models.ManagementPlan).filter(models.ManagementPlan.id == plan_id).first()

    def get_by_assessment(self, db: Session, user_id: int, case_id: int, is_post_ai: bool):
        return db.query(models.ManagementPlan).filter(
            models.ManagementPlan.assessment_user_id == user_id,
            models.ManagementPlan.assessment_case_id == case_id,
            models.ManagementPlan.assessment_is_post_ai == is_post_ai
        ).first()

    def create(self, db: Session, plan: schemas.ManagementPlanCreate):
        db_plan = models.ManagementPlan(**plan.model_dump()) # Pydantic v2
        db.add(db_plan)
        db.commit()
        db.refresh(db_plan)
        return db_plan

    def update(self, db: Session, user_id: int, case_id: int, is_post_ai: bool, plan: schemas.ManagementPlanUpdate):
        db_plan = self.get_by_assessment(db=db, user_id=user_id, case_id=case_id, is_post_ai=is_post_ai)
        if not db_plan:
            return None
            
        for field, value in plan.model_dump(exclude_unset=True).items():
            setattr(db_plan, field, value)
            
        db.add(db_plan)
        db.commit()
        db.refresh(db_plan)
        return db_plan

management_plan = ManagementPlanCRUD()
