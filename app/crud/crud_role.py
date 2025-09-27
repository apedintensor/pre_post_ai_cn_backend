# backend/app/crud/crud_role.py
from sqlalchemy.orm import Session
from app.models import models
from app.schemas import schemas

class RoleCRUD:
    def get(self, db: Session, role_id: int):
        return db.query(models.Role).filter(models.Role.id == role_id).first()

    def get_by_name(self, db: Session, name: str):
        return db.query(models.Role).filter(models.Role.name == name).first()

    def get_multi(self, db: Session, skip: int = 0, limit: int = 100):
        return db.query(models.Role).offset(skip).limit(limit).all()

    def create(self, db: Session, role: schemas.RoleCreate):
        db_role = models.Role(name=role.name)
        db.add(db_role)
        db.commit()
        db.refresh(db_role)
        return db_role

role = RoleCRUD()
