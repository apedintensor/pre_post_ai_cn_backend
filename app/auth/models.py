# backend/app/auth/models.py
from typing import Optional, List
from datetime import datetime

# Updated import for newer fastapi-users versions
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTable
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship

from app.db.base import Base

class User(SQLAlchemyBaseUserTable[int], Base):
    # Use the standard 'users' table name to match DBML
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Additional fields from our schema
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=True)
    age_bracket = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    years_experience = Column(Integer, nullable=True)
    years_derm_experience = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    # ISO 3166-1 alpha-2 country code, e.g. "AU", "US"
    country_code = Column(String(2), nullable=True, index=True)
    
    # Relationships
    role = relationship("Role", back_populates="users")