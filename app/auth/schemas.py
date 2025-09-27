# backend/app/auth/schemas.py
from typing import Optional, Literal
from datetime import datetime

from fastapi_users import schemas
from pydantic import EmailStr, Field, field_validator


GenderLiteral = Literal["Male", "Female", "Other"]


class UserRead(schemas.BaseUser[int]):
    id: int
    email: EmailStr
    role_id: Optional[int] = None
    age_bracket: Optional[str] = None
    gender: Optional[GenderLiteral] = None
    years_experience: Optional[int] = None
    years_derm_experience: Optional[int] = None
    country_code: Optional[str] = Field(default=None, max_length=2)
    created_at: datetime
    is_active: bool = True
    is_superuser: bool = False
    is_verified: bool = False

    class Config:
        from_attributes = True


class UserCreate(schemas.BaseUserCreate):
    email: EmailStr
    password: str
    role_id: Optional[int] = None
    age_bracket: Optional[str] = None
    gender: Optional[GenderLiteral] = None
    years_experience: Optional[int] = None
    years_derm_experience: Optional[int] = None
    country_code: Optional[str] = Field(default=None, max_length=2)
    is_active: bool = True
    is_superuser: bool = False
    is_verified: bool = False

    @field_validator("gender")
    @classmethod
    def _norm_gender(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        m = v.strip().lower()
        if m in ("male", "m"):
            return "Male"
        if m in ("female", "f"):
            return "Female"
        if m in ("other", "non-binary", "nonbinary", "nb", "prefer not to say"):
            return "Other"
        if v not in ("Male", "Female", "Other"):
            raise ValueError("gender must be one of: Male, Female, Other")
        return v


class UserUpdate(schemas.BaseUserUpdate):
    password: Optional[str] = None
    email: Optional[EmailStr] = None
    role_id: Optional[int] = None
    age_bracket: Optional[str] = None
    gender: Optional[GenderLiteral] = None
    years_experience: Optional[int] = None
    years_derm_experience: Optional[int] = None
    country_code: Optional[str] = Field(default=None, max_length=2)
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None
    is_verified: Optional[bool] = None

    @field_validator("country_code")
    @classmethod
    def _norm_cc(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        v = v.upper()
        if len(v) != 2:
            raise ValueError("country_code must be 2 letters (ISO 3166-1 alpha-2)")
        return v

    @field_validator("gender")
    @classmethod
    def _norm_gender(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        # Accept case-insensitive input, coerce to title-case options
        m = v.strip().lower()
        if m in ("male", "m"):
            return "Male"
        if m in ("female", "f"):
            return "Female"
        if m in ("other", "non-binary", "nonbinary", "nb", "prefer not to say"):
            return "Other"
        # Enforce allowed set
        if v not in ("Male", "Female", "Other"):
            raise ValueError("gender must be one of: Male, Female, Other")
        return v