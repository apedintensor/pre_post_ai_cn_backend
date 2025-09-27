# backend/app/auth/routes.py
from fastapi import APIRouter, Depends, HTTPException

from app.auth.manager import fastapi_users, jwt_backend, current_active_user
# Import User directly from auth.models for consistency
from app.auth.models import User
from app.auth.schemas import UserCreate, UserRead, UserUpdate

# Create an API router for authentication routes
auth_router = APIRouter()

# Include the routes for registration, authentication, etc.
auth_router.include_router(
    fastapi_users.get_auth_router(jwt_backend),
    prefix="/jwt",
    tags=["auth"],
)

auth_router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    # No extra prefix so final path is /auth/register (avoids double /register/register)
    tags=["auth"],
)

# Legacy compatibility: some tests expect /auth/register/register
# Include the register router again but under an extra "/register" prefix so the final path becomes
# /auth/register/register (since the inner router already uses path "/register").
auth_router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/register",
    tags=["auth"],
)

auth_router.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/reset-password",
    tags=["auth"],
)

auth_router.include_router(
    fastapi_users.get_verify_router(UserRead),
    prefix="/verify",
    tags=["auth"],
)

auth_router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)

# Add a "me" endpoint to get current user info
@auth_router.get("/me", response_model=UserRead)
async def get_current_user(user: User = Depends(current_active_user)):
    """Get details of currently logged in user"""
    return user