# backend/app/api/deps.py
from fastapi import Depends, HTTPException, status
from typing import Generator

# Import the centralized database session
from app.db.session import get_db, get_async_db
from app.auth.manager import current_active_user, current_superuser
from app.auth.models import User

# Re-export the synchronous database session
def get_db_session():
    return get_db()

# Re-export the asynchronous database session
async def get_async_db_session():
    async for session in get_async_db():
        yield session

# Re-export auth dependencies
def get_current_user(user: User = Depends(current_active_user)) -> User:
    return user

def get_current_superuser(user: User = Depends(current_superuser)) -> User:
    return user
