# backend/app/auth/db.py
from typing import AsyncGenerator

from fastapi import Depends
# Updated import for newer fastapi-users versions
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession

# Import from our unified database configuration
from app.db.session import get_async_db, async_engine
from app.auth.models import User

async def get_user_db(session: AsyncSession = Depends(get_async_db)):
    yield SQLAlchemyUserDatabase(session, User)