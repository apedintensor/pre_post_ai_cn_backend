# backend/app/db/session.py
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from app.core.config import settings

# Determine if we're using SQLite (only case where check_same_thread applies)
sync_uri = settings.SQLALCHEMY_SYNC_DATABASE_URI
async_uri = settings.SQLALCHEMY_ASYNC_DATABASE_URI

is_sqlite_sync = sync_uri.startswith("sqlite")
is_sqlite_async = async_uri.startswith("sqlite")

# Create sync engine
if is_sqlite_sync:
    sync_engine = create_engine(
        sync_uri,
        connect_args={"check_same_thread": False},
    )
else:
    sync_engine = create_engine(sync_uri)

# Create async engine
if is_sqlite_async:
    async_engine = create_async_engine(
        async_uri,
        connect_args={"check_same_thread": False},
    )
else:
    async_engine = create_async_engine(async_uri)

# Create session factories
# Sync session for compatibility with existing code
SessionLocal = sessionmaker(bind=sync_engine, autoflush=False, autocommit=False)

# Async session for new code and fastapi-users
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)

# Helper functions for dependency injection
def get_db():
    """Synchronous database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db():
    """Asynchronous database session dependency"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()