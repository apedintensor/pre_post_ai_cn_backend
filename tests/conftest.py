# backend/tests/conftest.py
import asyncio
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import os
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
import uuid

from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import create_engine
from sqlalchemy import inspect, text

from app.db.base import Base
# Ensure small block size for tests (complete a block with a single case)
os.environ.setdefault("GAME_BLOCK_SIZE", "1")
from app.main import app as fastapi_app
from app.models.models import User, Role
from app.core.config import settings
from app.db.session import get_async_db

# Override database settings for testing
# Use file-based SQLite for testing
test_db_file = f"test_{uuid.uuid4()}.db"
TEST_SQLALCHEMY_DATABASE_URI = f"sqlite+aiosqlite:///{test_db_file}"

# Create test async engine
test_engine = create_async_engine(
    TEST_SQLALCHEMY_DATABASE_URI,
    connect_args={"check_same_thread": False},
    echo=True  # Added echo=True to see SQL statements for debugging
)

# Create test async session
TestAsyncSessionLocal = sessionmaker(
    test_engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Create a sync engine/session bound to the same SQLite file for sync endpoints
TEST_SQLALCHEMY_SYNC_URI = f"sqlite:///{test_db_file}"
test_sync_engine = create_engine(
    TEST_SQLALCHEMY_SYNC_URI,
    connect_args={"check_same_thread": False},
)
TestSyncSessionLocal = sessionmaker(bind=test_sync_engine, autoflush=False, autocommit=False)

# Override the get_async_db dependency to use test database
async def override_get_async_db() -> AsyncGenerator[AsyncSession, None]:
    async with TestAsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Override the get_db dependency to use test sync database
def override_get_db():
    db = TestSyncSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Test user data
test_user = {
    "email": "test@example.com",
    "password": "password123",
}

test_superuser = {
    "email": "admin@example.com",
    "password": "adminpass123",
}


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def init_db():
    """Set up the test database"""
    # Create all tables in the test database
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    # Ensure new assessment columns exist (SQLite lacks auto migrations)
    insp = inspect(test_sync_engine)
    cols = [c["name"] for c in insp.get_columns("assessments")]
    with test_sync_engine.begin() as conn:
        if "investigation_action" not in cols:
            conn.execute(text("ALTER TABLE assessments ADD COLUMN investigation_action VARCHAR"))
        if "next_step_action" not in cols:
            conn.execute(text("ALTER TABLE assessments ADD COLUMN next_step_action VARCHAR"))
    # Seed minimal data required for tests (roles, diagnosis terms/synonyms, cases, images, ai_outputs)
    # Use the helper functions from scripts/seed_basic but with the test sync session.
    try:
        from scripts.seed_basic import ensure_roles, load_terms_from_json, load_cases, DATA_DIR
        terms_json = DATA_DIR / "derm_dictionary.json"
        cases_csv = DATA_DIR / "ai_prediction_by_id.csv"
        with TestSyncSessionLocal() as db:
            ensure_roles(db, ["GP", "Nurse", "Other"])
            load_terms_from_json(db, terms_json)
            load_cases(db, cases_csv)
            db.commit()
    except Exception as e:  # pragma: no cover
        # If seeding fails for any reason, continue; some tests may create their own data.
        print(f"[tests] Seed skipped due to error: {e}")
    
    # Removed seeding of extra roles (Admin/Doctor/Researcher) to mirror production role set.
    # If specific tests need roles, they should create them explicitly within the test.
    async with TestAsyncSessionLocal() as session:
        await session.commit()
    
    yield
    
    # Clean up resources after tests
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()
    test_sync_engine.dispose()
    
    # Remove the test database file
    try:
        os.remove(test_db_file)
    except FileNotFoundError:
        pass


@pytest_asyncio.fixture(scope="session")
async def app(init_db):
    # Override the get_async_db and get_db dependencies to use test database
    from app.db.session import get_async_db, get_db
    fastapi_app.dependency_overrides[get_async_db] = override_get_async_db
    fastapi_app.dependency_overrides[get_db] = override_get_db
    
    yield fastapi_app


@pytest_asyncio.fixture(scope="function")
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """
    Create a new FastAPI AsyncClient that uses the `app` fixture to make requests.
    """
    from httpx import ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Create a fresh database session for tests.
    """
    async with TestAsyncSessionLocal() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def registered_user(client: AsyncClient) -> dict:
    """
    Register a test user and return user data.
    If the user already exists, log them in to verify credentials.
    """
    # Try to register the user
    response = await client.post(
        "/auth/register", 
        json=test_user,
    )
    
    # If the user already exists, that's fine too
    if response.status_code in (200, 201):
        pass
    elif response.status_code == 400 and "REGISTER_USER_ALREADY_EXISTS" in response.text:
        # Verify the user by logging in
        login_data = {
            "username": test_user["email"],
            "password": test_user["password"],
        }
        login_response = await client.post("/auth/jwt/login", data=login_data)
        assert login_response.status_code == 200, f"Failed to login with existing user: {login_response.text}"
    else:
        # New user registration should succeed
        assert response.status_code == 201, f"Failed to register user: {response.status_code} {response.text}"
    
    return test_user


@pytest_asyncio.fixture(scope="function")
async def user_token_headers(client: AsyncClient, registered_user: dict) -> dict:
    """
    Get token headers for registered user.
    """
    login_data = {
        "username": registered_user["email"],
        "password": registered_user["password"],
    }
    response = await client.post("/auth/jwt/login", data=login_data)
    assert response.status_code == 200, f"Failed to login: {response.text}"
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}