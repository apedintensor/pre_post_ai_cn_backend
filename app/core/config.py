"""Application configuration (Pydantic Settings).

Previous version triggered PydanticUserError due to unannotated class-level
assignments (ASYNC_DB_URL / SYNC_DB_URL). All dynamic DB URL logic is now
handled inside __init__ with properly annotated fields.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # Basic
    PROJECT_NAME: str = "Reader Study Web API"
    API_V1_STR: str = "/api/v1"  # (not currently applied to routers)

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "supersecretkey")
    JWT_ALGORITHM: str = "HS256"
    JWT_LIFETIME_SECONDS: int = 3600 * 24 * 7

    # Raw database URL (optional) provided via env
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

    # Resolved SQLAlchemy URIs (populated in __init__)
    SQLALCHEMY_ASYNC_DATABASE_URI: str = ""
    SQLALCHEMY_SYNC_DATABASE_URI: str = ""

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # User bootstrap
    SUPERUSER_EMAIL: str = os.getenv("SUPERUSER_EMAIL", "admin@example.com")
    SUPERUSER_PASSWORD: str = os.getenv("SUPERUSER_PASSWORD", "adminpassword")

    # Media
    IMAGE_BASE_URL: str = os.getenv("IMAGE_BASE_URL", "")

    # Game
    GAME_BLOCK_SIZE: int = int(os.getenv("GAME_BLOCK_SIZE", "3"))

    class Config:
        case_sensitive = True
        env_file = ".env"

    def __init__(self, **values):
        super().__init__(**values)
        # Derive DB URLs
        if self.DATABASE_URL:
            url = self.DATABASE_URL
            if url.startswith("postgres://"):
                self.SQLALCHEMY_ASYNC_DATABASE_URI = url.replace("postgres://", "postgresql+asyncpg://", 1)
                self.SQLALCHEMY_SYNC_DATABASE_URI = url.replace("postgres://", "postgresql+psycopg2://", 1)
            elif url.startswith("postgresql://"):
                self.SQLALCHEMY_ASYNC_DATABASE_URI = url.replace("postgresql://", "postgresql+asyncpg://", 1)
                self.SQLALCHEMY_SYNC_DATABASE_URI = url.replace("postgresql://", "postgresql+psycopg2://", 1)
            else:
                # Fallback to SQLite if unexpected
                print(
                    f"Warning: DATABASE_URL not recognized as PostgreSQL: {url}. Falling back to SQLite test.db"
                )
                self.SQLALCHEMY_ASYNC_DATABASE_URI = "sqlite+aiosqlite:///./test.db"
                self.SQLALCHEMY_SYNC_DATABASE_URI = "sqlite:///./test.db"
        else:
            self.SQLALCHEMY_ASYNC_DATABASE_URI = os.getenv(
                "ASYNC_DATABASE_URI", "sqlite+aiosqlite:///./test.db"
            )
            self.SQLALCHEMY_SYNC_DATABASE_URI = os.getenv(
                "SYNC_DATABASE_URI", "sqlite:///./test.db"
            )


settings = Settings()