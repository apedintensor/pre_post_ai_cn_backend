# backend/app/main.py
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.api.routes import api_router
from app.auth.routes import auth_router
from app.db.session import async_engine
from app.db.base import Base
from app.core.config import settings
from app.db.init_db import init_db
from app.core.exceptions import (
    EntityNotFoundException, entity_not_found_exception_handler,
    PermissionDeniedException, permission_denied_exception_handler,
    DuplicateEntryException, duplicate_entry_exception_handler,
    GenericServerError, generic_server_error_handler,
    http_exception_handler, validation_exception_handler,
    pydantic_validation_handler, unexpected_exception_handler
)

# Create FastAPI app
app = FastAPI(title=settings.PROJECT_NAME)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
app.add_exception_handler(EntityNotFoundException, entity_not_found_exception_handler)
app.add_exception_handler(PermissionDeniedException, permission_denied_exception_handler)
app.add_exception_handler(DuplicateEntryException, duplicate_entry_exception_handler)
app.add_exception_handler(GenericServerError, generic_server_error_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ValidationError, pydantic_validation_handler)
app.add_exception_handler(Exception, unexpected_exception_handler)

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint that provides basic information about the API.
    """
    return {
        "title": settings.PROJECT_NAME,
        "description": "Backend API for Reader Study Web Application",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

# Admin endpoint
@app.get("/admin", tags=["admin"])
async def admin():
    """
    Admin endpoint that redirects to the admin interface.
    In the future, this could be a protected admin interface.
    """
    # You can customize this to redirect to your admin UI when it's ready
    # For now, redirecting to API docs as placeholder
    return RedirectResponse(url="/docs")

"""Main FastAPI application setup."""

# Include the authentication router (primary)
app.include_router(auth_router, prefix="/auth", tags=["auth"])

# Compatibility: duplicate auth routes under /api/auth (hidden from docs)
app.include_router(auth_router, prefix="/api/auth", include_in_schema=False)

# Canonical (current) routes without prefix remain visible in docs
app.include_router(api_router)

# Compatibility layer: duplicate routes under /api prefix (hidden from docs to avoid duplication)
app.include_router(api_router, prefix="/api", include_in_schema=False)

# Add startup event to create tables and initialize data
@app.on_event("startup")
async def on_startup():
    # Initialize database with tables and initial data
    await init_db()

# Add shutdown event to close resources
@app.on_event("shutdown")
async def on_shutdown():
    # Close any open connections
    await async_engine.dispose()