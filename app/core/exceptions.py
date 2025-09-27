"""
Centralized exception classes and handlers for FastAPI project.
"""
import logging
from typing import Union, Optional
from fastapi import Request, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

logger = logging.getLogger("app.core.exceptions")

# --- Custom Exception Classes ---
class EntityNotFoundException(Exception):
    """404: Raised when a requested entity is not found."""
    def __init__(self, entity_type: str, entity_id: Union[str, int], detail: Optional[str] = None):
        self.detail = detail or f"{entity_type} with id '{entity_id}' not found"
        super().__init__(self.detail)

class PermissionDeniedException(Exception):
    """403: Raised when a user lacks permission for an operation."""
    def __init__(self, detail: str = "You do not have permission to perform this action"):
        self.detail = detail
        super().__init__(detail)

class DuplicateEntryException(Exception):
    """409: Raised when attempting to create a duplicate entity."""
    def __init__(self, entity_type: str, identifier: str, detail: Optional[str] = None):
        self.detail = detail or f"{entity_type} with {identifier} already exists"
        super().__init__(self.detail)

class GenericServerError(Exception):
    """500: Generic server error for unexpected exceptions."""
    def __init__(self, detail: str = "An unexpected error occurred"):
        self.detail = detail
        super().__init__(detail)

# --- Exception Handlers ---
async def entity_not_found_exception_handler(request: Request, exc: EntityNotFoundException):
    return JSONResponse(status_code=404, content={"detail": exc.detail})

async def permission_denied_exception_handler(request: Request, exc: PermissionDeniedException):
    return JSONResponse(status_code=403, content={"detail": exc.detail})

async def duplicate_entry_exception_handler(request: Request, exc: DuplicateEntryException):
    return JSONResponse(status_code=409, content={"detail": exc.detail})

async def generic_server_error_handler(request: Request, exc: GenericServerError):
    logger.error(f"Server error: {exc.detail}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "An internal server error occurred"})

async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=getattr(exc, "headers", None))

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

async def pydantic_validation_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

async def unexpected_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred"})

# --- Helper Functions ---
def raise_not_found(entity_type: str, entity_id: Union[str, int], detail: Optional[str] = None):
    raise EntityNotFoundException(entity_type, entity_id, detail)

def raise_permission_denied(detail: Optional[str] = None):
    raise PermissionDeniedException(detail)

def raise_duplicate(entity_type: str, identifier: str, detail: Optional[str] = None):
    raise DuplicateEntryException(entity_type, identifier, detail)

def raise_server_error(detail: str = "An unexpected error occurred"):
    raise GenericServerError(detail)