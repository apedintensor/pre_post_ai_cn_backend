# Project Log

## 2025-05-01

*   Initialized project structure with FastAPI.
*   Defined initial database schema using DBML (`docs/db.dbml`).
*   Outlined backend design, auth strategy, routes, schemas, and security plan (`docs/backend-design.md`).
*   Created SQLAlchemy models based on DBML (`app/models/models.py`).
*   Created Pydantic schemas for API request/response validation (`app/schemas/schemas.py`).
*   Implemented initial CRUD operations for User, Role, and Case models (`app/crud/`).
*   Implemented CRUD operations for Assessment, Diagnosis, ManagementPlan, AIOutput, DiagnosisTerm, ManagementStrategy, Image, and CaseMetaData models (`app/crud/`).

## 2025-05-02

*   Created FastAPI API endpoint files for all entities (`app/api/endpoints/`).
*   Created database session dependency (`app/api/deps.py`).
*   Updated main API router (`app/api/routes.py`) to include all endpoint routers.
*   Implemented authentication system using fastapi-users (`app/auth/`).
*   Created User model that extends fastapi-users' BaseUserTable with additional fields.
*   Added JWT-based authentication with configurable token lifetime.
*   Set up authentication endpoints for registration, login, and profile management.
*   Added role support for user authentication.
*   Created dependency functions for accessing the current authenticated user.
*   Transitioned to asynchronous SQLite database access for authentication.
*   Updated FastAPI app to initialize database tables asynchronously on startup.
*   Fixed dependency issues by adding `pydantic-settings>=2.0.0` to project dependencies.
*   Resolved CRUD operation access in `init_db.py` by updating from dot notation to dictionary access.
*   Fixed user creation in database initialization by directly instantiating SQLAlchemy user database and UserManager.
*   Created database recreation script (`recreate_db.py`) to resolve schema issues with missing columns.
*   Added root endpoint ("/") to provide API information and documentation links.
*   Added admin endpoint ("/admin") that currently redirects to API documentation for future expansion.
*   Successfully validated application startup with all core functionality working properly.
*   Implemented centralized error handling:
    *   Defined custom exception classes: `EntityNotFoundException`, `PermissionDeniedException`, `DuplicateEntryException`, `GenericServerError` in `app/core/exceptions.py`.
    *   Added exception handlers for each, plus handlers for `RequestValidationError` and `HTTPException`, returning structured JSON error responses.
    *   Registered all handlers globally in `main.py`.
    *   Added logging for 500 and unexpected errors.
    *   Provided reusable helper functions for raising HTTP exceptions.

## Authentication Implementation

I've successfully implemented authentication using fastapi-users in your FastAPI backend project. Here's a summary of what's been set up:

### Authentication System Overview
- **User Model and Database**:
  - Created a User model that extends SQLAlchemyBaseUserTable with your specific fields (role, experience, etc.)
  - Set up async SQLite database support with aiosqlite (successfully installed)
- **Authentication Components**:
  - JWT-based authentication with a 30-day token lifetime
  - Password hashing (handled by fastapi-users)
  - Role support via the role_id field
- **Auth Endpoints**:
  - `/auth/jwt/login` - For user login
  - `/auth/register` - For creating new accounts
  - `/auth/me` - To get the current user's profile
  - Also includes password reset and account verification routes
- **Protected Routes**:
  - Provided dependency functions (current_active_user, current_superuser) for securing routes

### Test Fixes (May 2025)

Fixed several issues in the test suite:

1. **Authentication Dependency Issue**:
   - Problem: Routes were using a nested dependency chain (`current_user = Depends(deps.get_current_user)`) which wasn't being properly enforced in tests.
   - Fix: Updated endpoint files to import and use authentication dependencies directly (`current_user: User = Depends(current_active_user)`)

2. **CRUD Operations Format**:
   - Problem: Code was using object attribute notation (`crud.module.function`) but the CRUD modules were exporting dictionaries.
   - Fix: Updated all endpoint handlers to use dictionary syntax (`crud.module["function"]`)

3. **API Endpoint Paths**:
   - Problem: Tests were using `/api/v1/` prefix for endpoints, but the actual routes didn't include this prefix.
   - Fix: Removed the prefix from test files to match the actual endpoint paths.

4. **Unique Data Generation**:
   - Problem: Tests were creating entities with hardcoded names, causing conflicts when run multiple times.
   - Fix: Added timestamp-based unique names to prevent duplicate key errors.

5. **Schema Validation**:
   - Problem: Some fields were optional when they should be required.
   - Fix: Updated schema definitions to properly mark required fields.

### Future Authentication Improvements

1. **Direct Authentication Dependencies**:
   - Always directly import authentication dependencies from `app.auth.manager` rather than going through `deps.py`.
   - Example: `from app.auth.manager import current_active_user`
   - This ensures the dependency chain is kept simple and works reliably in tests.

2. **Role-Based Authentication**:
   - Implement finer-grained permissions based on user roles.
   - Example: Create custom dependencies like `get_doctor_user` that check both authentication and role.

3. **API Key Support**:
   - For system-to-system communication, consider adding API key authentication alongside JWT.

4. **Enhanced Security**:
   - Set up token refresh mechanism for longer sessions
   - Implement rate limiting to prevent brute force attacks
   - Add IP-based restrictions for sensitive operations
