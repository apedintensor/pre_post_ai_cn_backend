✅ 1. Project Overview
Name: Reader Study Web API
Goal: Backend API to manage a clinical reader study, support pre/post-AI image assessments, track confidence, decisions, and integrate with a Vue frontend.
Stack: FastAPI + SQLAlchemy + PostgreSQL (via Alembic), Vue (frontend)

✅ 2. Auth Strategy
Aspect	Strategy
Method	JWT (JSON Web Tokens)
Library	fastapi-users or custom
User Roles	GP, Nurse, Other, Admin
Endpoints	/auth/login, /auth/register, /auth/me
Passwords	Hash via bcrypt

✅ 3. Routes Table
Route	Method	Description	Auth	Related Model
/cases/	GET	List all cases	✅	Case
/cases/{id}	GET	Retrieve case (w/ images, metadata)	✅	Case
/users/me	GET	Current logged-in user info	✅	User
/assessments/	POST	Submit a new assessment (pre/post-AI)	✅	Assessment
/diagnoses/	POST	Submit ranked diagnoses	✅	Diagnosis
/management/	POST	Submit management plan	✅	ManagementPlan
/ai/outputs/{case_id}	GET	Get AI output for case	✅	AIOutput

✅ 4. Schemas (Pydantic)
UserCreate, UserRead, UserUpdate

CaseRead, ImageRead, MetaDataRead

AssessmentCreate, AssessmentRead

DiagnosisCreate, DiagnosisRead

ManagementPlanCreate, ManagementPlanRead

AIOutputRead

✅ 5. Database Models (SQLAlchemy)
Reflecting the normalized DB:

User, Role

Case, Image, CaseMetaData

Assessment, Diagnosis, ManagementPlan

AIOutput

DiagnosisTerm, ManagementStrategy

✅ 6. Error Strategy
Error Type	HTTP Code	Response Format
Validation	422	{ detail: [...] }
Auth Error	401/403	{ detail: "Not authenticated" }
Not Found	404	{ detail: "Not found" }
Server Error	500	{ detail: "Internal server error" }

Standardize via FastAPI exception handlers.

✅ 7. Security Plan
✅ JWT for user authentication

✅ Role-based access (e.g., only Admin can create cases)

✅ SQLAlchemy ORM to prevent SQL injection

✅ Input validation via Pydantic

✅ CORS configured for frontend

✅ Use HTTPS (handled in production/deployment layer)

✅ 8. Frontend Integration Map
Frontend Page	API Used
Login/Register	/auth/*
Case List	/cases/
Case Viewer	/cases/{id}, /images/, /ai/outputs/{id}
Assessment Form	/assessments/, /diagnoses/, /management/
User Profile	/users/me
Admin Dashboard (optional)	/users/, /roles/

✅ 9. Open Questions
Should we use fastapi-users, or roll our own auth system?

Should Admins be able to create/edit cases from frontend?

Will the AI output be pre-generated and stored, or fetched from an AI API?

Will evaluations (e.g., management quality scores) be user-entered or expert-reviewed?