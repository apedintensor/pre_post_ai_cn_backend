# backend/app/crud/__init__.py
from .crud_user import user
from .crud_role import role
from .crud_case import case
from .crud_ai_output import ai_output
from .crud_diagnosis_term import diagnosis_term, synonyms as crud_diagnosis_synonyms, synonyms
from .crud_image import image
from .crud_management_strategy import management_strategy
from .crud_case_metadata import case_metadata
"""Aggregate CRUD objects for active parts.

`diagnosis_term` basic term CRUD.
`crud_diagnosis_synonyms` / `synonyms` provide synonym create/list/suggest.
Legacy CRUD modules removed after refactor.
"""

# Import other crud objects as they are created
# ... and so on
