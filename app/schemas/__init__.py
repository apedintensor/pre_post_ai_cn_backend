"""Lightweight re-exports of actively used schema classes.

Legacy schemas (case metadata, management strategies/plans, legacy diagnosis) removed.
Import from app.schemas.schemas directly for full list.
"""

from .schemas import (
    RoleBase, RoleRead,
    DiagnosisTermBase, DiagnosisTermRead,
    DiagnosisSynonymCreate, DiagnosisSynonymRead, DiagnosisSuggestion,
    CaseCreate, CaseRead,
    ImageCreate, ImageRead,
    AIOutputCreate, AIOutputRead,
    AssessmentCreate, AssessmentRead,
    DiagnosisEntryCreate, DiagnosisEntryRead,
    BlockFeedbackRead,
    StartGameResponse, ActiveGameResponse, ReportCardResponse,
)

__all__ = [name for name in globals().keys() if not name.startswith('_')]