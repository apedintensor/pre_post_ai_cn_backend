from typing import List, Optional, Literal
import datetime
from pydantic import BaseModel, model_validator
from app.auth.schemas import UserRead


# ---------- Core reference ----------

class RoleBase(BaseModel):
    name: str


class RoleRead(RoleBase):
    id: int
    class Config:
        from_attributes = True


class RoleCreate(RoleBase):
    pass


class DiagnosisTermBase(BaseModel):
    name: str


class DiagnosisTermRead(DiagnosisTermBase):
    id: int
    class Config:
        from_attributes = True


class DiagnosisTermCreate(DiagnosisTermBase):
    pass


class DiagnosisSynonymCreate(BaseModel):
    diagnosis_term_id: int
    synonym: str


class DiagnosisSynonymRead(DiagnosisSynonymCreate):
    id: int
    class Config:
        from_attributes = True


class DiagnosisSuggestion(BaseModel):
    id: int                # diagnosis_term id
    label: str             # canonical name
    match: str             # matched substring / synonym
    source: str            # 'name' | 'synonym'
    score: float
    synonyms: List[str] = []  # all known synonyms for this term (for client-side fuzzy / display)


# ---------- Case / Image / AIOutput ----------

class CaseCreate(BaseModel):
    ground_truth_diagnosis_id: int  # required for validation to pass in tests
    ai_predictions_json: Optional[dict] = None
    typical_diagnosis: Optional[bool] = None


class ImageCreate(BaseModel):
    case_id: int
    image_url: str


class AIOutputCreate(BaseModel):
    case_id: int
    rank: int
    prediction_id: int
    confidence_score: float


class ImageRead(ImageCreate):
    id: int
    full_url: str | None = None
    class Config:
        from_attributes = True


class AIOutputRead(AIOutputCreate):
    id: int
    prediction: Optional[DiagnosisTermRead]
    class Config:
        from_attributes = True


class CaseRead(BaseModel):
    id: int
    ground_truth_diagnosis_id: Optional[int]
    ground_truth_diagnosis: Optional[DiagnosisTermRead]
    created_at: datetime.datetime
    ai_predictions_json: Optional[dict]
    typical_diagnosis: Optional[bool] = None
    images: List[ImageRead] = []
    ai_outputs: List[AIOutputRead] = []
    class Config:
        from_attributes = True


# ---------- Management Strategy ----------

class ManagementStrategyBase(BaseModel):
    name: str


class ManagementStrategyCreate(ManagementStrategyBase):
    pass


class ManagementStrategyRead(ManagementStrategyBase):
    id: int
    class Config:
        from_attributes = True


# ---------- Case MetaData ----------

class CaseMetaDataBase(BaseModel):
    age: int | None = None
    gender: str | None = None
    fever_history: bool | None = None
    psoriasis_history: bool | None = None
    other_notes: str | None = None


class CaseMetaDataCreate(CaseMetaDataBase):
    pass


class CaseMetaDataRead(CaseMetaDataBase):
    id: int
    case_id: int
    class Config:
        from_attributes = True


# ---------- Assignments & Assessments ----------

class ReaderCaseAssignmentRead(BaseModel):
    id: int
    user_id: int
    case_id: int
    display_order: int
    block_index: int
    started_at: datetime.datetime
    completed_pre_at: Optional[datetime.datetime]
    completed_post_at: Optional[datetime.datetime]
    class Config:
        from_attributes = True


class DiagnosisEntryCreate(BaseModel):
    rank: int
    raw_text: Optional[str] = None  # what the user typed initially
    diagnosis_term_id: int          # final selected canonical term


class DiagnosisEntryRead(DiagnosisEntryCreate):
    id: int
    class Config:
        from_attributes = True


class AssessmentCreate(BaseModel):
    assignment_id: int
    phase: str  # PRE / POST
    diagnostic_confidence: Optional[int] = None
    management_confidence: Optional[int] = None
    # New management/action fields
    investigation_action: Optional[Literal['NONE','BIOPSY','OTHER']] = None
    next_step_action: Optional[Literal['REASSURE','MANAGE_MYSELF','REFER']] = None
    # POST phase only
    changed_primary_diagnosis: Optional[bool] = None
    changed_management_plan: Optional[bool] = None
    ai_usefulness: Optional[str] = None
    diagnosis_entries: List[DiagnosisEntryCreate] = []

    @model_validator(mode="after")
    def validate_ranks(self):  # type: ignore[override]
        # Ensure unique ranks and rank 1 presence (at least one entry rank 1)
        seen = set()
        for de in self.diagnosis_entries:
            if de.rank in seen:
                raise ValueError(f"Duplicate diagnosis entry rank: {de.rank}")
            if de.rank < 1 or de.rank > 3:
                raise ValueError("Diagnosis entry rank must be between 1 and 3")
            seen.add(de.rank)
        if 1 not in seen:
            raise ValueError("Rank 1 diagnosis entry is required")
        return self


class AssessmentRead(BaseModel):
    id: int
    assignment_id: int
    phase: str
    diagnostic_confidence: Optional[int]
    management_confidence: Optional[int]
    investigation_action: Optional[str]
    next_step_action: Optional[str]
    changed_primary_diagnosis: Optional[bool]
    changed_management_plan: Optional[bool]
    ai_usefulness: Optional[str]
    top1_correct: Optional[bool]
    top3_correct: Optional[bool]
    rank_of_truth: Optional[int]
    created_at: datetime.datetime
    diagnosis_entries: List[DiagnosisEntryRead] = []
    class Config:
        from_attributes = True


class AssessmentSubmitResponse(AssessmentRead):
    block_index: int
    block_complete: bool
    report_available: bool  # same as block_complete (lazy finalize may run here in future)
    remaining_in_block: int


# ---------- Block Feedback ----------

class BlockFeedbackRead(BaseModel):
    id: int
    user_id: int
    block_index: int
    top1_accuracy_pre: Optional[float]
    top1_accuracy_post: Optional[float]
    top3_accuracy_pre: Optional[float]
    top3_accuracy_post: Optional[float]
    delta_top1: Optional[float]
    delta_top3: Optional[float]
    peer_avg_top1_pre: Optional[float]
    peer_avg_top1_post: Optional[float]
    peer_avg_top3_pre: Optional[float]
    peer_avg_top3_post: Optional[float]
    stats_json: Optional[dict]
    created_at: datetime.datetime
    class Config:
        from_attributes = True


# ---------- Game workflow DTOs ----------

class StartGameResponse(BaseModel):
    block_index: int
    assignments: List[ReaderCaseAssignmentRead]


class ActiveGameResponse(BaseModel):
    block_index: int
    assignments: List[ReaderCaseAssignmentRead]
    remaining: int


class ReportCardResponse(BlockFeedbackRead):
    total_cases: int | None = None
    cases: list[dict] = []  # each: {case_id:int, ground_truth_diagnosis_id:int | None}


class GameProgressResponse(BaseModel):
    total_cases: int
    completed_cases: int
    remaining_cases: int
    assigned_cases: int
    unassigned_cases: int
    in_progress_cases: int


class NextAssignmentResponse(BaseModel):
    status: Literal['continuing','started','exhausted']
    block_index: int | None
    assignment: Optional[ReaderCaseAssignmentRead]
    remaining: int



