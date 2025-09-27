"""Modernized models aligned with docs/db.dbml.

Legacy composite-key assessment/diagnosis structures replaced by:
  - ReaderCaseAssignment (one per user/case/block)
  - Assessment (id PK, assignment FK, phase PRE/POST)
  - DiagnosisEntry (ranked 1..3 per assessment)
  - BlockFeedback (aggregated metrics per block)

Older classes kept removed; if needed, retrieve from VCS history.
"""

from datetime import datetime
from enum import Enum
from sqlalchemy import (
    Column, Integer, String, Boolean, Float, ForeignKey, DateTime, Text, UniqueConstraint, JSON
)
from sqlalchemy.orm import relationship
from app.db.base import Base
from app.auth.models import User


# ---------------- Enums -----------------



class AssessmentPhase(str, Enum):
    PRE = "PRE"
    POST = "POST"




# --------------- Core reference tables ---------------

class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False, index=True)
    users = relationship("User", back_populates="role")


class DiagnosisTerm(Base):
    __tablename__ = "diagnosis_terms"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False, index=True)
    cases = relationship("Case", back_populates="ground_truth_diagnosis")
    ai_outputs = relationship("AIOutput", back_populates="prediction")
    diagnosis_entries = relationship("DiagnosisEntry", back_populates="diagnosis_term")
    synonyms = relationship("DiagnosisSynonym", back_populates="term", cascade="all, delete-orphan")


# --------------- Case & image & AI outputs ---------------

class Case(Base):
    __tablename__ = "cases"
    id = Column(Integer, primary_key=True)
    ground_truth_diagnosis_id = Column(Integer, ForeignKey("diagnosis_terms.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    ai_predictions_json = Column(JSON)  # full probability vector (term_id -> score)
    # Legacy/analysis flag retained for compatibility with tests
    typical_diagnosis = Column(Boolean)

    ground_truth_diagnosis = relationship("DiagnosisTerm", back_populates="cases")
    images = relationship("Image", back_populates="case")
    ai_outputs = relationship("AIOutput", back_populates="case")
    assignments = relationship("ReaderCaseAssignment", back_populates="case")


class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False, unique=True)
    image_url = Column(String)
    case = relationship("Case", back_populates="images")

    @property
    def full_url(self) -> str:
        """Return absolute URL if IMAGE_BASE_URL configured, else raw image_url.

        If IMAGE_BASE_URL ends with '/', don't duplicate slash.
        """
        from app.core.config import settings  # local import to avoid circular at module import
        base = settings.IMAGE_BASE_URL.strip()
        if not base:
            return self.image_url or ""
        if not self.image_url:
            return base.rstrip('/') + '/'
        if base.endswith('/'):
            return base + self.image_url.lstrip('/')
        return base + '/' + self.image_url.lstrip('/')


class AIOutput(Base):
    __tablename__ = "ai_outputs"
    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    rank = Column(Integer, nullable=False)
    prediction_id = Column(Integer, ForeignKey("diagnosis_terms.id"), nullable=False)
    confidence_score = Column(Float)

    case = relationship("Case", back_populates="ai_outputs")
    prediction = relationship("DiagnosisTerm", back_populates="ai_outputs")
    __table_args__ = (UniqueConstraint("case_id", "rank", name="uix_case_rank"),)


# --------------- Game / assignment & assessments ---------------

class ReaderCaseAssignment(Base):
    __tablename__ = "reader_case_assignments"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    display_order = Column(Integer, nullable=False)
    block_index = Column(Integer, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_pre_at = Column(DateTime)
    completed_post_at = Column(DateTime)

    user = relationship("User", back_populates="assignments")
    case = relationship("Case", back_populates="assignments")
    assessments = relationship("Assessment", back_populates="assignment")
    __table_args__ = (UniqueConstraint("user_id", "case_id", name="uix_user_case"),)


class Assessment(Base):
    __tablename__ = "assessments"
    id = Column(Integer, primary_key=True)
    assignment_id = Column(Integer, ForeignKey("reader_case_assignments.id"), nullable=False)
    phase = Column(String, nullable=False)  # PRE / POST
    diagnostic_confidence = Column(Integer)
    management_confidence = Column(Integer)
    # New management action fields
    investigation_action = Column(String)  # NONE | BIOPSY | OTHER
    next_step_action = Column(String)      # REASSURE | MANAGE_MYSELF | REFER
    # Legacy fields removed (use investigation_action / next_step_action only)
    # Post-AI only
    changed_primary_diagnosis = Column(Boolean)
    changed_management_plan = Column(Boolean)
    ai_usefulness = Column(String)
    # Correctness flags
    top1_correct = Column(Boolean)
    top3_correct = Column(Boolean)
    rank_of_truth = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    assignment = relationship("ReaderCaseAssignment", back_populates="assessments")
    diagnosis_entries = relationship("DiagnosisEntry", back_populates="assessment", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("assignment_id", "phase", name="uix_assignment_phase"),)


class DiagnosisEntry(Base):
    __tablename__ = "diagnosis_entries"
    id = Column(Integer, primary_key=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id"), nullable=False)
    rank = Column(Integer, nullable=False)
    raw_text = Column(String)
    diagnosis_term_id = Column(Integer, ForeignKey("diagnosis_terms.id"), nullable=False)

    assessment = relationship("Assessment", back_populates="diagnosis_entries")
    diagnosis_term = relationship("DiagnosisTerm", back_populates="diagnosis_entries")
    __table_args__ = (UniqueConstraint("assessment_id", "rank", name="uix_assessment_rank"),)


class BlockFeedback(Base):
    __tablename__ = "block_feedback"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    block_index = Column(Integer, nullable=False)
    stats_json = Column(JSON)
    top1_accuracy_pre = Column(Float)
    top1_accuracy_post = Column(Float)
    top3_accuracy_pre = Column(Float)
    top3_accuracy_post = Column(Float)
    delta_top1 = Column(Float)
    delta_top3 = Column(Float)
    # Peer average accuracy (not percentile) - post only legacy aggregate
    # (Removed legacy aggregate post-only averages peer_avg_top1 / peer_avg_top3)
    # Explicit pre/post peer averages
    peer_avg_top1_pre = Column(Float)
    peer_avg_top1_post = Column(Float)
    peer_avg_top3_pre = Column(Float)
    peer_avg_top3_post = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("user_id", "block_index", name="uix_user_block"),)


class ManagementStrategy(Base):
    __tablename__ = "management_strategies"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False, index=True)


class CaseMetaData(Base):
    __tablename__ = "case_metadata"
    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False, unique=True)
    age = Column(Integer)
    gender = Column(String)
    fever_history = Column(Boolean)
    psoriasis_history = Column(Boolean)
    other_notes = Column(Text)

# ---------------- User back-pop references ----------------
User.role = relationship("Role", back_populates="users")
User.assignments = relationship("ReaderCaseAssignment", back_populates="user")
User.assessments = relationship("Assessment", secondary="reader_case_assignments", viewonly=True)


class DiagnosisSynonym(Base):
    __tablename__ = "diagnosis_synonyms"
    id = Column(Integer, primary_key=True)
    diagnosis_term_id = Column(Integer, ForeignKey("diagnosis_terms.id"), nullable=False, index=True)
    synonym = Column(String, nullable=False, index=True, unique=True)

    term = relationship("DiagnosisTerm", back_populates="synonyms")

    def __repr__(self):  # pragma: no cover
        return f"<DiagnosisSynonym term_id={self.diagnosis_term_id} synonym='{self.synonym}'>"

