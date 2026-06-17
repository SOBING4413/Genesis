import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Float, Integer, Boolean,
    DateTime, ForeignKey, JSON, Enum as SAEnum, Index,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from app.database.session import Base
import enum


class PaperStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Paper(Base):
    __tablename__ = "papers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    authors = Column(ARRAY(String), default=[])
    abstract = Column(Text, default="")
    source = Column(String(50), nullable=False, index=True)
    source_id = Column(String(200), unique=True, nullable=True, index=True)
    url = Column(Text, nullable=True)
    pdf_path = Column(Text, nullable=True)
    published_date = Column(DateTime, nullable=True, index=True)
    keywords = Column(ARRAY(String), default=[])
    categories = Column(ARRAY(String), default=[])
    citation_count = Column(Integer, default=0)
    references = Column(ARRAY(String), default=[])
    full_text = Column(Text, default="")
    status = Column(SAEnum(PaperStatus), default=PaperStatus.PENDING, nullable=False, index=True)
    metadata_json = Column(JSON, default=dict)
    embedding_id = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    concepts = relationship("ConceptExtraction", back_populates="paper", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_papers_source_status", "source", "status"),
    )


class ConceptExtraction(Base):
    __tablename__ = "concept_extractions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    concepts = Column(JSON, default=list)
    relationships = Column(JSON, default=list)
    confidence_scores = Column(JSON, default=dict)
    extraction_method = Column(String(50), default="transformer")
    created_at = Column(DateTime, default=datetime.utcnow)

    paper = relationship("Paper", back_populates="concepts")

    __table_args__ = (
        Index("ix_concept_extraction_paper_id", "paper_id"),
    )


class Hypothesis(Base):
    __tablename__ = "hypotheses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    description = Column(Text, default="")
    hypothesis_type = Column(String(50), default="cross_domain", index=True)
    confidence_score = Column(Float, default=0.0, index=True)
    evidence_count = Column(Integer, default=0)
    reasoning_chain = Column(JSON, default=list)
    supporting_papers = Column(ARRAY(String), default=[])
    source_concepts = Column(ARRAY(String), default=[])
    target_concepts = Column(ARRAY(String), default=[])
    status = Column(String(30), default="generated", nullable=False, index=True)
    agent_id = Column(String(200), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_hypotheses_status_confidence", "status", "confidence_score"),
    )


class ResearchAgentLog(Base):
    __tablename__ = "research_agent_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_type = Column(String(50), nullable=False, index=True)
    session_id = Column(String(200), nullable=False, index=True)
    input_data = Column(JSON, default=dict)
    output_data = Column(JSON, default=dict)
    reasoning = Column(Text, default="")
    duration_ms = Column(Integer, default=0)
    success = Column(Boolean, default=True)  # NEW: track success/failure
    error_message = Column(Text, nullable=True)  # NEW: capture error details
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("ix_agent_logs_session", "session_id", "agent_type"),
    )
