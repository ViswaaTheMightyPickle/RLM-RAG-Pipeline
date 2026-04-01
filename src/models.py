"""Pydantic data models for HR-RAG system."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class RejectedCluster(BaseModel):
    """Model for a rejected cluster entry in discarded_context.json."""

    cluster_id: int = Field(..., description="Unique identifier for the cluster")
    source_pages: str = Field(..., description="Source document/pages reference")
    persona_used: str = Field(..., description="Persona used during distillation")
    summary: str = Field(..., description="The distilled summary text")
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Relevance score (0.0 - 1.0)"
    )
    iteration_count: int = Field(
        ..., ge=1, description="Number of voting iterations performed"
    )
    votes_yes: int = Field(0, ge=0, description="Number of YES votes")
    votes_no: int = Field(0, ge=0, description="Number of NO votes")


class DiscardedContextLog(BaseModel):
    """Model for the discarded_context.json file structure."""

    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO-8601 timestamp",
    )
    root_query: str = Field(..., description="The original user prompt")
    rejected_clusters: list[RejectedCluster] = Field(
        default_factory=list, description="List of rejected cluster entries"
    )


class ProcessingResult(BaseModel):
    """Model for the result of processing a single cluster."""

    cluster_id: int
    source_pages: str
    summary: str
    is_relevant: bool
    relevance_score: Optional[float] = None
    voting_iterations: int = 0
    votes_yes: int = 0
    votes_no: int = 0


class ClusterData(BaseModel):
    """Model for a chunk/cluster of text from the RAG engine."""

    cluster_id: int
    content: str
    source_pages: str
    metadata: dict = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """Model for the final RAG system response."""

    answer: str
    core_question: str
    context_used: list[str] = Field(default_factory=list)
    discarded_count: int = 0
    recovered_count: int = 0
    confidence: str = "UNKNOWN"
    used_discarded_log: bool = False
