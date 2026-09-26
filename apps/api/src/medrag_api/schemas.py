"""Request and response models of the public API."""

import uuid

from pydantic import BaseModel, Field

from medrag_core.evidence import AnswerFormat, EvidenceStatus

DISCLAIMER = (
    "Research use only. This is not medical advice and must not be used for clinical decisions."
)

EVIDENCE_MESSAGES = {
    EvidenceStatus.SUPPORTED: "Supported by the retrieved literature.",
    EvidenceStatus.PARTIALLY_SUPPORTED: (
        "Partly supported by the retrieved literature; unsupported statements are marked."
    ),
    EvidenceStatus.NOT_SUPPORTED: (
        "Not supported by the retrieved literature: this answer relies on the model's "
        "general medical knowledge."
    ),
    EvidenceStatus.CONTRADICTED: (
        "Contradicted by retrieved literature: review the contradicting passages."
    ),
    EvidenceStatus.NO_EVIDENCE: "No relevant literature was retrieved.",
}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    answer_format: AnswerFormat = Field(
        default=AnswerFormat.FREE,
        description="yes_no and multiple_choice (options in the question) constrain the answer",
    )


class Citation(BaseModel):
    pmid: int
    title: str
    url: str
    passage: str


class StatementEvidenceOut(BaseModel):
    kind: str = Field(default="rationale", description='"claim" (the answer itself) or "rationale"')
    text: str
    support: float = Field(description="best entailment probability over the passages")
    supported: bool
    contradicted: bool
    supporting_pmid: int | None
    contradicting_pmid: int | None


class EvidenceOut(BaseModel):
    status: EvidenceStatus
    supported_fraction: float
    message: str
    statements: list[StatementEvidenceOut]


class AskResponse(BaseModel):
    run_id: uuid.UUID
    question: str
    answer_format: AnswerFormat
    answer: str = Field(description='shown to users; "uncertain" when unsupported yes/no')
    model_answer: str = Field(description="the model's own answer")
    note: str | None = None
    evidence: EvidenceOut
    citations: list[Citation]
    iterations: int
    stop_reason: str
    model: str
    latency_seconds: float
    cost: float
    disclaimer: str = DISCLAIMER


class FeedbackRequest(BaseModel):
    rating: int = Field(description="1 = helpful, -1 = not helpful")
    comment: str | None = Field(default=None, max_length=2000)
