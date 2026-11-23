"""Pydantic request and response schemas for the API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1)
    reference_answer: str | None = Field(default=None, min_length=1)
    providers: list[Literal["gemini", "nvidia", "groq", "ollama"]] | None = Field(
        default=None,
        min_length=1,
    )

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must not be blank")
        return value

    @field_validator("reference_answer")
    @classmethod
    def reference_answer_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Reference answer must not be blank")
        return value

    @field_validator("providers")
    @classmethod
    def providers_must_be_unique(
        cls,
        value: list[Literal["gemini", "nvidia", "groq", "ollama"]] | None,
    ) -> list[Literal["gemini", "nvidia", "groq", "ollama"]] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("Providers must not contain duplicates")
        return value


class ScoreBreakdown(BaseModel):
    accuracy: int | None = Field(default=None, ge=1, le=5)
    relevance: int | None = Field(default=None, ge=1, le=5)
    completeness: int | None = Field(default=None, ge=1, le=5)
    clarity: int | None = Field(default=None, ge=1, le=5)
    instruction_following: int | None = Field(default=None, ge=1, le=5)
    formatting: int | None = Field(default=None, ge=1, le=5)


class MetricEvidence(BaseModel):
    reason: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    missing_requirements: list[str] = Field(default_factory=list)


class EvaluationEvidence(BaseModel):
    response_quality: MetricEvidence
    reasoning_clarity: MetricEvidence
    factual_support: MetricEvidence
    instruction_following: MetricEvidence
    formatting: MetricEvidence
    completeness: MetricEvidence


class SimpleChecks(BaseModel):
    word_count: int = Field(ge=0)
    ending_check: bool | None = None


class ModelRunResult(BaseModel):
    model: str
    provider: str
    model_id: str
    response: str | None
    http_status: int | None
    response_time_ms: float
    error: str | None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    simple_checks: SimpleChecks | None = None
    scores: ScoreBreakdown | None = None
    score_evidence: EvaluationEvidence | None = None
    score_error: str | None = None
    score_source: Literal["ai_rubric"] | None = None
    score_rationale: str | None = None
    quality_score: float | None = None
    rank: int | None = None


class CompareResponse(BaseModel):
    prompt: str
    status: Literal["success", "partial_failure", "failed"]
    evaluation_status: Literal["not_run", "success", "partial_failure", "failed"]
    evaluation_model: str | None = None
    evaluation_error: str | None = None
    responses: list[ModelRunResult]
    winner_model: str | None = None
    winner_models: list[str] = Field(default_factory=list)
    best_response: ModelRunResult | None = None
    best_responses: list[ModelRunResult] = Field(default_factory=list)


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("Conversation title must not be blank")
        return value


class ConversationRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Conversation title must not be blank")
        return value


class ConversationTurnRequest(CompareRequest):
    pass


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    turn_count: int = Field(ge=0)


class ConversationTurn(BaseModel):
    id: int
    prompt: str
    reference_answer: str | None
    created_at: str
    result: CompareResponse


class ConversationDetail(ConversationSummary):
    turns: list[ConversationTurn]
