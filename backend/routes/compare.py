"""Concurrent multi-provider comparison route."""

import asyncio
from time import perf_counter

from fastapi import APIRouter

from backend.models.schemas import CompareRequest, CompareResponse, ModelRunResult
from backend.services.evaluator import (
    EvaluationError,
    calculate_simple_checks,
    evaluate_responses,
    get_evaluation_model_id,
    score_and_rank,
)
from backend.services.llm_clients import (
    ConversationHistory,
    ProviderError,
    estimate_cost_usd,
    generate_response,
    get_provider_config,
    SUPPORTED_MODELS,
)

router = APIRouter()


async def _run_model(
    model: str,
    prompt: str,
    conversation_history: ConversationHistory = (),
) -> ModelRunResult:
    config = get_provider_config(model)
    started_at = perf_counter()
    try:
        if conversation_history:
            result = await generate_response(
                model,
                prompt,
                conversation_history=conversation_history,
            )
        else:
            result = await generate_response(model, prompt)
    except ProviderError as exc:
        return ModelRunResult(
            model=model,
            provider=config.provider,
            model_id=config.model_id,
            response=None,
            http_status=exc.http_status,
            response_time_ms=round((perf_counter() - started_at) * 1000, 2),
            error=str(exc),
        )

    return ModelRunResult(
        model=model,
        provider=result.provider,
        model_id=result.model_id,
        response=result.response,
        http_status=result.http_status,
        response_time_ms=round((perf_counter() - started_at) * 1000, 2),
        error=None,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        estimated_cost_usd=estimate_cost_usd(
            result.model_id,
            result.prompt_tokens,
            result.completion_tokens,
        ),
        simple_checks=calculate_simple_checks(result.response),
    )


async def run_comparison(
    request: CompareRequest,
    conversation_contexts: dict[str, ConversationHistory] | None = None,
) -> CompareResponse:
    models_to_run = request.providers or SUPPORTED_MODELS
    conversation_contexts = conversation_contexts or {}
    responses = await asyncio.gather(
        *(
            _run_model(
                model,
                request.prompt,
                conversation_contexts.get(model, ()),
            )
            for model in models_to_run
        )
    )
    successful_count = sum(response.error is None for response in responses)

    if successful_count == len(responses):
        status = "success"
    elif successful_count:
        status = "partial_failure"
    else:
        status = "failed"

    successful_responses = [
        response
        for response in responses
        if response.error is None and response.response is not None
    ]
    evaluation_status = "not_run"
    evaluation_model = None
    evaluation_error = None
    winner_models: list[str] = []
    best_response = None
    best_responses: list[ModelRunResult] = []

    if successful_responses:
        evaluation_status = "failed"
        evaluation_model = get_evaluation_model_id(successful_responses)
        try:
            if any(conversation_contexts.values()):
                evaluations = await evaluate_responses(
                    request.prompt,
                    request.reference_answer,
                    successful_responses,
                    conversation_contexts=conversation_contexts,
                )
            else:
                evaluations = await evaluate_responses(
                    request.prompt,
                    request.reference_answer,
                    successful_responses,
                )
        except EvaluationError as exc:
            evaluation_error = str(exc)
        else:
            winner_models, best_response, best_responses = score_and_rank(
                responses,
                evaluations.scores,
            )
            for response in responses:
                response.score_error = evaluations.errors.get(response.model)
            if evaluations.errors:
                evaluation_error = "; ".join(
                    f"{model}: {error}"
                    for model, error in evaluations.errors.items()
                )
                evaluation_status = (
                    "partial_failure"
                    if evaluations.scores
                    else "failed"
                )
            else:
                evaluation_status = "success"

    return CompareResponse(
        prompt=request.prompt,
        status=status,
        evaluation_status=evaluation_status,
        evaluation_model=evaluation_model,
        evaluation_error=evaluation_error,
        responses=responses,
        winner_model=winner_models[0] if len(winner_models) == 1 else None,
        winner_models=winner_models,
        best_response=best_response,
        best_responses=best_responses,
    )


@router.post("/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest) -> CompareResponse:
    return await run_comparison(request)
