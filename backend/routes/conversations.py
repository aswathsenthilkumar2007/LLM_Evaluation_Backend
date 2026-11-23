"""Conversation lifecycle and context-aware comparison endpoints."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Response, status

from backend.models.schemas import (
    CompareRequest,
    ConversationCreateRequest,
    ConversationDetail,
    ConversationRenameRequest,
    ConversationSummary,
    ConversationTurn,
    ConversationTurnRequest,
)
from backend.services.conversation_store import (
    add_turn,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    rename_conversation,
)
from backend.services.llm_clients import (
    ConversationHistory,
    SUPPORTED_MODELS,
    limit_conversation_history,
)
from backend.routes.compare import run_comparison

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post(
    "",
    response_model=ConversationSummary,
    status_code=status.HTTP_201_CREATED,
)
def create_new_conversation(
    request: ConversationCreateRequest,
) -> dict[str, object]:
    return create_conversation(request.title)


@router.get("", response_model=list[ConversationSummary])
def get_conversations() -> list[dict[str, object]]:
    return list_conversations()


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation_detail(conversation_id: UUID) -> dict[str, object]:
    conversation = get_conversation(str(conversation_id))
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return conversation


@router.patch(
    "/{conversation_id}",
    response_model=ConversationSummary,
)
def rename_existing_conversation(
    conversation_id: UUID,
    request: ConversationRenameRequest,
) -> dict[str, object]:
    if not rename_conversation(str(conversation_id), request.title):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    conversation = get_conversation(str(conversation_id))
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    conversation.pop("turns", None)
    return conversation


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_existing_conversation(conversation_id: UUID) -> Response:
    if not delete_conversation(str(conversation_id)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{conversation_id}/turns",
    response_model=ConversationTurn,
)
async def submit_conversation_turn(
    conversation_id: UUID,
    request: ConversationTurnRequest,
) -> dict[str, object]:
    conversation_id = str(conversation_id)
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    conversation_contexts = _provider_contexts(conversation["turns"])
    comparison_request = CompareRequest(
        prompt=request.prompt,
        reference_answer=request.reference_answer,
        providers=request.providers,
    )
    result = await run_comparison(comparison_request, conversation_contexts)
    turn = add_turn(
        conversation_id,
        request.prompt,
        request.reference_answer,
        result,
    )
    if turn is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation was deleted before the turn could be saved",
        )
    return turn


def _provider_contexts(
    turns: list[dict[str, object]],
) -> dict[str, ConversationHistory]:
    histories: dict[str, list[tuple[str, str | None]]] = {
        model: [] for model in SUPPORTED_MODELS
    }
    for turn in turns:
        prompt = turn["prompt"]
        result = turn["result"]
        if not isinstance(prompt, str) or not isinstance(result, dict):
            continue
        responses = result.get("responses")
        responses_by_model = {
            response.get("model"): response
            for response in responses
            if isinstance(response, dict)
        } if isinstance(responses, list) else {}
        for model in SUPPORTED_MODELS:
            response = responses_by_model.get(model)
            answer = (
                response.get("response")
                if isinstance(response, dict)
                and response.get("error") is None
                and isinstance(response.get("response"), str)
                else None
            )
            histories[model].append((prompt, answer))

    return {
        model: limit_conversation_history(history)
        for model, history in histories.items()
    }
