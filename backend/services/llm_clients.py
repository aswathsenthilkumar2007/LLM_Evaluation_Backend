"""Provider-specific LLM request clients."""

from __future__ import annotations
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

REQUEST_TIMEOUT_SECONDS = 60.0
NVIDIA_REQUEST_TIMEOUT_SECONDS = 180.0
MAX_CONVERSATION_CONTEXT_TOKENS = 4096
MODEL_PRICING_ENV = "MODEL_PRICING_USD_PER_1M_TOKENS"
SUPPORTED_MODELS = ("gemini", "nvidia", "groq", "ollama")
DEFAULT_OLLAMA_BASE_URL = "https://ollama.com"

DEFAULT_MODEL_IDS = {
    "gemini": "gemini-3.5-flash-lite",
    "nvidia": "nvidia/nemotron-3.5-bolt-e4b",
    "groq": "openai/gpt-oss-20b",
    "ollama": "llama3.2",
}

RETIRED_NVIDIA_MODEL_IDS = {"meta/llama-3.2-3b-instruct"}

MODEL_ENV_VARS = {
    "gemini": "GEMINI_MODEL",
    "nvidia": "NVIDIA_MODEL",
    "ollama": "OLLAMA_MODEL",
}

PROVIDER_NAMES = {
    "gemini": "Google Gemini",
    "nvidia": "NVIDIA NIM",
    "groq": "Groq",
    "ollama": "Ollama",
}

CHAT_COMPLETIONS_ENDPOINTS = {
    "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
}

HTTP_ERROR_MESSAGES = {
    401: "authentication failed",
    403: "access forbidden",
    404: "model or endpoint not found",
    410: "model or endpoint has been retired",
    422: "request was rejected by the provider",
    429: "provider rate limit exceeded",
    500: "provider reported an internal server error",
}

logger = logging.getLogger(__name__)

ConversationHistory = Sequence[tuple[str, str | None]]


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model_id: str
    endpoint: str
    required_envs: tuple[str, ...]


@dataclass(frozen=True)
class ProviderResponse:
    model: str
    provider: str
    model_id: str
    response: str
    http_status: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def estimate_cost_usd(
    model_id: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    """Estimate cost from exact-model rates in MODEL_PRICING_USD_PER_1M_TOKENS."""
    raw_pricing = os.getenv(MODEL_PRICING_ENV, "").strip()
    if not raw_pricing or prompt_tokens is None or completion_tokens is None:
        return None

    try:
        pricing = json.loads(raw_pricing)
    except json.JSONDecodeError:
        logger.warning("%s must contain a JSON object; cost is unavailable", MODEL_PRICING_ENV)
        return None

    if not isinstance(pricing, dict):
        logger.warning("%s must contain a JSON object; cost is unavailable", MODEL_PRICING_ENV)
        return None

    rates = pricing.get(model_id)
    if not isinstance(rates, dict):
        return None

    input_rate = _nonnegative_rate(rates.get("input"))
    output_rate = _nonnegative_rate(rates.get("output"))
    if input_rate is None or output_rate is None:
        logger.warning(
            "Pricing for model %s must include nonnegative input and output rates",
            model_id,
        )
        return None

    return (
        prompt_tokens * input_rate + completion_tokens * output_rate
    ) / 1_000_000


def _nonnegative_rate(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0 <= value < float("inf"):
        return None
    return float(value)


def limit_conversation_history(
    history: ConversationHistory,
) -> tuple[tuple[str, str | None], ...]:
    """Keep the newest complete turns within a conservative context budget."""
    selected: list[tuple[str, str | None]] = []
    estimated_tokens = 0
    for user_message, assistant_message in reversed(history):
        turn_tokens = 8 + sum(
            (len(message) + 2) // 3
            for message in (user_message, assistant_message)
            if message is not None
        )
        if estimated_tokens + turn_tokens > MAX_CONVERSATION_CONTEXT_TOKENS:
            break
        selected.append((user_message, assistant_message))
        estimated_tokens += turn_tokens
    return tuple(reversed(selected))


class ProviderError(Exception):
    """A provider call failure with any HTTP status that was received."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


def get_provider_config(model: str) -> ProviderConfig:
    """Return provider metadata using the current environment configuration."""
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}")

    if model == "groq":
        model_id = DEFAULT_MODEL_IDS[model]
    else:
        model_id = os.getenv(MODEL_ENV_VARS[model], DEFAULT_MODEL_IDS[model]).strip()
        if not model_id:
            model_id = DEFAULT_MODEL_IDS[model]

    if model == "gemini":
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(model_id, safe='-._')}:generateContent"
        )
        required_envs = ("GEMINI_API_KEY",)
    elif model == "ollama":
        base_url = (
            os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")
            or DEFAULT_OLLAMA_BASE_URL
        )
        endpoint = f"{base_url}/api/chat"
        required_envs = ("OLLAMA_API_KEY",)
    else:
        endpoint = CHAT_COMPLETIONS_ENDPOINTS[model]
        required_envs = (f"{model.upper()}_API_KEY",)

    return ProviderConfig(
        provider=PROVIDER_NAMES[model],
        model_id=model_id,
        endpoint=endpoint,
        required_envs=required_envs,
    )



async def generate_response(
    model: str,
    prompt: str,
    *,
    temperature: float | None = None,
    system_prompt: str | None = None,
    response_format: Literal["json_object"] | None = None,
    conversation_history: ConversationHistory = (),
) -> ProviderResponse:
    """Send one prompt to a configured provider and return its text and status."""

    config = get_provider_config(model)

    print("DEBUG PROVIDER:", model)
    print("DEBUG MODEL ID:", config.model_id)
    print("DEBUG ENDPOINT:", config.endpoint)

    credentials = {
        env_name: os.getenv(env_name, "").strip()
        for env_name in config.required_envs
    }

    missing = [name for name, value in credentials.items() if not value]

    if missing:
        raise ProviderError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )

    if model == "nvidia":
        _validate_nvidia_free_model(config.model_id)

    headers, body = _request_data(
        model,
        config,
        prompt,
        credentials,
        temperature=temperature,
        system_prompt=system_prompt,
        response_format=response_format,
        conversation_history=limit_conversation_history(conversation_history),
    )
    secrets = tuple(value for value in credentials.values() if value)

    timeout_seconds = (
        NVIDIA_REQUEST_TIMEOUT_SECONDS
        if model == "nvidia"
        else REQUEST_TIMEOUT_SECONDS
    )

    try:
        async with httpx.AsyncClient(
            timeout=timeout_seconds
        ) as client:

            for attempt in range(3):
                try:
                    response = await client.post(
                        config.endpoint,
                        headers=headers,
                        json=body,
                    )
                except (httpx.ConnectError, httpx.RemoteProtocolError):
                    if model != "gemini" or attempt == 2:
                        raise
                    logger.warning(
                        "Gemini connection dropped before a response; "
                        "retrying attempt %d of 3",
                        attempt + 1,
                    )
                    await asyncio.sleep(2 ** attempt)
                    continue

                if (
                    model == "gemini"
                    and response.status_code == 503
                    and attempt < 2
                ):
                    await asyncio.sleep(2 ** attempt)
                    continue

                break

    except httpx.TimeoutException as exc:
        raise ProviderError(
            f"{config.provider} request timed out"
        ) from exc

    except httpx.ConnectError as exc:
        if model == "ollama":
            base_url = config.endpoint.rsplit("/api/chat", 1)[0]
            raise ProviderError(
                f"Ollama is unavailable at {base_url}; "
                "ensure the service is running"
            ) from exc

        message = _redact_secrets(str(exc), secrets)
        raise ProviderError(
            f"{config.provider} connection failed: {message}"
        ) from exc

    except httpx.RequestError as exc:
        message = _redact_secrets(str(exc), secrets)
        raise ProviderError(
            f"{config.provider} request failed: {message}"
        ) from exc

    if not response.is_success:
        raise _http_error(
            config.provider, response, secrets, config.model_id
        )

    payload = _response_json(response, config.provider)
    text = _response_text(model, payload)
    prompt_tokens, completion_tokens = _response_usage(model, payload)

    if not text.strip():
        raise ProviderError(
            f"{config.provider} returned an empty response",
            http_status=response.status_code,
        )

    return ProviderResponse(
        model=model,
        provider=config.provider,
        model_id=config.model_id,
        response=text,
        http_status=response.status_code,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )




def _request_data(
    model: str,
    config: ProviderConfig,
    prompt: str,
    credentials: dict[str, str],
    *,
    temperature: float | None = None,
    system_prompt: str | None = None,
    response_format: Literal["json_object"] | None = None,
    conversation_history: ConversationHistory = (),
) -> tuple[dict[str, str], dict[str, Any]]:
    if response_format is not None and model != "nvidia":
        raise ValueError("Structured response format is only supported for NVIDIA")

    headers = {"Content-Type": "application/json"}
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for user_message, assistant_message in conversation_history:
        messages.append({"role": "user", "content": user_message})
        if assistant_message is not None:
            messages.append(
                {"role": "assistant", "content": assistant_message}
            )
    messages.append({"role": "user", "content": prompt})

    if model == "gemini":
        headers["x-goog-api-key"] = credentials["GEMINI_API_KEY"]
        contents = []
        for user_message, assistant_message in conversation_history:
            contents.append(
                {"role": "user", "parts": [{"text": user_message}]}
            )
            if assistant_message is not None:
                contents.append(
                    {"role": "model", "parts": [{"text": assistant_message}]}
                )
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        body = {"contents": contents}

    elif model == "ollama":
        headers["Authorization"] = (
            f"Bearer {credentials['OLLAMA_API_KEY']}"
        )
        body = {
            "model": config.model_id,
            "messages": messages,
            "stream": False,
        }

    else:
        key_name = f"{model.upper()}_API_KEY"
        headers["Authorization"] = f"Bearer {credentials[key_name]}"

        body = {
            "model": config.model_id,
            "messages": messages,
            "stream": False,
        }

        if model == "nvidia":
            headers["Accept"] = "application/json"
            body["max_tokens"] = 2048
            body["temperature"] = 0.7 if temperature is None else temperature
            body["chat_template_kwargs"] = {
                "enable_thinking": False
            }
            if response_format is not None:
                body["response_format"] = {"type": response_format}

    return headers, body




def _validate_nvidia_free_model(model_id: str) -> None:
    if model_id in RETIRED_NVIDIA_MODEL_IDS:
        raise ProviderError(
            f"NVIDIA_MODEL {model_id!r} is retired; update NVIDIA_MODEL to a "
            "currently supported NIM model"
        )

    configured_free_ids = os.getenv(
        "NVIDIA_FREE_MODEL_IDS",
        "",
    )
    allowed_ids = {
        value.strip()
        for value in configured_free_ids.split(",")
        if value.strip()
    }
    if model_id not in allowed_ids:
        raise ProviderError(
            "NVIDIA_MODEL must be listed in NVIDIA_FREE_MODEL_IDS; "
            "only add models shown as Free Endpoint in NVIDIA Build"
        )


def _http_error(
    provider: str,
    response: httpx.Response,
    secrets: tuple[str, ...],
    model_id: str,
) -> ProviderError:
    status_code = response.status_code
    description = HTTP_ERROR_MESSAGES.get(status_code, "request failed")
    detail = _redact_secrets(response.text[:500].strip(), secrets)
    message = f"{provider} {description} (HTTP {status_code})"
    if provider == PROVIDER_NAMES["ollama"] and status_code == 404:
        message += (
            f". If Ollama reports the model is missing, download it with "
            f"`ollama pull {model_id}`"
        )
    retry_after = response.headers.get("Retry-After") if status_code == 429 else None
    if retry_after:
        message += f"; Retry-After: {retry_after}"
    if detail:
        message = f"{message}: {detail}"
    return ProviderError(message, http_status=status_code)


def _redact_secrets(value: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        value = value.replace(secret, "[REDACTED]")
    return value


def _response_json(
    response: httpx.Response,
    provider: str,
) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderError(
            f"{provider} returned invalid JSON",
            http_status=response.status_code,
        ) from exc
    if not isinstance(payload, dict):
        raise ProviderError(
            f"{provider} returned an unexpected response format",
            http_status=response.status_code,
        )
    return payload



def _response_text(model: str, payload: dict[str, Any]) -> str:
    if model == "gemini":
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            return ""

        parts = [
            part.get("text", "")
            for candidate in candidates
            if isinstance(candidate, dict)
            for part in _candidate_parts(candidate)
            if isinstance(part, dict)
        ]

        return "".join(
            part for part in parts if isinstance(part, str)
        ).strip()

    if model == "ollama":
        message = payload.get("message")

        if not isinstance(message, dict):
            return ""

        content = message.get("content")

        return content.strip() if isinstance(content, str) else ""

    # NVIDIA and Groq use OpenAI-compatible responses.
    choices = payload.get("choices")

    if isinstance(choices, list) and choices:
        choice = choices[0]

        if isinstance(choice, dict):
            message = choice.get("message")

            if isinstance(message, dict):
                content = message.get("content")

                if isinstance(content, str):
                    return content.strip()

                if isinstance(content, list):
                    return "".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict)
                        and isinstance(item.get("text"), str)
                    ).strip()

    return ""


def _response_usage(
    model: str,
    payload: dict[str, Any],
) -> tuple[int | None, int | None]:
    if model == "gemini":
        usage = payload.get("usageMetadata")
        return (
            _usage_count(usage, "promptTokenCount"),
            _usage_count(usage, "candidatesTokenCount"),
        )

    if model == "ollama":
        return (
            _usage_count(payload, "prompt_eval_count"),
            _usage_count(payload, "eval_count"),
        )

    usage = payload.get("usage")
    return (
        _usage_count(usage, "prompt_tokens"),
        _usage_count(usage, "completion_tokens"),
    )


def _usage_count(value: Any, key: str) -> int | None:
    if not isinstance(value, dict):
        return None
    count = value.get(key)
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return count
    return None



def _candidate_parts(candidate: dict[str, Any]) -> list[Any]:
    content = candidate.get("content")
    if not isinstance(content, dict):
        return []
    parts = content.get("parts")
    return parts if isinstance(parts, list) else []
