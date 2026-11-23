import json
import os
import unittest
from typing import Literal
from unittest.mock import AsyncMock, call, patch

import httpx

from backend.services.llm_clients import (
    DEFAULT_MODEL_IDS,
    MODEL_PRICING_ENV,
    NVIDIA_REQUEST_TIMEOUT_SECONDS,
    ProviderError,
    SUPPORTED_MODELS,
    estimate_cost_usd,
    generate_response,
    limit_conversation_history,
)

REAL_ASYNC_CLIENT = httpx.AsyncClient


class LLMClientTests(unittest.IsolatedAsyncioTestCase):
    async def _run_with_mock(
        self,
        model: str,
        handler: object,
        environment: dict[str, str],
        temperature: float | None = None,
        system_prompt: str | None = None,
        response_format: Literal["json_object"] | None = None,
        prompt: str = "test prompt",
        conversation_history: list[tuple[str, str | None]] | None = None,
    ):
        transport = httpx.MockTransport(handler)
        with (
            patch.dict(os.environ, environment),
            patch(
                "backend.services.llm_clients.httpx.AsyncClient",
                side_effect=lambda **kwargs: REAL_ASYNC_CLIENT(
                    transport=transport,
                    timeout=kwargs.get("timeout"),
                ),
            ),
        ):
            return await generate_response(
                model,
                prompt,
                temperature=temperature,
                system_prompt=system_prompt,
                response_format=response_format,
                conversation_history=conversation_history or (),
            )

    async def test_supported_providers_use_their_expected_request_formats(self) -> None:
        configured_models = {
            "gemini": "gemini-test",
            "nvidia": "meta/test-chat",
            "groq": "openai/gpt-oss-20b",
            "ollama": "llama-test",
        }
        environment = {
            "GEMINI_API_KEY": "gemini-secret",
            "NVIDIA_API_KEY": "nvidia-secret",
            "NVIDIA_FREE_MODEL_IDS": configured_models["nvidia"],
            "GROQ_API_KEY": "groq-secret",
            "OLLAMA_BASE_URL": "http://ollama.test/",
            **{
                f"{model.upper()}_MODEL": model_id
                for model, model_id in configured_models.items()
            },
        }
        environment["GROQ_MODEL"] = "paid-model-must-be-ignored"
        self.assertEqual(
            SUPPORTED_MODELS,
            ("gemini", "nvidia", "groq", "ollama"),
        )

        for model in SUPPORTED_MODELS:
            with self.subTest(model=model):
                requests: list[httpx.Request] = []
                model_id = configured_models[model]

                def handler(request: httpx.Request) -> httpx.Response:
                    requests.append(request)
                    payload = json.loads(request.read()) if model != "gemini" else {}
                    if model == "gemini":
                        self.assertEqual(
                            request.headers["x-goog-api-key"],
                            "gemini-secret",
                        )
                        payload = json.loads(request.read())
                        self.assertEqual(
                            payload["contents"][0]["parts"][0]["text"],
                            "test prompt",
                        )
                        body = {
                            "candidates": [
                                {"content": {"parts": [{"text": "generated"}]}}
                            ]
                        }
                    elif model == "ollama":
                        self.assertEqual(
                            str(request.url),
                            "http://ollama.test/api/chat",
                        )
                        self.assertNotIn("authorization", request.headers)
                        self.assertEqual(payload["model"], model_id)
                        if model == "nvidia":
                            self.assertEqual(payload["temperature"], 0.7)
                        self.assertEqual(
                            payload["messages"],
                            [{"role": "user", "content": "test prompt"}],
                        )
                        self.assertFalse(payload["stream"])
                        body = {
                            "message": {
                                "role": "assistant",
                                "content": "generated",
                            }
                        }
                    else:
                        expected_key = environment[f"{model.upper()}_API_KEY"]
                        self.assertEqual(
                            request.headers["authorization"],
                            f"Bearer {expected_key}",
                        )
                        if model == "groq":
                            self.assertEqual(
                                str(request.url),
                                "https://api.groq.com/openai/v1/chat/completions",
                            )
                        self.assertEqual(payload["model"], model_id)
                        self.assertEqual(
                            payload["messages"],
                            [{"role": "user", "content": "test prompt"}],
                        )
                        body = {
                            "choices": [{"message": {"content": "generated"}}]
                        }

                    return httpx.Response(200, json=body, request=request)

                result = await self._run_with_mock(model, handler, environment)

                self.assertEqual(result.response, "generated")
                self.assertEqual(result.model_id, model_id)
                self.assertEqual(result.http_status, 200)
                self.assertEqual(len(requests), 1)

    async def test_groq_handles_multiple_prompts_and_returns_token_usage(self) -> None:
        prompts: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            request_payload = json.loads(request.read())
            prompt = request_payload["messages"][-1]["content"]
            prompts.append(prompt)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": f"answer: {prompt}"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 7},
                },
                request=request,
            )

        environment = {"GROQ_API_KEY": "groq-secret"}
        first = await self._run_with_mock(
            "groq", handler, environment, prompt="first prompt"
        )
        second = await self._run_with_mock(
            "groq", handler, environment, prompt="second prompt"
        )

        self.assertEqual(prompts, ["first prompt", "second prompt"])
        self.assertEqual(first.response, "answer: first prompt")
        self.assertEqual(second.response, "answer: second prompt")
        self.assertEqual(first.prompt_tokens, 12)
        self.assertEqual(first.completion_tokens, 7)

    async def test_conversation_history_uses_each_providers_native_format(self) -> None:
        history = [("first user turn", "first provider answer"), ("second user turn", None)]
        environment = {
            "GEMINI_API_KEY": "gemini-secret",
            "GEMINI_MODEL": DEFAULT_MODEL_IDS["gemini"],
            "NVIDIA_API_KEY": "nvidia-secret",
            "NVIDIA_MODEL": DEFAULT_MODEL_IDS["nvidia"],
            "NVIDIA_FREE_MODEL_IDS": DEFAULT_MODEL_IDS["nvidia"],
            "GROQ_API_KEY": "groq-secret",
            "OLLAMA_API_KEY": "ollama-secret",
            "OLLAMA_BASE_URL": "http://ollama.test",
            "OLLAMA_MODEL": "llama-test",
        }

        for model in SUPPORTED_MODELS:
            with self.subTest(model=model):
                observed: dict[str, object] = {}

                def handler(request: httpx.Request) -> httpx.Response:
                    payload = json.loads(request.read())
                    if model == "gemini":
                        observed["messages"] = [
                            (
                                message["role"],
                                message["parts"][0]["text"],
                            )
                            for message in payload["contents"]
                        ]
                        body = {
                            "candidates": [
                                {"content": {"parts": [{"text": "generated"}]}}
                            ]
                        }
                    elif model == "ollama":
                        observed["messages"] = [
                            (message["role"], message["content"])
                            for message in payload["messages"]
                        ]
                        body = {"message": {"content": "generated"}}
                    else:
                        observed["messages"] = [
                            (message["role"], message["content"])
                            for message in payload["messages"]
                        ]
                        body = {
                            "choices": [{"message": {"content": "generated"}}]
                        }
                    return httpx.Response(200, json=body, request=request)

                await self._run_with_mock(
                    model,
                    handler,
                    environment,
                    prompt="current user turn",
                    conversation_history=history,
                )
                assistant_role = "model" if model == "gemini" else "assistant"
                self.assertEqual(
                    observed["messages"],
                    [
                        ("user", "first user turn"),
                        (assistant_role, "first provider answer"),
                        ("user", "second user turn"),
                        ("user", "current user turn"),
                    ],
                )

    def test_conversation_context_budget_keeps_recent_complete_turns(self) -> None:
        history = [
            ("older " * 7000, "older response"),
            ("recent user turn", "recent assistant response"),
        ]
        self.assertEqual(
            limit_conversation_history(history),
            (("recent user turn", "recent assistant response"),),
        )

    async def test_gemini_retries_remote_disconnect_and_succeeds(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) < 3:
                raise httpx.RemoteProtocolError(
                    "Server disconnected without sending a response.",
                    request=request,
                )
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": "generated"}]}}
                    ]
                },
                request=request,
            )

        sleep = AsyncMock()
        with patch(
            "backend.services.llm_clients.asyncio.sleep",
            new=sleep,
        ):
            result = await self._run_with_mock(
                "gemini",
                handler,
                {"GEMINI_API_KEY": "gemini-secret"},
            )

        self.assertEqual(result.response, "generated")
        self.assertEqual(result.http_status, 200)
        self.assertEqual(len(requests), 3)
        self.assertEqual(sleep.await_args_list, [call(1), call(2)])

    async def test_gemini_reports_failure_after_remote_disconnect_retries(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            raise httpx.RemoteProtocolError(
                "Server disconnected without sending a response.",
                request=request,
            )

        sleep = AsyncMock()
        with (
            patch(
                "backend.services.llm_clients.asyncio.sleep",
                new=sleep,
            ),
            self.assertRaises(ProviderError) as raised,
        ):
            await self._run_with_mock(
                "gemini",
                handler,
                {"GEMINI_API_KEY": "gemini-secret"},
            )

        self.assertIn("Server disconnected", str(raised.exception))
        self.assertEqual(len(requests), 3)
        self.assertEqual(sleep.await_args_list, [call(1), call(2)])

    async def test_nvidia_temperature_override_is_respected(self) -> None:
        environment = {
            "NVIDIA_API_KEY": "nvidia-secret",
            "NVIDIA_MODEL": DEFAULT_MODEL_IDS["nvidia"],
            "NVIDIA_FREE_MODEL_IDS": DEFAULT_MODEL_IDS["nvidia"],
        }
        observed: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            observed.update(json.loads(request.read()))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "generated"}}]},
                request=request,
            )

        await self._run_with_mock(
            "nvidia",
            handler,
            environment,
            temperature=0,
            system_prompt="Evaluate neutrally.",
        )

        self.assertEqual(observed["temperature"], 0)
        self.assertEqual(
            observed["messages"],
            [
                {"role": "system", "content": "Evaluate neutrally."},
                {"role": "user", "content": "test prompt"},
            ],
        )

    async def test_nvidia_json_response_format_is_forwarded(self) -> None:
        environment = {
            "NVIDIA_API_KEY": "nvidia-secret",
            "NVIDIA_MODEL": DEFAULT_MODEL_IDS["nvidia"],
            "NVIDIA_FREE_MODEL_IDS": DEFAULT_MODEL_IDS["nvidia"],
        }
        observed: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            observed.update(json.loads(request.read()))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "{}"}}]},
                request=request,
            )

        await self._run_with_mock(
            "nvidia",
            handler,
            environment,
            temperature=0,
            system_prompt="Return JSON.",
            response_format="json_object",
        )

        self.assertEqual(
            observed["response_format"],
            {"type": "json_object"},
        )

    async def test_provider_usage_metadata_is_returned_for_each_native_format(self) -> None:
        environment = {
            "GEMINI_API_KEY": "gemini-secret",
            "NVIDIA_API_KEY": "nvidia-secret",
            "NVIDIA_MODEL": DEFAULT_MODEL_IDS["nvidia"],
            "NVIDIA_FREE_MODEL_IDS": DEFAULT_MODEL_IDS["nvidia"],
            "GROQ_API_KEY": "groq-secret",
            "OLLAMA_API_KEY": "ollama-secret",
            "OLLAMA_MODEL": "llama-test",
        }

        for model in SUPPORTED_MODELS:
            with self.subTest(model=model):
                def handler(request: httpx.Request) -> httpx.Response:
                    if model == "gemini":
                        body = {
                            "candidates": [
                                {"content": {"parts": [{"text": "generated"}]}}
                            ],
                            "usageMetadata": {
                                "promptTokenCount": 67,
                                "candidatesTokenCount": 711,
                            },
                        }
                    elif model == "ollama":
                        body = {
                            "message": {"content": "generated"},
                            "prompt_eval_count": 67,
                            "eval_count": 711,
                        }
                    else:
                        body = {
                            "choices": [{"message": {"content": "generated"}}],
                            "usage": {
                                "prompt_tokens": 67,
                                "completion_tokens": 711,
                            },
                        }
                    return httpx.Response(200, json=body, request=request)

                result = await self._run_with_mock(model, handler, environment)
                self.assertEqual(result.prompt_tokens, 67)
                self.assertEqual(result.completion_tokens, 711)

    def test_cost_requires_actual_usage_and_configured_exact_model_rates(self) -> None:
        pricing = json.dumps({
            "priced-model": {"input": 0.5, "output": 1.0},
            "free-model": {"input": 0, "output": 0},
        })
        with patch.dict(os.environ, {MODEL_PRICING_ENV: pricing}):
            self.assertAlmostEqual(
                estimate_cost_usd("priced-model", 67, 711),
                0.0007445,
            )
            self.assertEqual(estimate_cost_usd("free-model", 67, 711), 0)
            self.assertIsNone(estimate_cost_usd("unknown-model", 67, 711))
            self.assertIsNone(estimate_cost_usd("priced-model", None, 711))

        with patch.dict(os.environ, {MODEL_PRICING_ENV: ""}):
            self.assertIsNone(estimate_cost_usd("priced-model", 67, 711))

    async def test_invalid_provider_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported model"):
            await generate_response("unsupported", "question")

    async def test_groq_authentication_failure_redacts_api_key(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                text="invalid API key: groq-secret",
                request=request,
            )

        with self.assertRaises(ProviderError) as raised:
            await self._run_with_mock(
                "groq",
                handler,
                {
                    "GROQ_API_KEY": "groq-secret",
                },
            )

        self.assertEqual(raised.exception.http_status, 401)
        self.assertIn("HTTP 401", str(raised.exception))
        self.assertNotIn("groq-secret", str(raised.exception))

    async def test_groq_timeout_and_network_errors_are_reported(self) -> None:
        failures = (
            (httpx.ReadTimeout("request timed out"), "Groq request timed out"),
            (httpx.ConnectError("network unavailable"), "Groq connection failed"),
        )
        for error, expected_message in failures:
            with self.subTest(error=type(error).__name__):

                def handler(request: httpx.Request) -> httpx.Response:
                    raise error

                with self.assertRaises(ProviderError) as raised:
                    await self._run_with_mock(
                        "groq",
                        handler,
                        {"GROQ_API_KEY": "groq-secret"},
                    )

                self.assertIn(expected_message, str(raised.exception))

    async def test_groq_rejects_empty_response(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": ""}}]},
                request=request,
            )

        with self.assertRaisesRegex(ProviderError, "returned an empty response"):
            await self._run_with_mock(
                "groq",
                handler,
                {"GROQ_API_KEY": "groq-secret"},
            )

    async def test_ollama_missing_model_includes_pull_command(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={"error": "model 'llama-test' not found"},
                request=request,
            )

        with self.assertRaises(ProviderError) as raised:
            await self._run_with_mock(
                "ollama",
                handler,
                {
                    "OLLAMA_BASE_URL": "http://ollama.test",
                    "OLLAMA_MODEL": "llama-test",
                },
            )

        self.assertEqual(raised.exception.http_status, 404)
        self.assertIn("ollama pull llama-test", str(raised.exception))

    async def test_ollama_connection_failure_is_actionable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with self.assertRaises(ProviderError) as raised:
            await self._run_with_mock(
                "ollama",
                handler,
                {"OLLAMA_BASE_URL": "http://ollama.test"},
            )

        self.assertIsNone(raised.exception.http_status)
        self.assertIn("Ollama is unavailable", str(raised.exception))
        self.assertIn("ollama serve", str(raised.exception))

    async def test_nvidia_model_must_be_in_the_free_endpoint_allowlist(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "NVIDIA_API_KEY": "nvidia-secret",
                    "NVIDIA_MODEL": "meta/paid-model",
                    "NVIDIA_FREE_MODEL_IDS": DEFAULT_MODEL_IDS["nvidia"],
                },
            ),
            patch("backend.services.llm_clients.httpx.AsyncClient") as client,
        ):
            with self.assertRaisesRegex(ProviderError, "Free Endpoint"):
                await generate_response("nvidia", "question")
        client.assert_not_called()

    async def test_retired_nvidia_model_is_rejected_before_a_request(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "NVIDIA_API_KEY": "nvidia-secret",
                    "NVIDIA_MODEL": "meta/llama-3.2-3b-instruct",
                    "NVIDIA_FREE_MODEL_IDS": "meta/llama-3.2-3b-instruct",
                },
            ),
            patch("backend.services.llm_clients.httpx.AsyncClient") as client,
        ):
            with self.assertRaisesRegex(ProviderError, "is retired"):
                await generate_response("nvidia", "question")
        client.assert_not_called()

    async def test_nvidia_uses_extended_timeout_for_model_startup(self) -> None:
        model_id = DEFAULT_MODEL_IDS["nvidia"]
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": "generated"}}]},
                request=request,
            )
        )
        captured: dict[str, float] = {}

        def make_client(**kwargs):
            captured["timeout"] = kwargs["timeout"]
            return REAL_ASYNC_CLIENT(
                transport=transport,
                timeout=kwargs["timeout"],
            )

        with (
            patch.dict(
                os.environ,
                {
                    "NVIDIA_API_KEY": "nvidia-secret",
                    "NVIDIA_MODEL": model_id,
                    "NVIDIA_FREE_MODEL_IDS": model_id,
                },
            ),
            patch(
                "backend.services.llm_clients.httpx.AsyncClient",
                side_effect=make_client,
            ),
        ):
            result = await generate_response("nvidia", "question")

        self.assertEqual(result.response, "generated")
        self.assertEqual(captured["timeout"], NVIDIA_REQUEST_TIMEOUT_SECONDS)

    async def test_http_errors_preserve_status_and_redact_credentials(self) -> None:
        for status_code in (401, 403, 404, 422, 429, 500):
            with self.subTest(status_code=status_code):

                def handler(request: httpx.Request) -> httpx.Response:
                    return httpx.Response(
                        status_code,
                        text="provider detail; groq-secret",
                        headers={"Retry-After": "30"} if status_code == 429 else None,
                        request=request,
                    )

                with self.assertRaises(ProviderError) as raised:
                    await self._run_with_mock(
                        "groq",
                        handler,
                        {
                            "GROQ_API_KEY": "groq-secret",
                        },
                    )

                self.assertEqual(raised.exception.http_status, status_code)
                self.assertIn(f"HTTP {status_code}", str(raised.exception))
                self.assertNotIn("groq-secret", str(raised.exception))
                if status_code == 429:
                    self.assertIn("Retry-After: 30", str(raised.exception))

    async def test_missing_provider_credentials_fail_before_a_request(self) -> None:
        empty_credentials = {
            "GEMINI_API_KEY": "",
            "NVIDIA_API_KEY": "",
            "GROQ_API_KEY": "",
        }
        expected_missing = {
            "gemini": ("GEMINI_API_KEY",),
            "nvidia": ("NVIDIA_API_KEY",),
            "groq": ("GROQ_API_KEY",),
        }
        for model, missing_names in expected_missing.items():
            with self.subTest(model=model):
                with (
                    patch.dict(os.environ, empty_credentials),
                    patch("backend.services.llm_clients.httpx.AsyncClient") as client,
                ):
                    with self.assertRaises(ProviderError) as raised:
                        await generate_response(model, "question")
                for env_name in missing_names:
                    self.assertIn(env_name, str(raised.exception))
                client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
