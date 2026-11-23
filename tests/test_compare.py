import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from backend.main import app
from backend.services.evaluator import (
    EvaluationError,
    EvaluationResults,
    _JudgeScore,
    _MetricJudgment,
)
from backend.services.llm_clients import (
    ProviderError,
    ProviderResponse,
    SUPPORTED_MODELS,
    get_provider_config,
)


def _judge_score(response_id: str, score: int) -> _JudgeScore:
    def metric() -> _MetricJudgment:
        return _MetricJudgment(
            score=score,
            reason="Mock judge assessment.",
            evidence="Mock response evidence.",
            missing_requirements=[],
        )

    return _JudgeScore(
        response_id=response_id,
        response_quality=metric(),
        reasoning_clarity=metric(),
        factual_support=metric(),
        instruction_following=metric(),
        formatting=metric(),
        completeness=metric(),
        rationale="Mock evidence-based evaluation.",
    )


def _evaluation_results(responses: list[object]) -> EvaluationResults:
    return EvaluationResults(
        scores={
            response.model: _judge_score(response.model, 4)
            for response in responses
        },
        errors={},
    )


class CompareRouteTests(unittest.IsolatedAsyncioTestCase):
    async def _request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            return await client.request(method, path, **kwargs)

    async def test_compare_evaluates_all_models_and_returns_winner(self) -> None:
        calls: list[tuple[str, str]] = []
        requests_started = 0
        all_requests_started = asyncio.Event()

        async def generate(model: str, prompt: str) -> ProviderResponse:
            nonlocal requests_started
            calls.append((model, prompt))
            requests_started += 1
            if requests_started == len(SUPPORTED_MODELS):
                all_requests_started.set()
            await asyncio.wait_for(all_requests_started.wait(), timeout=1)
            config = get_provider_config(model)
            return ProviderResponse(
                model=model,
                provider=config.provider,
                model_id=config.model_id,
                response=f"response-{model}",
                http_status=200,
            )

        async def evaluate(
            prompt: str,
            reference_answer: str | None,
            responses: list[object],
        ) -> EvaluationResults:
            self.assertEqual(prompt, "same prompt")
            self.assertEqual(reference_answer, "ground truth")
            return EvaluationResults(
                scores={
                    response.model: _judge_score(
                        response.model,
                        5 if response.model == "gemini" else 3,
                    )
                    for response in responses
                },
                errors={},
            )

        with (
            patch("backend.routes.compare.generate_response", new=generate),
            patch("backend.routes.compare.evaluate_responses", new=evaluate),
        ):
            result = await self._request(
                "POST",
                "/compare",
                json={"prompt": "same prompt", "reference_answer": "ground truth"},
            )

        self.assertEqual(result.status_code, 200)
        body = result.json()
        self.assertEqual(len(body["responses"]), 4)
        expected_response_fields = {
            "model",
            "provider",
            "model_id",
            "response",
            "http_status",
            "response_time_ms",
            "error",
            "prompt_tokens",
            "completion_tokens",
            "estimated_cost_usd",
            "simple_checks",
            "scores",
            "score_evidence",
            "score_source",
            "quality_score",
            "rank",
        }
        for item in body["responses"]:
            self.assertTrue(expected_response_fields.issubset(item))
            self.assertIsNotNone(item["scores"])
            self.assertTrue(item["score_evidence"]["response_quality"]["evidence"])
            self.assertEqual(item["score_source"], "ai_rubric")
            self.assertIsNotNone(item["quality_score"])
            self.assertIsNotNone(item["rank"])
        self.assertEqual({prompt for _, prompt in calls}, {"same prompt"})
        self.assertEqual({model for model, _ in calls}, set(SUPPORTED_MODELS))
        self.assertEqual(body["evaluation_status"], "success")
        self.assertIn("NVIDIA NIM", body["evaluation_model"])
        self.assertIn("Google Gemini", body["evaluation_model"])
        self.assertEqual(body["winner_model"], "gemini")
        self.assertEqual(body["winner_models"], ["gemini"])
        self.assertEqual(body["best_response"]["model"], "gemini")
        self.assertEqual(body["best_response"]["quality_score"], 100)
        gemini = next(item for item in body["responses"] if item["model"] == "gemini")
        self.assertEqual(gemini["simple_checks"]["word_count"], 2)
        self.assertFalse(gemini["simple_checks"]["ending_check"])
        self.assertEqual(
            {item["http_status"] for item in body["responses"]},
            {200},
        )

    async def test_compare_preserves_each_providers_full_formatted_response(self) -> None:
        formatted_response = (
            "## Background\n\n"
            + "A detailed paragraph with preserved line breaks. " * 20
            + "\n\n- First point\n- Second point\n\n"
            + "1. First step\n2. Second step\n\n"
            + "**Important detail**\n\n"
            + "```python\nprint('complete')\n```\n"
        )
        responses = {
            model: f"# {model}\n\n{formatted_response}"
            for model in SUPPORTED_MODELS
        }

        async def generate(model: str, prompt: str) -> ProviderResponse:
            config = get_provider_config(model)
            return ProviderResponse(
                model=model,
                provider=config.provider,
                model_id=config.model_id,
                response=responses[model],
                http_status=200,
            )

        async def evaluate(
            prompt: str,
            reference_answer: str | None,
            results: list[object],
        ) -> EvaluationResults:
            return _evaluation_results(results)

        with (
            patch("backend.routes.compare.generate_response", new=generate),
            patch("backend.routes.compare.evaluate_responses", new=evaluate),
        ):
            result = await self._request("POST", "/compare", json={"prompt": "long"})

        self.assertEqual(result.status_code, 200)
        body = result.json()
        self.assertEqual(
            {item["model"]: item["response"] for item in body["responses"]},
            responses,
        )

    async def test_provider_failure_does_not_stop_other_models(self) -> None:
        async def generate(model: str, prompt: str) -> ProviderResponse:
            if model == "ollama":
                raise ProviderError("provider unavailable", http_status=503)
            config = get_provider_config(model)
            return ProviderResponse(
                model=model,
                provider=config.provider,
                model_id=config.model_id,
                response=f"response-{model}",
                http_status=200,
            )

        async def evaluate(
            prompt: str,
            reference_answer: str | None,
            responses: list[object],
        ) -> EvaluationResults:
            return _evaluation_results(responses)

        with (
            patch("backend.routes.compare.generate_response", new=generate),
            patch("backend.routes.compare.evaluate_responses", new=evaluate),
        ):
            result = await self._request(
                "POST",
                "/compare",
                json={"prompt": "keep going"},
            )

        body = result.json()
        self.assertEqual(body["status"], "partial_failure")
        self.assertEqual(body["evaluation_status"], "success")
        ollama = next(
            item for item in body["responses"] if item["model"] == "ollama"
        )
        self.assertEqual(ollama["http_status"], 503)
        self.assertIsNone(ollama["response"])
        self.assertEqual(ollama["error"], "provider unavailable")
        self.assertIsNone(ollama["quality_score"])
        self.assertEqual(len(body["responses"]), 4)

    async def test_compare_runs_only_selected_providers(self) -> None:
        calls: list[str] = []

        async def generate(model: str, prompt: str) -> ProviderResponse:
            calls.append(model)
            config = get_provider_config(model)
            return ProviderResponse(
                model=model,
                provider=config.provider,
                model_id=config.model_id,
                response=f"response-{model}",
                http_status=200,
            )

        async def evaluate(
            prompt: str,
            reference_answer: str | None,
            responses: list[object],
        ) -> EvaluationResults:
            return _evaluation_results(responses)

        with (
            patch("backend.routes.compare.generate_response", new=generate),
            patch("backend.routes.compare.evaluate_responses", new=evaluate),
        ):
            result = await self._request(
                "POST",
                "/compare",
                json={
                    "prompt": "Translate hello",
                    "providers": ["nvidia", "groq"],
                },
            )

        self.assertEqual(result.status_code, 200)
        self.assertEqual(set(calls), {"nvidia", "groq"})
        self.assertEqual(
            {item["model"] for item in result.json()["responses"]},
            {"nvidia", "groq"},
        )

    async def test_compare_returns_native_usage_and_priced_cost(self) -> None:
        config = get_provider_config("groq")

        async def generate(model: str, prompt: str) -> ProviderResponse:
            return ProviderResponse(
                model=model,
                provider=config.provider,
                model_id=config.model_id,
                response="A measured response.",
                http_status=200,
                prompt_tokens=67,
                completion_tokens=711,
            )

        async def evaluate(
            prompt: str,
            reference_answer: str | None,
            responses: list[object],
        ) -> EvaluationResults:
            return _evaluation_results(responses)

        pricing = json.dumps({config.model_id: {"input": 0.5, "output": 1.0}})
        with (
            patch.dict(os.environ, {"MODEL_PRICING_USD_PER_1M_TOKENS": pricing}),
            patch("backend.routes.compare.generate_response", new=generate),
            patch("backend.routes.compare.evaluate_responses", new=evaluate),
        ):
            result = await self._request(
                "POST",
                "/compare",
                json={"prompt": "same prompt", "providers": ["groq"]},
            )

        self.assertEqual(result.status_code, 200)
        item = result.json()["responses"][0]
        self.assertEqual(item["prompt_tokens"], 67)
        self.assertEqual(item["completion_tokens"], 711)
        self.assertAlmostEqual(item["estimated_cost_usd"], 0.0007445)
        self.assertEqual(item["simple_checks"]["word_count"], 3)
        self.assertTrue(item["simple_checks"]["ending_check"])
        self.assertGreaterEqual(item["response_time_ms"], 0)

    async def test_evaluation_failure_keeps_provider_output_and_checks(self) -> None:
        config = get_provider_config("gemini")

        async def generate(model: str, prompt: str) -> ProviderResponse:
            return ProviderResponse(
                model=model,
                provider=config.provider,
                model_id=config.model_id,
                response="Provider response.",
                http_status=200,
                prompt_tokens=12,
                completion_tokens=8,
            )

        async def fail_evaluation(
            prompt: str,
            reference_answer: str | None,
            responses: list[object],
        ) -> EvaluationResults:
            raise EvaluationError("judge unavailable")

        with (
            patch("backend.routes.compare.generate_response", new=generate),
            patch("backend.routes.compare.evaluate_responses", new=fail_evaluation),
        ):
            result = await self._request(
                "POST",
                "/compare",
                json={"prompt": "same prompt", "providers": ["gemini"]},
            )

        self.assertEqual(result.status_code, 200)
        body = result.json()
        item = body["responses"][0]
        self.assertEqual(body["evaluation_status"], "failed")
        self.assertEqual(body["evaluation_error"], "judge unavailable")
        self.assertEqual(item["response"], "Provider response.")
        self.assertEqual(item["prompt_tokens"], 12)
        self.assertEqual(item["completion_tokens"], 8)
        self.assertEqual(item["simple_checks"]["word_count"], 2)
        self.assertTrue(item["simple_checks"]["ending_check"])
        self.assertIsNone(item["scores"])
        self.assertIsNone(item["quality_score"])
        self.assertEqual(body["winner_models"], [])

    async def test_partial_judge_failure_preserves_other_provider_scores(self) -> None:
        async def generate(model: str, prompt: str) -> ProviderResponse:
            config = get_provider_config(model)
            return ProviderResponse(
                model=model,
                provider=config.provider,
                model_id=config.model_id,
                response=f"response-{model}",
                http_status=200,
            )

        async def evaluate(
            prompt: str,
            reference_answer: str | None,
            responses: list[object],
        ) -> EvaluationResults:
            return EvaluationResults(
                scores={"groq": _judge_score("groq", 4)},
                errors={"gemini": "NVIDIA evaluation unavailable"},
            )

        with (
            patch("backend.routes.compare.generate_response", new=generate),
            patch("backend.routes.compare.evaluate_responses", new=evaluate),
        ):
            result = await self._request(
                "POST",
                "/compare",
                json={
                    "prompt": "score independently",
                    "providers": ["gemini", "groq"],
                },
            )

        self.assertEqual(result.status_code, 200)
        body = result.json()
        self.assertEqual(body["evaluation_status"], "partial_failure")
        self.assertIn("NVIDIA evaluation unavailable", body["evaluation_error"])
        gemini = next(item for item in body["responses"] if item["model"] == "gemini")
        groq = next(
            item for item in body["responses"] if item["model"] == "groq"
        )
        self.assertIsNone(gemini["scores"])
        self.assertIsNone(gemini["quality_score"])
        self.assertEqual(
            gemini["score_error"],
            "NVIDIA evaluation unavailable",
        )
        self.assertIsNotNone(groq["scores"])
        self.assertIsNotNone(groq["quality_score"])

    async def test_all_provider_failures_skip_evaluation_and_have_no_winner(self) -> None:
        async def generate(model: str, prompt: str) -> ProviderResponse:
            raise ProviderError("offline")

        evaluator = AsyncMock()
        with (
            patch("backend.routes.compare.generate_response", new=generate),
            patch("backend.routes.compare.evaluate_responses", new=evaluator),
        ):
            result = await self._request(
                "POST",
                "/compare",
                json={"prompt": "fail safely"},
            )

        body = result.json()
        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["evaluation_status"], "not_run")
        self.assertEqual(len(body["responses"]), 4)
        self.assertIsNone(body["winner_model"])
        self.assertEqual(body["winner_models"], [])
        self.assertIsNone(body["best_response"])
        self.assertEqual(body["best_responses"], [])
        evaluator.assert_not_awaited()

    async def test_blank_prompt_and_reference_are_rejected(self) -> None:
        missing_prompt = await self._request(
            "POST",
            "/compare",
            json={},
        )
        blank_prompt = await self._request(
            "POST",
            "/compare",
            json={"prompt": "  "},
        )
        blank_reference = await self._request(
            "POST",
            "/compare",
            json={"prompt": "question", "reference_answer": "  "},
        )
        invalid_prompt = await self._request(
            "POST",
            "/compare",
            json={"prompt": 123},
        )
        invalid_body = await self._request(
            "POST",
            "/compare",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(missing_prompt.status_code, 422)
        self.assertEqual(blank_prompt.status_code, 422)
        self.assertEqual(blank_reference.status_code, 422)
        self.assertEqual(invalid_prompt.status_code, 422)
        self.assertEqual(invalid_body.status_code, 422)

    async def test_invalid_provider_selection_is_rejected(self) -> None:
        for providers in ([], ["unknown"], ["gemini", "gemini"]):
            with self.subTest(providers=providers):
                result = await self._request(
                    "POST",
                    "/compare",
                    json={"prompt": "question", "providers": providers},
                )
                self.assertEqual(result.status_code, 422)

    async def test_vite_origin_is_allowed_by_cors(self) -> None:
        result = await self._request(
            "OPTIONS",
            "/compare",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(
            result.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )

    async def test_ask_rejects_an_unsupported_model(self) -> None:
        result = await self._request(
            "POST",
            "/ask",
            json={"model": "unsupported", "prompt": "question"},
        )
        self.assertEqual(result.status_code, 400)
        self.assertIn("Unsupported model", result.json()["detail"])

    async def test_existing_models_and_ask_endpoints_remain_available(self) -> None:
        config = get_provider_config("gemini")
        generated = ProviderResponse(
            model="gemini",
            provider=config.provider,
            model_id=config.model_id,
            response="real provider result",
            http_status=200,
        )
        with patch(
            "backend.main.generate_response",
            new=AsyncMock(return_value=generated),
        ):
            models = await self._request("GET", "/models")
            answer = await self._request(
                "POST",
                "/ask",
                json={"model": "gemini", "prompt": "question"},
            )

        self.assertEqual(models.status_code, 200)
        self.assertEqual(
            models.json()["models"],
            ["gemini", "nvidia", "groq", "ollama"],
        )
        self.assertEqual(models.json()["total"], 4)
        self.assertEqual(answer.status_code, 200)
        self.assertEqual(answer.json()["response"], "real provider result")


if __name__ == "__main__":
    unittest.main()
