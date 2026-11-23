import os
import tempfile
import unittest
from unittest.mock import patch

import httpx

from backend.main import app
from backend.services.evaluator import (
    EvaluationResults,
    _JudgeScore,
    _MetricJudgment,
)
from backend.services.llm_clients import (
    ProviderError,
    ProviderResponse,
    get_provider_config,
)


def _evaluations_for(responses: list[object]) -> EvaluationResults:
    def metric() -> _MetricJudgment:
        return _MetricJudgment(
            score=4,
            reason="Mock evaluation.",
            evidence="Mock response evidence.",
            missing_requirements=[],
        )

    return EvaluationResults(
        scores={
            response.model: _JudgeScore(
                response_id="R1",
                response_quality=metric(),
                reasoning_clarity=metric(),
                factual_support=metric(),
                instruction_following=metric(),
                formatting=metric(),
                completeness=metric(),
                rationale="Mock evidence-based evaluation.",
            )
            for response in responses
        },
        errors={},
    )


class ConversationRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_environment = patch.dict(
            os.environ,
            {
                "CONVERSATION_DB_PATH": os.path.join(
                    self.temporary_directory.name,
                    "conversations.sqlite3",
                )
            },
        )
        self.database_environment.start()

    async def asyncTearDown(self) -> None:
        self.database_environment.stop()
        self.temporary_directory.cleanup()

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: object,
    ) -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            return await client.request(method, path, **kwargs)

    async def _create_conversation(self) -> str:
        response = await self._request("POST", "/conversations", json={})
        self.assertEqual(response.status_code, 201)
        return response.json()["id"]

    async def test_create_list_rename_and_delete_conversations(self) -> None:
        first_id = await self._create_conversation()
        second_id = await self._create_conversation()
        self.assertNotEqual(first_id, second_id)

        listing = await self._request("GET", "/conversations")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(
            {item["id"] for item in listing.json()},
            {first_id, second_id},
        )

        renamed = await self._request(
            "PATCH",
            f"/conversations/{first_id}",
            json={"title": "Project research"},
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["title"], "Project research")

        deleted = await self._request("DELETE", f"/conversations/{first_id}")
        self.assertEqual(deleted.status_code, 204)
        missing = await self._request("GET", f"/conversations/{first_id}")
        self.assertEqual(missing.status_code, 404)
        remaining = await self._request("GET", f"/conversations/{second_id}")
        self.assertEqual(remaining.status_code, 200)

    async def test_turns_reuse_comparison_pipeline_and_preserve_scores(self) -> None:
        conversation_id = await self._create_conversation()
        generation_calls: list[tuple[str, str, tuple[tuple[str, str | None], ...]]] = []
        evaluation_calls: list[
            tuple[str, str | None, dict[str, tuple[tuple[str, str | None], ...]] | None]
        ] = []

        async def generate(
            model: str,
            prompt: str,
            *,
            conversation_history: tuple[tuple[str, str | None], ...] = (),
        ) -> ProviderResponse:
            generation_calls.append((model, prompt, conversation_history))
            config = get_provider_config(model)
            return ProviderResponse(
                model=model,
                provider=config.provider,
                model_id=config.model_id,
                response=f"{model} response to {prompt}",
                http_status=200,
                prompt_tokens=14,
                completion_tokens=9,
            )

        async def evaluate(
            prompt: str,
            reference_answer: str | None,
            responses: list[object],
            *,
            conversation_contexts: dict[
                str, tuple[tuple[str, str | None], ...]
            ] | None = None,
        ) -> EvaluationResults:
            evaluation_calls.append((prompt, reference_answer, conversation_contexts))
            return _evaluations_for(responses)

        with (
            patch("backend.routes.compare.generate_response", new=generate),
            patch("backend.routes.compare.evaluate_responses", new=evaluate),
        ):
            first = await self._request(
                "POST",
                f"/conversations/{conversation_id}/turns",
                json={
                    "prompt": "Explain context windows",
                    "reference_answer": "They limit model input.",
                    "providers": ["groq"],
                },
            )
            second = await self._request(
                "POST",
                f"/conversations/{conversation_id}/turns",
                json={
                    "prompt": "How does that apply here?",
                    "providers": ["groq"],
                },
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(evaluation_calls), 2)
        self.assertEqual(evaluation_calls[0][0], "Explain context windows")
        self.assertEqual(evaluation_calls[1][0], "How does that apply here?")
        self.assertEqual(
            generation_calls[1][2],
            (("Explain context windows", "groq response to Explain context windows"),),
        )
        self.assertEqual(
            evaluation_calls[1][2]["groq"],
            (("Explain context windows", "groq response to Explain context windows"),),
        )

        detail = await self._request("GET", f"/conversations/{conversation_id}")
        body = detail.json()
        self.assertEqual(
            [turn["prompt"] for turn in body["turns"]],
            ["Explain context windows", "How does that apply here?"],
        )
        self.assertEqual(body["title"], "Explain context windows")
        for turn in body["turns"]:
            result = turn["result"]
            self.assertEqual(result["responses"][0]["prompt_tokens"], 14)
            self.assertIsNotNone(result["responses"][0]["scores"])
            self.assertIsNotNone(result["responses"][0]["rank"])
            self.assertIsNotNone(result["responses"][0]["quality_score"])
            self.assertEqual(result["winner_models"], ["groq"])

    async def test_context_isolated_between_conversations(self) -> None:
        first_id = await self._create_conversation()
        second_id = await self._create_conversation()
        contexts_by_prompt: dict[str, tuple[tuple[str, str | None], ...]] = {}

        async def generate(
            model: str,
            prompt: str,
            *,
            conversation_history: tuple[tuple[str, str | None], ...] = (),
        ) -> ProviderResponse:
            contexts_by_prompt[prompt] = conversation_history
            config = get_provider_config(model)
            return ProviderResponse(
                model=model,
                provider=config.provider,
                model_id=config.model_id,
                response=f"{model} says {prompt}",
                http_status=200,
            )

        async def evaluate(
            prompt: str,
            reference_answer: str | None,
            responses: list[object],
            *,
            conversation_contexts: dict[
                str, tuple[tuple[str, str | None], ...]
            ] | None = None,
        ) -> EvaluationResults:
            return _evaluations_for(responses)

        with (
            patch("backend.routes.compare.generate_response", new=generate),
            patch("backend.routes.compare.evaluate_responses", new=evaluate),
        ):
            for conversation_id, prompt in (
                (first_id, "first conversation secret"),
                (second_id, "second conversation secret"),
                (first_id, "follow up in first"),
            ):
                response = await self._request(
                    "POST",
                    f"/conversations/{conversation_id}/turns",
                    json={"prompt": prompt, "providers": ["groq"]},
                )
                self.assertEqual(response.status_code, 200)

        first_context = contexts_by_prompt["follow up in first"]
        self.assertEqual(
            first_context,
            (("first conversation secret", "groq says first conversation secret"),),
        )
        self.assertNotIn("second conversation secret", str(first_context))

    async def test_provider_failure_is_persisted_without_stopping_other_models(self) -> None:
        conversation_id = await self._create_conversation()

        async def generate(
            model: str,
            prompt: str,
            *,
            conversation_history: tuple[tuple[str, str | None], ...] = (),
        ) -> ProviderResponse:
            if model == "groq":
                raise ProviderError("Groq temporarily unavailable", http_status=503)
            config = get_provider_config(model)
            return ProviderResponse(
                model=model,
                provider=config.provider,
                model_id=config.model_id,
                response="Available provider response",
                http_status=200,
            )

        async def evaluate(
            prompt: str,
            reference_answer: str | None,
            responses: list[object],
            *,
            conversation_contexts: dict[
                str, tuple[tuple[str, str | None], ...]
            ] | None = None,
        ) -> EvaluationResults:
            return _evaluations_for(responses)

        with (
            patch("backend.routes.compare.generate_response", new=generate),
            patch("backend.routes.compare.evaluate_responses", new=evaluate),
        ):
            submitted = await self._request(
                "POST",
                f"/conversations/{conversation_id}/turns",
                json={"prompt": "continue despite one failure", "providers": ["groq", "ollama"]},
            )

        self.assertEqual(submitted.status_code, 200)
        result = submitted.json()["result"]
        self.assertEqual(result["status"], "partial_failure")
        failed = next(item for item in result["responses"] if item["model"] == "groq")
        succeeded = next(item for item in result["responses"] if item["model"] == "ollama")
        self.assertEqual(failed["error"], "Groq temporarily unavailable")
        self.assertIsNone(failed["rank"])
        self.assertIsNotNone(succeeded["rank"])

        reopened = await self._request("GET", f"/conversations/{conversation_id}")
        self.assertEqual(len(reopened.json()["turns"]), 1)
        self.assertEqual(
            reopened.json()["turns"][0]["result"]["status"],
            "partial_failure",
        )

    async def test_invalid_turn_and_missing_conversation_are_rejected(self) -> None:
        conversation_id = await self._create_conversation()
        blank_prompt = await self._request(
            "POST",
            f"/conversations/{conversation_id}/turns",
            json={"prompt": "  "},
        )
        missing = await self._request(
            "POST",
            "/conversations/00000000-0000-0000-0000-000000000000/turns",
            json={"prompt": "question"},
        )
        self.assertEqual(blank_prompt.status_code, 422)
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
