import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from pydantic import ValidationError

from backend.models.schemas import ModelRunResult
from backend.services.evaluator import (
    _MetricJudgment,
    _JudgeScore,
    calculate_simple_checks,
    evaluate_responses,
    score_and_rank,
)
from backend.services.llm_clients import ProviderError, SUPPORTED_MODELS


def _result(model: str, score_text: str, elapsed_ms: float = 1) -> ModelRunResult:
    return ModelRunResult(
        model=model,
        provider=f"{model} provider",
        model_id=f"{model}-model-id",
        response=score_text,
        http_status=200,
        response_time_ms=elapsed_ms,
        error=None,
    )


def _judge_score(
    response_id: str,
    scores: dict[str, int | None] | None = None,
) -> _JudgeScore:
    metric_scores = {
        "response_quality": 5,
        "reasoning_clarity": 5,
        "factual_support": 5,
        "instruction_following": 5,
        "formatting": 5,
        "completeness": 5,
    }
    if scores is not None:
        metric_scores.update(scores)

    def metric(name: str) -> _MetricJudgment:
        return _MetricJudgment(
            score=metric_scores[name],
            reason=f"{name} assessment",
            evidence=f"Evidence for {name}.",
            missing_requirements=[],
        )

    return _JudgeScore(
        response_id=response_id,
        response_quality=metric("response_quality"),
        reasoning_clarity=metric("reasoning_clarity"),
        factual_support=metric("factual_support"),
        instruction_following=metric("instruction_following"),
        formatting=metric("formatting"),
        completeness=metric("completeness"),
        rationale="rubric score",
    )


def _metric_payload(
    content: str,
    name: str,
    score: int | None,
    missing_requirements: list[str] | None = None,
) -> dict[str, object]:
    return {
        "score": score,
        "reason": f"{name} assessment based on the response.",
        "evidence": content[:120] or "No response content was provided.",
        "missing_requirements": missing_requirements or [],
    }


def _judge_payload(
    response_id: str,
    content: str,
    scores: dict[str, int | None],
    missing_requirements: list[str] | None = None,
) -> dict[str, object]:
    return {
        "response_id": response_id,
        "response_quality": _metric_payload(
            content, "Response quality", scores["response_quality"], missing_requirements
        ),
        "reasoning_clarity": _metric_payload(
            content, "Reasoning clarity", scores["reasoning_clarity"], missing_requirements
        ),
        "factual_support": _metric_payload(
            content, "Factual support", scores["factual_support"], missing_requirements
        ),
        "instruction_following": _metric_payload(
            content,
            "Instruction following",
            scores["instruction_following"],
            missing_requirements,
        ),
        "formatting": _metric_payload(
            content, "Formatting", scores["formatting"], missing_requirements
        ),
        "completeness": _metric_payload(
            content, "Completeness", scores["completeness"], missing_requirements
        ),
        "rationale": "Each metric is scored from the response evidence.",
    }


class EvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_conversation_context_informs_but_does_not_replace_current_prompt(self) -> None:
        observed_input: dict[str, object] = {}

        async def judge(evaluation_input: dict[str, object]) -> str:
            observed_input.update(evaluation_input)
            score_values = {
                "response_quality": 4,
                "reasoning_clarity": 4,
                "factual_support": 4,
                "instruction_following": 4,
                "formatting": 4,
                "completeness": 4,
            }
            return json.dumps({
                "evaluations": [
                    _judge_payload(
                        "R1",
                        "Current provider response.",
                        score_values,
                    )
                ]
            })

        with patch(
            "backend.services.evaluator._request_gemini_judgment",
            new=judge,
        ):
            evaluations = await evaluate_responses(
                "What does that mean?",
                None,
                [_result("groq", "Current provider response.")],
                conversation_contexts={
                    "groq": [("Define evaluation.", "Prior provider answer.")]
                },
            )

        self.assertIn("groq", evaluations.scores)
        self.assertEqual(observed_input["prompt"], "What does that mean?")
        self.assertEqual(
            observed_input["conversation_context"],
            [{"user": "Define evaluation.", "assistant": "Prior provider answer."}],
        )

    async def test_judge_api_scores_are_mapped_to_anonymized_candidates(self) -> None:
        responses = [
            _result("nvidia", "ordinary response"),
            _result("sambanova", "strong reference match"),
        ]
        observed_request: dict[str, object] = {}
        observed_inputs: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            observed_request["headers"] = dict(request.headers)
            request_payload = json.loads(request.content)
            observed_request["generation_config"] = request_payload[
                "generationConfig"
            ]
            observed_request["system_instruction"] = request_payload[
                "systemInstruction"
            ]["parts"][0]["text"]
            evaluation_input = json.loads(
                request_payload["contents"][0]["parts"][0]["text"]
            )
            observed_inputs.append(evaluation_input)
            response_data = evaluation_input["response"]
            content = response_data["content"]
            score = 5 if "strong" in content else 2
            output = {
                "evaluations": [
                    _judge_payload(
                        response_data["response_id"],
                        content,
                        {
                            "response_quality": score,
                            "reasoning_clarity": score,
                            "factual_support": score,
                            "instruction_following": score,
                            "formatting": score,
                            "completeness": score,
                        },
                    )
                ]
            }
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": (
                                            f"```json\n{json.dumps(output)}\n```"
                                        )
                                    }
                                ]
                            }
                        }
                    ]
                },
                request=request,
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch(
                "backend.services.evaluator.httpx.AsyncClient",
                side_effect=lambda **kwargs: real_async_client(
                    transport=transport,
                    timeout=kwargs.get("timeout"),
                ),
            ),
        ):
            evaluations = await evaluate_responses(
                "question",
                "reference",
                responses,
            )

        self.assertEqual(evaluations.scores["sambanova"].factual_support.score, 5)
        self.assertEqual(evaluations.scores["nvidia"].factual_support.score, 2)
        self.assertEqual(
            evaluations.scores["sambanova"].instruction_following.score, 5
        )
        generation_config = observed_request["generation_config"]
        self.assertEqual(generation_config["temperature"], 0)
        self.assertEqual(generation_config["responseMimeType"], "application/json")
        self.assertIn(
            "response_quality measures",
            observed_request["system_instruction"],
        )
        self.assertIn(
            "Score formatting independently",
            observed_request["system_instruction"],
        )
        response_schema = generation_config["responseJsonSchema"]
        self.assertEqual(response_schema["required"], ["evaluations"])
        evaluation_schema = response_schema["properties"]["evaluations"]["items"]
        self.assertEqual(
            evaluation_schema["required"],
            [
                "response_id",
                "response_quality",
                "reasoning_clarity",
                "factual_support",
                "instruction_following",
                "formatting",
                "completeness",
                "rationale",
            ],
        )
        self.assertEqual(len(observed_inputs), 2)
        self.assertEqual(
            [evaluation_input["response"]["content"] for evaluation_input in observed_inputs],
            ["ordinary response", "strong reference match"],
        )
        for evaluation_input in observed_inputs:
            self.assertEqual(evaluation_input["prompt"], "question")
            self.assertEqual(evaluation_input["reference_answer"], "reference")
            self.assertEqual(evaluation_input["response"]["response_id"], "R1")
            self.assertEqual(set(evaluation_input), {"prompt", "reference_answer", "response"})
        serialized = json.dumps(observed_inputs)
        self.assertNotIn("nvidia", serialized)
        self.assertNotIn("sambanova", serialized)

    async def test_invalid_judge_json_is_rejected_without_fabricated_scores(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": "not valid JSON"}]}}
                    ]
                },
                request=request,
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch(
                "backend.services.evaluator.httpx.AsyncClient",
                side_effect=lambda **kwargs: real_async_client(
                    transport=transport,
                    timeout=kwargs.get("timeout"),
                ),
            ),
        ):
            evaluations = await evaluate_responses(
                "question",
                None,
                [_result("sambanova", "answer")],
            )

        self.assertIn(
            "invalid score payload",
            evaluations.errors["sambanova"],
        )
        self.assertIn("Invalid JSON", evaluations.errors["sambanova"])

    async def test_five_with_unmet_metric_requirements_is_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            evaluation_input = json.loads(
                json.loads(request.content)["contents"][0]["parts"][0]["text"]
            )
            response_data = evaluation_input["response"]
            scores = {
                "response_quality": 4,
                "reasoning_clarity": 4,
                "factual_support": 4,
                "instruction_following": 4,
                "formatting": 4,
                "completeness": 5,
            }
            output = {
                "evaluations": [
                    _judge_payload(
                        response_data["response_id"],
                        response_data["content"],
                        scores,
                        ["architecture"],
                    )
                ]
            }
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": json.dumps(output)}]}}
                    ]
                },
                request=request,
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch(
                "backend.services.evaluator.httpx.AsyncClient",
                side_effect=lambda **kwargs: real_async_client(
                    transport=transport,
                    timeout=kwargs.get("timeout"),
                ),
            ),
        ):
            evaluations = await evaluate_responses(
                "Explain the system.",
                None,
                [_result("groq", "A brief answer.")],
            )

        self.assertIn(
            "returned a 5 for completeness",
            evaluations.errors["groq"],
        )

    async def test_missing_judge_key_is_explicit(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            evaluations = await evaluate_responses(
                "question", None, [_result("groq", "answer")]
            )
        self.assertIn("GEMINI_API_KEY", evaluations.errors["groq"])

    async def test_quality_levels_are_scored_independently_with_evidence(self) -> None:
        prompt = (
            "Explain how a hospital can optimize patient queues using AI, "
            "including architecture, algorithms and limitations."
        )
        responses = [
            _result(
                "nvidia",
                "Architecture: registration sends arrival data to a queue service. "
                "Triage rules prioritize urgent patients; a priority queue handles "
                "ordering, while a forecasting model estimates wait times. A database "
                "stores queue events and real-time updates notify staff. Pilot the "
                "system with human override, privacy controls, and monitoring. "
                "Limitations include biased data, changing demand, and integration risk.",
            ),
            _result(
                "groq",
                "AI can predict busy times, schedule appointments, and help manage "
                "the queue. Staff can use the predictions to reduce waiting.",
            ),
            _result("ollama", "AI can make hospital queues faster."),
        ]
        observed_inputs: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            request_payload = json.loads(request.content)
            evaluation_input = json.loads(
                request_payload["contents"][0]["parts"][0]["text"]
            )
            observed_inputs.append(evaluation_input)
            response_data = evaluation_input["response"]
            content = response_data["content"]
            if "priority queue" in content:
                scores = {
                    "response_quality": 5,
                    "reasoning_clarity": 4,
                    "factual_support": 4,
                    "instruction_following": 5,
                    "formatting": 4,
                    "completeness": 5,
                }
                missing = []
            elif "predict busy times" in content:
                scores = {
                    "response_quality": 3,
                    "reasoning_clarity": 3,
                    "factual_support": 3,
                    "instruction_following": 3,
                    "formatting": 4,
                    "completeness": 2,
                }
                missing = ["architecture", "detailed algorithms", "limitations"]
            else:
                scores = {
                    "response_quality": 2,
                    "reasoning_clarity": 1,
                    "factual_support": 3,
                    "instruction_following": 1,
                    "formatting": 4,
                    "completeness": 1,
                }
                missing = ["architecture", "algorithms", "limitations"]
            output = {
                "evaluations": [
                    _judge_payload(
                        response_data["response_id"],
                        content,
                        scores,
                        missing,
                    )
                ]
            }
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": json.dumps(output)}]
                            }
                        }
                    ]
                },
                request=request,
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch(
                "backend.services.evaluator.httpx.AsyncClient",
                side_effect=lambda **kwargs: real_async_client(
                    transport=transport,
                    timeout=kwargs.get("timeout"),
                ),
            ),
        ):
            evaluations = await evaluate_responses(prompt, None, responses)

        high = evaluations.scores["nvidia"]
        medium = evaluations.scores["groq"]
        low = evaluations.scores["ollama"]
        self.assertGreater(
            high.response_quality.score, medium.response_quality.score
        )
        self.assertGreater(
            medium.response_quality.score, low.response_quality.score
        )
        self.assertGreater(high.completeness.score, medium.completeness.score)
        self.assertGreater(
            medium.completeness.score, low.completeness.score
        )
        self.assertEqual(high.formatting.score, low.formatting.score)
        self.assertEqual(len(observed_inputs), 3)
        self.assertEqual(
            {
                evaluation_input["response"]["content"]
                for evaluation_input in observed_inputs
            },
            {response.response for response in responses},
        )
        self.assertTrue(
            all(
                evaluation_input["prompt"] == prompt
                and len(evaluation_input) == 3
                for evaluation_input in observed_inputs
            )
        )
        self.assertIn("Architecture:", high.factual_support.evidence)

        score_and_rank(responses, evaluations.scores)
        self.assertEqual(
            responses[0].score_evidence.response_quality.evidence[:20],
            responses[0].response[:20],
        )
        self.assertEqual(
            responses[1].score_evidence.completeness.missing_requirements,
            ["architecture", "detailed algorithms", "limitations"],
        )

    async def test_identical_response_has_consistent_scores_across_providers(self) -> None:
        shared_response = "A concise, structured answer with accurate facts."

        def handler(request: httpx.Request) -> httpx.Response:
            evaluation_input = json.loads(
                json.loads(request.content)["contents"][0]["parts"][0]["text"]
            )
            response_data = evaluation_input["response"]
            score = 4 if "accurate" in response_data["content"] else 2
            scores = {
                "response_quality": score,
                "reasoning_clarity": score,
                "factual_support": score,
                "instruction_following": score,
                "formatting": score,
                "completeness": score,
            }
            output = {
                "evaluations": [
                    _judge_payload(
                        response_data["response_id"],
                        response_data["content"],
                        scores,
                    )
                ]
            }
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": json.dumps(output)}]}}
                    ]
                },
                request=request,
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch(
                "backend.services.evaluator.httpx.AsyncClient",
                side_effect=lambda **kwargs: real_async_client(
                    transport=transport,
                    timeout=kwargs.get("timeout"),
                ),
            ),
        ):
            evaluations = await evaluate_responses(
                "Explain the topic.",
                None,
                [
                    _result("nvidia", shared_response),
                    _result("ollama", shared_response),
                ],
            )

        self.assertEqual(
            evaluations.scores["nvidia"].model_dump(),
            evaluations.scores["ollama"].model_dump(),
        )

    async def test_unsupported_claim_reduces_factual_support(self) -> None:
        responses = [
            _result("nvidia", "A priority queue can order patients by triage score."),
            _result(
                "ollama",
                "A priority queue guarantees that all patients have zero wait time.",
            ),
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            evaluation_input = json.loads(
                json.loads(request.content)["contents"][0]["parts"][0]["text"]
            )
            response_data = evaluation_input["response"]
            unsupported = "guarantees" in response_data["content"]
            scores = {
                "response_quality": 3,
                "reasoning_clarity": 3,
                "factual_support": 1 if unsupported else 4,
                "instruction_following": 3,
                "formatting": 3,
                "completeness": 3,
            }
            output = {
                "evaluations": [
                    _judge_payload(
                        response_data["response_id"],
                        response_data["content"],
                        scores,
                    )
                ]
            }
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": json.dumps(output)}]}}
                    ]
                },
                request=request,
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch(
                "backend.services.evaluator.httpx.AsyncClient",
                side_effect=lambda **kwargs: real_async_client(
                    transport=transport,
                    timeout=kwargs.get("timeout"),
                ),
            ),
        ):
            evaluations = await evaluate_responses("Explain queues.", None, responses)

        self.assertGreater(
            evaluations.scores["nvidia"].factual_support.score,
            evaluations.scores["ollama"].factual_support.score,
        )

    async def test_gemini_response_uses_a_different_provider_as_judge(self) -> None:
        response = _result("gemini", "A response to evaluate.")
        expected = _judge_score("R1", {"response_quality": 3})
        judge_output = json.dumps(
            {"evaluations": [expected.model_dump()]}
        )

        with patch(
            "backend.services.evaluator.generate_response",
            new=AsyncMock(
                return_value=type(
                    "JudgeResponse",
                    (),
                    {"response": judge_output},
                )()
            ),
        ) as generate:
            evaluations = await evaluate_responses("Prompt.", None, [response])

        generate.assert_awaited_once()
        self.assertEqual(generate.await_args.args[0], "nvidia")
        self.assertEqual(generate.await_args.kwargs["temperature"], 0)
        self.assertIn(
            "Evaluate exactly one response",
            generate.await_args.kwargs["system_prompt"],
        )
        self.assertEqual(
            generate.await_args.kwargs["response_format"],
            "json_object",
        )
        self.assertEqual(evaluations.scores["gemini"].response_quality.score, 3)

    async def test_all_providers_are_scored_and_gemini_is_ranked(self) -> None:
        responses = [
            _result(model, f"answer-{model}", elapsed_ms=index + 1)
            for index, model in enumerate(SUPPORTED_MODELS)
        ]
        nvidia_inputs: list[dict[str, object]] = []
        gemini_inputs: list[dict[str, object]] = []

        async def generate_nvidia(
            model: str,
            prompt: str,
            **kwargs: object,
        ) -> SimpleNamespace:
            self.assertEqual(model, "nvidia")
            self.assertEqual(kwargs["response_format"], "json_object")
            nvidia_inputs.append(json.loads(prompt))
            judgment = _judge_payload(
                "R1",
                nvidia_inputs[-1]["response"]["content"],
                {
                    "response_quality": 3,
                    "reasoning_clarity": 3,
                    "factual_support": 3,
                    "instruction_following": 3,
                    "formatting": 3,
                    "completeness": 3,
                },
            )
            return SimpleNamespace(
                response=json.dumps({"evaluations": [judgment]})
            )

        def gemini_handler(request: httpx.Request) -> httpx.Response:
            request_payload = json.loads(request.content)
            evaluation_input = json.loads(
                request_payload["contents"][0]["parts"][0]["text"]
            )
            gemini_inputs.append(evaluation_input)
            content = evaluation_input["response"]["content"]
            judgment = _judge_payload(
                "R1",
                content,
                {
                    "response_quality": 4,
                    "reasoning_clarity": 4,
                    "factual_support": 4,
                    "instruction_following": 4,
                    "formatting": 4,
                    "completeness": 4,
                },
            )
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps(
                                            {"evaluations": [judgment]}
                                        )
                                    }
                                ]
                            }
                        }
                    ]
                },
                request=request,
            )

        transport = httpx.MockTransport(gemini_handler)
        real_async_client = httpx.AsyncClient
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch(
                "backend.services.evaluator.httpx.AsyncClient",
                side_effect=lambda **kwargs: real_async_client(
                    transport=transport,
                    timeout=kwargs.get("timeout"),
                ),
            ),
            patch(
                "backend.services.evaluator.generate_response",
                new=generate_nvidia,
            ),
        ):
            evaluations = await evaluate_responses(
                "Evaluate the responses independently.",
                None,
                responses,
            )

        self.assertEqual(set(evaluations.scores), set(SUPPORTED_MODELS))
        self.assertEqual(evaluations.errors, {})
        self.assertEqual(len(nvidia_inputs), 1)
        self.assertEqual(
            nvidia_inputs[0]["response"]["content"],
            "answer-gemini",
        )
        self.assertEqual(
            evaluations.scores["gemini"].response_quality.score,
            3,
        )
        self.assertEqual(
            {item["response"]["content"] for item in gemini_inputs},
            {f"answer-{model}" for model in SUPPORTED_MODELS if model != "gemini"},
        )
        winners, _, _ = score_and_rank(responses, evaluations.scores)
        self.assertEqual(
            {response.model for response in responses if response.rank is not None},
            set(SUPPORTED_MODELS),
        )
        self.assertEqual(responses[0].rank, 4)
        self.assertEqual(winners, ["nvidia"])

    async def test_nvidia_judge_failure_does_not_score_or_rank_gemini(self) -> None:
        responses = [
            _result(model, f"answer-{model}")
            for model in SUPPORTED_MODELS
        ]

        async def fail_nvidia(*args: object, **kwargs: object) -> None:
            self.assertEqual(args[0], "nvidia")
            self.assertEqual(kwargs["response_format"], "json_object")
            raise ProviderError("synthetic NVIDIA judge failure")

        def gemini_handler(request: httpx.Request) -> httpx.Response:
            evaluation_input = json.loads(
                json.loads(request.content)["contents"][0]["parts"][0]["text"]
            )
            content = evaluation_input["response"]["content"]
            scores = {
                name: 4
                for name in (
                    "response_quality",
                    "reasoning_clarity",
                    "factual_support",
                    "instruction_following",
                    "formatting",
                    "completeness",
                )
            }
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "evaluations": [
                                                    _judge_payload("R1", content, scores)
                                                ]
                                            }
                                        )
                                    }
                                ]
                            }
                        }
                    ]
                },
                request=request,
            )

        transport = httpx.MockTransport(gemini_handler)
        real_async_client = httpx.AsyncClient
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch(
                "backend.services.evaluator.httpx.AsyncClient",
                side_effect=lambda **kwargs: real_async_client(
                    transport=transport,
                    timeout=kwargs.get("timeout"),
                ),
            ),
            patch(
                "backend.services.evaluator.generate_response",
                new=fail_nvidia,
            ),
        ):
            evaluations = await evaluate_responses("Prompt.", None, responses)

        self.assertEqual(
            set(evaluations.scores),
            set(SUPPORTED_MODELS) - {"gemini"},
        )
        self.assertEqual(
            evaluations.errors,
            {"gemini": "NVIDIA evaluation request failed: synthetic NVIDIA judge failure"},
        )
        score_and_rank(responses, evaluations.scores)
        gemini = next(item for item in responses if item.model == "gemini")
        self.assertIsNone(gemini.quality_score)
        self.assertIsNone(gemini.rank)

    def test_quality_score_uses_latency_to_break_ties(self) -> None:
        responses = [
            _result("slow-best", "answer", elapsed_ms=100),
            _result("fast-low", "answer", elapsed_ms=1),
            _result("tied-best", "answer", elapsed_ms=50),
        ]

        evaluations = {
            "slow-best": _judge_score("R1"),
            "fast-low": _judge_score(
                "R2",
                {
                    "response_quality": 4,
                    "reasoning_clarity": 4,
                    "factual_support": 4,
                    "instruction_following": 4,
                    "formatting": 4,
                    "completeness": 4,
                },
            ),
            "tied-best": _judge_score("R3"),
        }

        winners, best_response, best_responses = score_and_rank(
            responses, evaluations
        )

        self.assertEqual(winners, ["tied-best"])
        self.assertEqual(best_response.model, "tied-best")
        self.assertEqual(
            [item.model for item in best_responses],
            ["tied-best"],
        )
        self.assertEqual(
            {item.model: item.rank for item in responses},
            {"slow-best": 2, "fast-low": 3, "tied-best": 1},
        )

    def test_quality_score_requires_all_six_metrics_without_filling_missing(self) -> None:
        responses = [
            _result("strong", "complete and relevant"),
            _result("partial", "some useful information"),
        ]
        evaluations = {
            "strong": _judge_score(
                "R1",
                {
                    "response_quality": 5,
                    "reasoning_clarity": 4,
                    "factual_support": 4,
                    "instruction_following": 5,
                    "formatting": 4,
                    "completeness": 4,
                },
            ),
            "partial": _judge_score(
                "R2",
                {
                    "response_quality": 3,
                    "reasoning_clarity": 3,
                    "factual_support": None,
                    "instruction_following": 3,
                    "formatting": 3,
                    "completeness": 3,
                },
            ),
        }

        winners, _, _ = score_and_rank(responses, evaluations)

        self.assertEqual(winners, ["strong"])
        self.assertEqual(responses[0].quality_score, 86.67)
        self.assertIsNone(responses[1].quality_score)
        self.assertIsNone(responses[1].scores.accuracy)
        self.assertEqual(responses[1].score_source, "ai_rubric")

    def test_unavailable_rubric_scores_do_not_create_a_quality_score_or_winner(self) -> None:
        response = _result("unknown", "Unverifiable claim.")
        evaluation = _judge_score(
            "R1",
            {
                "response_quality": None,
                "reasoning_clarity": None,
                "factual_support": None,
                "instruction_following": None,
                "formatting": None,
                "completeness": None,
            },
        )

        winners, best_response, best_responses = score_and_rank(
            [response],
            {"unknown": evaluation},
        )

        self.assertEqual(winners, [])
        self.assertIsNone(best_response)
        self.assertEqual(best_responses, [])
        self.assertIsNone(response.quality_score)
        self.assertIsNone(response.rank)
        self.assertEqual(response.score_source, "ai_rubric")

    def test_rubric_scores_are_bounded_to_one_through_five(self) -> None:
        with self.assertRaises(ValidationError):
            _MetricJudgment(
                score=0,
                reason="Scores use a one-to-five scale.",
                evidence="Evidence.",
                missing_requirements=[],
            )

    def test_simple_checks_are_deterministic_and_do_not_claim_correctness(self) -> None:
        complete = calculate_simple_checks("Two words, clearly stated.")
        unfinished = calculate_simple_checks("Two words without punctuation")
        empty = calculate_simple_checks(" \n ")

        self.assertEqual(complete.word_count, 4)
        self.assertTrue(complete.ending_check)
        self.assertFalse(unfinished.ending_check)
        self.assertEqual(unfinished.word_count, 4)
        self.assertEqual(empty.word_count, 0)
        self.assertIsNone(empty.ending_check)

    def test_all_failed_responses_cannot_produce_a_winner(self) -> None:
        failed = _result("offline", "", elapsed_ms=0)
        failed.response = None
        failed.error = "provider unavailable"

        winners, best_response, best_responses = score_and_rank(
            [failed], {"offline": _judge_score("R1")}
        )

        self.assertEqual(winners, [])
        self.assertIsNone(best_response)
        self.assertEqual(best_responses, [])
        self.assertIsNone(failed.rank)
        self.assertIsNone(failed.quality_score)



if __name__ == "__main__":
    unittest.main()
