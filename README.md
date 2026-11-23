# LLM Evaluation Backend

A FastAPI service for querying four LLM providers concurrently, scoring their
answers, and returning rankings and the best response. The existing `/models`,
`/ask`, `/compare`, and Swagger UI endpoints are retained.

## Providers

| Provider | Default model | Key / configuration |
| --- | --- | --- |
| Google Gemini | `gemini-3.5-flash-lite` | `GEMINI_API_KEY`, `GEMINI_MODEL` |
| NVIDIA NIM | `nvidia/riva-translate-4b-instruct-v2` | `NVIDIA_API_KEY`, `NVIDIA_MODEL` |
| Groq | `openai/gpt-oss-20b` | `GROQ_API_KEY` |
| Ollama | `llama3.2` | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` |

Other provider model IDs are configurable in `.env`; restart the server after
changing them. Groq uses its OpenAI-compatible chat completions API with the
fixed `openai/gpt-oss-20b` model, which is currently supported on Groq's Free
plan; the client does not select other models. Ollama runs locally and does
not require an API key. For NVIDIA, `NVIDIA_MODEL` must also appear in
`NVIDIA_FREE_MODEL_IDS`; only add model IDs whose NVIDIA Build page explicitly
shows **Free Endpoint**.

Free access is subject to provider account eligibility, model availability,
rate limits, and terms. Check each provider's current quota and terms before
use. Groq's Free plan is rate-limited; usage on a paid plan is billed at the
published model rates.

Each successful provider response is evaluated in a separate judge request
against the original prompt and optional `reference_answer`; other providers'
responses and identities are not included in that request. The configured
Gemini model judges non-Gemini responses, while the configured NVIDIA model
judges Gemini responses so a provider does not score its own output. Both judge
requests use deterministic temperature settings. Scores include per-metric
reasons, response evidence, and unmet requirements in `score_evidence`.
Evaluation quality is model-judged; ratings are AI-generated, not human
ratings. Provider responses are still returned if evaluation cannot run. A
failed judge call leaves that response unscored and records `score_error`;
successful independent judgments remain available. A metric the judge cannot
assess reliably is returned as unavailable, and an overall quality score is
only calculated when all six metrics are available.

## Configure

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If you do not already have a `.env` file, create one from the example:

```powershell
Copy-Item .env.example .env
```

If `.env` already exists, preserve it and add the entries below rather than
overwriting it. Obtain keys from each provider's developer console and put them
in the corresponding environment variables. `.env` is ignored by Git; do not
commit it.

```dotenv
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash-lite
NVIDIA_API_KEY=
NVIDIA_MODEL=nvidia/riva-translate-4b-instruct-v2
NVIDIA_FREE_MODEL_IDS=nvidia/riva-translate-4b-instruct-v2
GROQ_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=
```

Estimated cost is calculated only when providers return both token counts and
exact-model input/output rates are configured. Set
`MODEL_PRICING_USD_PER_1M_TOKENS` to a JSON object keyed by exact model ID,
with numeric `input` and `output` rates in USD per million tokens. No rates are
assumed; unconfigured prices or missing token usage display as N/A. An explicit
zero rate is required before a zero cost is reported.

Get an API key from the [Groq Console](https://console.groq.com/keys) and set
`GROQ_API_KEY` in the backend `.env` file. The client sends it as a Bearer
token to `https://api.groq.com/openai/v1/chat/completions`, with `model` fixed
to `openai/gpt-oss-20b`. Token usage is returned by Groq; estimated cost
remains unavailable unless exact-model rates are configured.

Install [Ollama for Windows](https://ollama.com/download/windows), then download
the default model and start its local server:

```powershell
ollama pull llama3.2
ollama serve
```

Set `OLLAMA_MODEL` to a model available from `ollama list`. The default base URL
is `http://localhost:11434`; set `OLLAMA_BASE_URL` only when your local Ollama
server uses a different address. A missing model returns an actionable
`ollama pull <model>` message, and an unavailable server is reported as an
individual provider failure.

The NVIDIA `nvidia/riva-translate-4b-instruct-v2` endpoint uses the
OpenAI-compatible chat-completions API and was verified with the configured
NVIDIA key. It is specialized for translation, so use translation prompts that
specify the source and target languages; it is not a general-purpose model for
unconstrained LLM comparisons. The application rejects NVIDIA models not
present in `NVIDIA_FREE_MODEL_IDS`; only add IDs shown as **Free Endpoint** in
NVIDIA Build. NVIDIA requests allow up to 180 seconds for model startup and
generation.

## Run

```powershell
python -m uvicorn backend.main:app --reload
```

The API listens at `http://127.0.0.1:8000` by default.

## Test

```powershell
python -m unittest discover -s tests -v
```

Automated tests use mocked HTTP responses and do not call provider APIs or
consume provider credits.

## Try all four providers in Swagger

1. Start the backend and open `http://127.0.0.1:8000/docs`.
2. Expand **POST `/compare`** and click **Try it out**.
3. Enter a prompt, optionally including a reference answer:

   ```json
   {
     "prompt": "Explain why leaves appear green.",
     "reference_answer": "Chlorophyll reflects green wavelengths of visible light."
   }
   ```

4. Click **Execute**. By default, the endpoint concurrently sends the prompt to
   all configured providers. The frontend can select a subset; the endpoint
   returns each selected result, HTTP status, response time, evaluation scores,
   rankings, and winner (if evaluation succeeds).

This live Swagger request uses provider quota. It only uses free models or
configured free allowances when the provider account and selected model qualify;
automated tests are fully mocked.

## Frontend

The React/Vite dashboard lives in `frontend/`. It reads the provider list from
`GET /models` and submits comparisons to `POST /compare`; omitting `providers`
from a request keeps the existing behavior of evaluating every provider.
To compare a subset, pass `providers` as a non-empty list of provider IDs
(`gemini`, `groq`, `ollama`, and/or `nvidia`).
Model Comparison is the primary multi-turn workspace: it combines persistent
conversation history, per-turn provider results, and conversation-wide
evaluation analytics. The Prompt Playground remains available for one-off
evaluations; its history is limited to the browser session. Conversations are
stored in SQLite at `backend/conversations.sqlite3` and are available through
these endpoints:

- `GET` and `POST /conversations`
- `GET`, `PATCH`, and `DELETE /conversations/{id}`
- `POST /conversations/{id}/turns`

Create a conversation first, then submit turns using the same prompt,
reference-answer, and provider fields as `/compare`:

```json
{
  "prompt": "How does that apply to the previous example?",
  "providers": ["gemini", "groq"]
}
```

The database is created automatically and ignored by Git.

Each conversation turn runs the same comparison and evaluation pipeline as
`POST /compare`. Provider histories are isolated: each provider receives prior
user prompts and only its own prior responses. The newest complete history is
limited to an estimated 4,096 tokens per provider; the current prompt remains
separate and is evaluated as the current turn, with that provider's history
supplied separately as context. Conversation turns are not capped, but SQLite
storage grows with use. This local database has no user authentication or
multi-user access controls.

```powershell
Set-Location frontend
Copy-Item .env.example .env
npm.cmd install
npm.cmd run dev
```

Set `VITE_API_BASE_URL` in `frontend/.env` if the backend is not at
`http://127.0.0.1:8000`. The FastAPI app allows the local Vite origins
`http://127.0.0.1:5173` and `http://localhost:5173`.

## Provider documentation

- [Gemini API pricing and free tier](https://ai.google.dev/gemini-api/docs/pricing)
- [Groq models and API documentation](https://console.groq.com/docs/models)
- [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [NVIDIA NIM API reference](https://docs.api.nvidia.com/nim/reference/nvidia-riva-translate-4b-instruct-v2)
- [NVIDIA Riva Translate 4B Instruct v2](https://build.nvidia.com/nvidia/riva-translate-4b-instruct-v2)
