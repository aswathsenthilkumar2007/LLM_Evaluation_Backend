

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from backend.routes.compare import router as compare_router
from backend.routes.conversations import router as conversations_router
from backend.services.llm_clients import (
    ProviderError,
    SUPPORTED_MODELS,
    generate_response,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)

# Available AI models
AVAILABLE_MODELS = list(SUPPORTED_MODELS)
app.include_router(compare_router)
app.include_router(conversations_router)


# Request structure
class AskRequest(BaseModel):
    model: str
    prompt: str


# API 1: Get available models
@app.get("/models")
def get_models():
    return {
        "models": AVAILABLE_MODELS,
        "total": len(AVAILABLE_MODELS),
        "status": "success"
    }


# API 2: Ask a question
@app.post("/ask")
async def ask_question(request: AskRequest):

    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {request.model}"
        )

    try:
        result = await generate_response(request.model, request.prompt)
    except ProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": str(exc),
                "provider_http_status": exc.http_status,
            },
        ) from exc

    return {
        "model": request.model,
        "provider": result.provider,
        "model_id": result.model_id,
        "prompt": request.prompt,
        "response": result.response,
        "http_status": result.http_status,
        "status": "success"
    }
