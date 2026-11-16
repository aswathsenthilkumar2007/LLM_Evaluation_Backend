

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Available AI models
AVAILABLE_MODELS = ["chatgpt", "gemini", "claude", "llama"]


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
def ask_question(request: AskRequest):

    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {request.model}"
        )

    return {
        "model": request.model,
        "prompt": request.prompt,
        "response": "This is a sample AI response.",
        "status": "success"
    }
