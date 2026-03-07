"""FastAPI server for the Eden RAG chat interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Eden Gardening Assistant")

# Injected at startup via configure()
_rag: Any = None

_HTML = (Path(__file__).parent / "static" / "index.html").read_text


def configure(rag: Any) -> None:
    """Inject the RAG instance before uvicorn starts."""
    global _rag  # noqa: PLW0603
    _rag = rag


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    thinking: str = ""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG not initialised.")
    result = _rag.chat(req.message, thread_id=req.thread_id)
    return ChatResponse(
        reply=result.reply, thread_id=req.thread_id, thinking=result.thinking
    )
