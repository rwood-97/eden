"""FastAPI server for the Eden RAG chat interface."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Eden Gardening Assistant")

# Injected at startup via configure()
_rag: Any = None

_HTML = (Path(__file__).parent / "static" / "index.html").read_text


def configure(rag: Any) -> None:
    """Inject the RAG instance before uvicorn starts."""
    global _rag  # noqa: PLW0603
    _rag = rag


def _check_auth(x_password: str | None = Header(default=None)) -> None:
    password = os.environ.get("EDEN_PASSWORD")
    if not password:
        return
    if x_password is None or not secrets.compare_digest(
        x_password.encode(), password.encode()
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")


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


@app.post("/auth")
def auth(x_password: str | None = Header(default=None)) -> JSONResponse:
    password = os.environ.get("EDEN_PASSWORD")
    if not password:
        return JSONResponse({"ok": True})
    if x_password is None or not secrets.compare_digest(
        x_password.encode(), password.encode()
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return JSONResponse({"ok": True})


@app.post("/chat/stream")
def chat_stream(
    req: ChatRequest, x_password: str | None = Header(default=None)
) -> StreamingResponse:
    _check_auth(x_password)
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG not initialized.")

    def event_stream():
        for event in _rag.chat_stream(req.message, thread_id=req.thread_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest, x_password: str | None = Header(default=None)
) -> ChatResponse:
    _check_auth(x_password)
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG not initialized.")
    result = _rag.chat(req.message, thread_id=req.thread_id)
    return ChatResponse(
        reply=result.reply, thread_id=req.thread_id, thinking=result.thinking
    )
