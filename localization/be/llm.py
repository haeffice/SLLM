"""LLM model loading and inference — PLACEHOLDER.

This file is the swap-in point for the user-implemented LLM.

- `load_model()` is invoked once at FastAPI startup (lifespan) and the
  returned object is stored on `app.state.llm_model`.
- `infer(model, payload)` is invoked per `/inference` request with the
  parsed JSON body.

Replace both function bodies with the real implementation; keep the
signatures so the rest of the wiring stays untouched.
"""

from __future__ import annotations

from typing import Any


def load_model() -> Any:
    return {"name": "placeholder-llm", "loaded": True}


def infer(model: Any, payload: dict) -> dict:
    return {
        "result": "[placeholder] implement inference here",
        "model": getattr(model, "get", lambda *_: None)("name") if isinstance(model, dict) else str(model),
        "echo": payload,
    }
