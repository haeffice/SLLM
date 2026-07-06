"""Translation-model registry and per-model device/checkpoint resolution.

Resolves which translation models to load at startup, where each one runs, and
exposes a uniform load/translate surface to the rest of the BE.

Environment variables
---------------------
AUDIO_LLM_ENABLED   comma-separated model ids to load (default: "mock")
AUDIO_LLM_DEFAULT   model id used when `/translate` is called without
                    `?model=` (default: first id in AUDIO_LLM_ENABLED)

<MODEL_ID>_DEVICE   per-model device override, e.g. MOCK_DEVICE=cuda:1.
                    Falls back to cuda:0 when CUDA is available, otherwise cpu.
<MODEL_ID>_CKPT     per-model checkpoint path, e.g. MOCK_CKPT=/path/to/ckpt.
"""

from __future__ import annotations

import logging
import os
from typing import Type

import torch

from models.base import Translator
from models.mock import MockTranslator

log = logging.getLogger("be.translator")


REGISTRY: dict[str, Type[Translator]] = {
    MockTranslator.model_id: MockTranslator,
}


def enabled_model_ids() -> list[str]:
    raw = os.environ.get("AUDIO_LLM_ENABLED", "mock")
    ids = [m.strip() for m in raw.split(",") if m.strip()]
    if not ids:
        return ["mock"]
    return ids


def default_model_id() -> str:
    explicit = os.environ.get("AUDIO_LLM_DEFAULT")
    if explicit:
        return explicit
    return enabled_model_ids()[0]


def tags_for(model_id: str) -> list[str]:
    """Architecture tags for a model id, used by the client chip row.

    Static class metadata, so it resolves regardless of load status (the client
    can show the chips while the model is still loading).
    """
    cls = REGISTRY.get(model_id)
    return list(getattr(cls, "tags", []) or []) if cls is not None else []


def device_for(model_id: str) -> torch.device:
    env_key = f"{model_id.upper()}_DEVICE"
    explicit = os.environ.get(env_key)
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def ckpt_for(model_id: str) -> str | None:
    return os.environ.get(f"{model_id.upper()}_CKPT")


def load_one(model_id: str) -> Translator:
    if model_id not in REGISTRY:
        raise RuntimeError(
            f"unknown model id {model_id!r}; available: {list(REGISTRY)}"
        )
    device = device_for(model_id)
    ckpt = ckpt_for(model_id)
    log.info("loading model %s on %s (ckpt=%s)", model_id, device, ckpt)
    instance = REGISTRY[model_id].load(device, ckpt)
    log.info("loaded model %s", model_id)
    return instance
