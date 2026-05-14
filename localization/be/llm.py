"""Audio-LLM registry and per-model device resolution.

Resolves which audio LLMs to load at startup, where each one runs, and
exposes a uniform load/infer surface to the rest of the BE.

Environment variables
---------------------
AUDIO_LLM_ENABLED   comma-separated model ids to load (default: "bat")
AUDIO_LLM_DEFAULT   model id used when `/inference` is called without
                    `?model=` (default: first id in AUDIO_LLM_ENABLED)

<MODEL_ID>_DEVICE   per-model device override, e.g. BAT_DEVICE=cuda:1.
                    Falls back to cuda:0 when CUDA is available, otherwise cpu.
"""

from __future__ import annotations

import logging
import os
from typing import Type

import torch

from models.base import AudioLLM
from models.bat import BAT

log = logging.getLogger("be.llm")


REGISTRY: dict[str, Type[AudioLLM]] = {
    BAT.model_id: BAT,
}


def enabled_model_ids() -> list[str]:
    raw = os.environ.get("AUDIO_LLM_ENABLED", "bat")
    ids = [m.strip() for m in raw.split(",") if m.strip()]
    if not ids:
        return ["bat"]
    return ids


def default_model_id() -> str:
    explicit = os.environ.get("AUDIO_LLM_DEFAULT")
    if explicit:
        return explicit
    return enabled_model_ids()[0]


def device_for(model_id: str) -> torch.device:
    env_key = f"{model_id.upper()}_DEVICE"
    explicit = os.environ.get(env_key)
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_one(model_id: str) -> AudioLLM:
    if model_id not in REGISTRY:
        raise RuntimeError(
            f"unknown model id {model_id!r}; available: {list(REGISTRY)}"
        )
    device = device_for(model_id)
    log.info("loading model %s on %s", model_id, device)
    instance = REGISTRY[model_id].load(device)
    log.info("loaded model %s", model_id)
    return instance
