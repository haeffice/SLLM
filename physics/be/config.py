"""Mesh-model registry and per-model device resolution (동적 로드 설정).

서버 시작 시(main.py lifespan) 어떤 mesh-to-mesh 모델을 어디에 로드할지
결정하고, BE 전역에 일관된 load 인터페이스를 노출한다. 로드된 인스턴스는
app.state.models[id]에 1개만 보관 → 싱글톤.

새 모델 추가 = config.py에 import 1줄 + REGISTRY 항목 1줄 + MESH_MODEL_ENABLED
변경 (main.py/routers 무수정).

Environment variables
---------------------
MESH_MODEL_ENABLED   comma-separated model ids to load (default: "dummy")
MESH_MODEL_DEFAULT   model id used when `/predict` is called without `?model=`
                     (default: first id in MESH_MODEL_ENABLED)

<MODEL_ID>_DEVICE    per-model device override, e.g. DUMMY_DEVICE=cpu.
                     기본 cpu. dummy는 torch 의존이 없어 문자열을 그대로
                     전달 — 실제 모델 추가 시 그 모델 load()에서
                     torch.device(device)로 변환한다.
"""

from __future__ import annotations

import logging
import os
from typing import Type

from models.base import BaseMeshPredictor
from models.dummy import DummyLinearDeformer
from models.metal_dent import MetalDentSimulator

log = logging.getLogger("be.config")


REGISTRY: dict[str, Type[BaseMeshPredictor]] = {
    MetalDentSimulator.model_id: MetalDentSimulator,  # 데모: 시간-시퀀스 dent
    DummyLinearDeformer.model_id: DummyLinearDeformer,  # 단일 프레임(비교용)
}

# 데모 기본값 — metal_dent(시퀀스)를 우선 로드/기본 모델로.
_DEFAULT_ENABLED = "metal_dent,dummy"


def enabled_model_ids() -> list[str]:
    raw = os.environ.get("MESH_MODEL_ENABLED", _DEFAULT_ENABLED)
    ids = [m.strip() for m in raw.split(",") if m.strip()]
    if not ids:
        return _DEFAULT_ENABLED.split(",")
    return ids


def default_model_id() -> str:
    explicit = os.environ.get("MESH_MODEL_DEFAULT")
    if explicit:
        return explicit
    return enabled_model_ids()[0]


def device_for(model_id: str) -> str:
    env_key = f"{model_id.upper()}_DEVICE"
    explicit = os.environ.get(env_key)
    if explicit:  # 빈 문자열도 cpu로 폴백 (seld/be/llm.py 규약과 동일)
        return explicit
    return "cpu"


def load_one(model_id: str) -> BaseMeshPredictor:
    if model_id not in REGISTRY:
        raise RuntimeError(
            f"unknown model id {model_id!r}; available: {list(REGISTRY)}"
        )
    device = device_for(model_id)
    log.info("loading model %s on %s", model_id, device)
    instance = REGISTRY[model_id].load(device)
    log.info("loaded model %s", model_id)
    return instance
