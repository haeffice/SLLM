"""자유 낙하(drop test) 시뮬레이터 (FreeFallSimulator) — 절차적 데모 모델.

학습된 생성 모델 자리의 mock — metal_dent와 같은 BaseMeshPredictor 인터페이스로
서빙한다. 물체를 지정 높이·자세에서 떨어뜨려 바닥의 **여러 접촉점**에 동시
충격(다중 dent + 감쇠 바운스)을 주는 시간-시퀀스를 절차적으로 생성한다.
수식/상수는 trajectory.py (FE fe/free_fall_sim.py와 바이트 미러) 참고.

결정적(동일 입력=동일 출력) → 피치 데모/회귀에 적합. 어떤 메쉬든 동작.
"""

from __future__ import annotations

import logging

import numpy as np

from models.base import BaseMeshPredictor
from models.free_fall.trajectory import DEFAULT_FRAMES, _num, free_fall_trajectory

log = logging.getLogger(__name__)


class FreeFallSimulator(BaseMeshPredictor):
    model_id = "free_fall"

    @classmethod
    def load(cls, device: str) -> "FreeFallSimulator":
        # 가중치 없음 — 즉시 준비. 실제 생성 모델이 이 자리를 대체한다.
        log.info("free_fall simulator ready on %s", device)
        return cls(device)

    def simulate(self, vertices, faces, action) -> np.ndarray:
        frames = free_fall_trajectory(vertices, action)
        log.info(
            "free_fall: T=%d N=%d h=%s e=%s",
            frames.shape[0], frames.shape[1],
            action.get("drop_height", 1.0), action.get("restitution", 0.3),
        )
        return frames

    def predict(self, vertices, faces, action) -> np.ndarray:
        # 정착 상태 = 궤적 마지막 프레임. metal_dent와 동일하게 2프레임만 계산
        # (linspace[-1]=1.0 → 동일 결과). frames 값 검증(잘못된 타입 → 400)은 유지.
        _num(action.get("frames", DEFAULT_FRAMES), int, "frames")
        return free_fall_trajectory(vertices, {**action, "frames": 2})[-1]
