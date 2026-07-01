"""데모용 절차적 금속 dent 시뮬레이터 (MetalDentSimulator).

학습된 생성 모델 자리의 mock — 같은 BaseMeshPredictor 인터페이스로 서빙하며,
충격에 대한 "시간에 따른 변형 궤적"을 절차적으로 생성한다. 금속이 찌그러지는
느낌 = 영구 소성 dent(공간 가우시안) + 방사형 감쇠 링잉(파동이 퍼지며 정착).
결정적(동일 입력=동일 출력) → 피치 데모/회귀에 적합. 어떤 메쉬든 동작.

frames[0] = 원본(변위 0), frames[-1] = 정착 dent(= predict 결과).

수식(정규화 t=0..1):
  d=충격점 거리, σ=radius|bbox비례, env=exp(-(d/σ)^2)
  rise(t)=(1-e^{-t/τ1})/(1-e^{-1/τ1})      # dent 형성→정착 (rise(0)=0, rise(1)=1)
  ring(d,t)=A·sin(k·d-ω·t)·rise·e^{-t/τ2}  # 방사형 감쇠 링잉(t=0에서 0)
  disp(n,t)=fdir·depth·env·(rise+ring),  depth=scale·|force|

이 수식은 fe/app.py의 `metal_dent_simulate`에 동일하게 미러링돼 있다(오프라인 반응).
"""

from __future__ import annotations

import logging

import numpy as np

from models.base import BaseMeshPredictor

log = logging.getLogger(__name__)

DEFAULT_FRAMES = 60
MAX_FRAMES = 240
DEFAULT_RADIUS_FRACTION = 0.25  # σ = 이 비율 × bbox 대각선
_TAU_RISE = 0.22  # dent 형성 시상수
_TAU_RING = 0.16  # 링잉 감쇠 시상수
_RING_AMP = 0.35  # 링잉 상대 진폭
_RING_CYCLES = 1.5  # σ 안의 파장 수
_RING_HZ = 2.5  # t=0..1 동안 진동 수


def _num(value, cast, name: str):
    """int/float 변환. 타입 불일치도 ValueError로 통일(router 400 매핑)."""
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be {cast.__name__}, got {value!r}")


def metal_dent_trajectory(vertices: np.ndarray, action: dict) -> np.ndarray:
    """(T, N, 3) 변형 궤적. BE 모델과 FE 오프라인 반응이 공유하는 순수 함수."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {vertices.shape}")
    n = vertices.shape[0]
    if n == 0:
        raise ValueError("empty vertices")

    impact_node = _num(action.get("impact_node", 0), int, "impact_node")
    if not 0 <= impact_node < n:
        raise ValueError(f"impact_node {impact_node} out of range [0,{n})")

    try:
        force = np.asarray(action.get("force", [0.0, 0.0, 0.0]), dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(
            f"force must be length-3 numbers, got {action.get('force')!r}"
        )
    if force.shape != (3,):
        raise ValueError(f"force must be length-3, got {force.shape}")

    scale = _num(action.get("scale", 1.0), float, "scale")
    frames = _num(action.get("frames", DEFAULT_FRAMES), int, "frames")
    frames = max(2, min(frames, MAX_FRAMES))

    fmag = float(np.linalg.norm(force))
    fdir = force / fmag if fmag > 0 else force  # 0이면 변형 없음
    depth = scale * fmag  # 충격점 정착 변위 크기

    radius = action.get("radius")
    if radius is None:
        diag = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
        sigma = DEFAULT_RADIUS_FRACTION * diag if diag > 0 else 1.0
    else:
        sigma = _num(radius, float, "radius")
    if sigma <= 0:
        raise ValueError(f"radius must be > 0, got {sigma}")

    d = np.linalg.norm(vertices - vertices[impact_node], axis=1)  # (N,)
    env = np.exp(-((d / sigma) ** 2))  # (N,) 공간 가우시안
    ts = np.linspace(0.0, 1.0, frames)  # (T,)

    # rise(0)=0, rise(1)=1 로 정규화 → 마지막 프레임이 정확히 depth 만큼 눌린다.
    rise = (1.0 - np.exp(-ts / _TAU_RISE)) / (1.0 - np.exp(-1.0 / _TAU_RISE))
    k = 2.0 * np.pi * _RING_CYCLES / sigma
    omega = 2.0 * np.pi * _RING_HZ
    phase = k * d[None, :] - omega * ts[:, None]  # (T,N)
    # rise로 게이팅 → t=0에서 링잉 0(깨끗한 시작), 이후 감쇠.
    ring = _RING_AMP * np.sin(phase) * rise[:, None] * np.exp(-ts[:, None] / _TAU_RING)

    amp = env[None, :] * (rise[:, None] + ring)  # (T,N)
    disp = depth * amp[..., None] * fdir[None, None, :]  # (T,N,3)
    frames_arr = vertices[None, :, :] + disp  # (T,N,3), frames[0]=원본(변위 0)

    log.info(
        "metal_dent: T=%d N=%d impact=%d depth=%.4f sigma=%.4f",
        frames, n, impact_node, depth, sigma,
    )
    return frames_arr


class MetalDentSimulator(BaseMeshPredictor):
    model_id = "metal_dent"

    @classmethod
    def load(cls, device: str) -> "MetalDentSimulator":
        # 가중치 없음 — 즉시 준비. 실제 생성 모델이 이 자리를 대체한다.
        log.info("metal_dent simulator ready on %s", device)
        return cls(device)

    def simulate(self, vertices, faces, action) -> np.ndarray:
        return metal_dent_trajectory(vertices, action)

    def predict(self, vertices, faces, action) -> np.ndarray:
        # 정착 상태 = 궤적 마지막 프레임. frames는 simulate 전용 knob이라 predict는
        # 2프레임만 계산해도 동일 결과(linspace[-1]=1.0) — 전체 궤적 계산/할당을 아낀다.
        # frames 값 검증(잘못된 타입 → 400)은 유지.
        _num(action.get("frames", DEFAULT_FRAMES), int, "frames")
        return metal_dent_trajectory(vertices, {**action, "frames": 2})[-1]
