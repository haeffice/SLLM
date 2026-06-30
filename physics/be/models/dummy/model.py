"""구조 검증용 dummy 변형 모델 (DummyLinearDeformer).

실제 mesh-to-mesh 추론 모델 자리의 drop-in 자리다 (models/<id>/ 추가 +
config.py REGISTRY 한 줄 + MESH_MODEL_ENABLED 변경). 가중치가 없으므로
load()는 즉시 인스턴스를 만든다.

predict는 입력받은 노드를 action 방향으로 "단순히 찌그러뜨리는" 결정적
변형을 수행한다: 충격 노드(impact_node)에서 force 만큼 변위를 주고, 거리가
멀어질수록 선형으로 감쇠(linear falloff)시켜 국소적인 dent를 만든다. 같은
입력이면 항상 같은 결과 → 파이프라인 검증/회귀 테스트에 적합.
"""

from __future__ import annotations

import logging

import numpy as np

from models.base import BaseMeshPredictor

log = logging.getLogger(__name__)

# 감쇠 반경 기본값 = bbox 대각선 * 이 비율 (action["radius"]로 덮어쓸 수 있음).
DEFAULT_RADIUS_FRACTION = 0.3


def _coerce(value, cast, name: str):
    """JSON 값을 int/float로 변환. 타입 불일치(TypeError)도 ValueError로
    통일해 라우터가 400(잘못된 입력)으로 매핑하게 한다 (base.py predict 계약)."""
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be {cast.__name__}, got {value!r}")


class DummyLinearDeformer(BaseMeshPredictor):
    model_id = "dummy"

    @classmethod
    def load(cls, device: str) -> "DummyLinearDeformer":
        # 가중치 로드 자리 — dummy는 즉시 준비 완료.
        log.info("dummy deformer ready on %s", device)
        return cls(device)

    def predict(
        self, vertices: np.ndarray, faces: np.ndarray, action: dict
    ) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3), got {vertices.shape}")
        num_nodes = vertices.shape[0]
        if num_nodes == 0:
            raise ValueError("empty vertices")

        # --- action 파싱/검증 (잘못된 입력은 ValueError → router 400) ----------
        impact_node = _coerce(action.get("impact_node", 0), int, "impact_node")
        if not 0 <= impact_node < num_nodes:
            raise ValueError(
                f"impact_node {impact_node} out of range [0, {num_nodes})"
            )

        try:
            force = np.asarray(action.get("force", [0.0, 0.0, 0.0]), dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError(
                f"force must be a length-3 list of numbers, got {action.get('force')!r}"
            )
        if force.shape != (3,):
            raise ValueError(f"force must be length-3, got shape {force.shape}")

        scale = _coerce(action.get("scale", 1.0), float, "scale")

        # 감쇠 반경: 명시값 우선, 없으면 bbox 대각선 비례 (degenerate면 1.0).
        radius = action.get("radius")
        if radius is None:
            diag = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
            radius = DEFAULT_RADIUS_FRACTION * diag if diag > 0 else 1.0
        radius = _coerce(radius, float, "radius")
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")

        # --- 선형 감쇠 변형: 충격점에서 force, radius 밖은 0 ------------------
        origin = vertices[impact_node]
        dist = np.linalg.norm(vertices - origin, axis=1)  # (N,)
        weight = np.clip(1.0 - dist / radius, 0.0, 1.0)  # (N,) 1=충격점 → 0=radius
        displacement = scale * weight[:, None] * force[None, :]  # (N, 3)
        new_vertices = vertices + displacement

        log.info(
            "dummy predict: N=%d, impact_node=%d, force=%s, radius=%.3f, "
            "max|disp|=%.4f",
            num_nodes,
            impact_node,
            force.tolist(),
            radius,
            float(np.abs(displacement).max()),
        )
        return new_vertices
