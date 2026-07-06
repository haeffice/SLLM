"""Common interface for mesh-to-mesh deformation models served via /predict.

Each concrete model lives in its own subdirectory under `models/` and
implements a subclass of [BaseMeshPredictor]. The class-level `model_id` is
the key under which the registry tracks the model and the value clients use
to address it (`POST /predict?model=<id>`).

새 모델 도입 = 이 클래스를 상속해 `load()` / `predict()`만 구현 → `config.py`에
import 1줄 + REGISTRY 항목 1줄 추가. main.py / routers 코드는 건드리지 않는다
(Pluggable).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseMeshPredictor(ABC):
    model_id: str = ""

    def __init__(self, device: str):
        # device는 str("cpu", "cuda:0") — dummy는 torch 의존이 없어 문자열을
        # 그대로 보관한다. 실제 모델은 load()에서 torch.device(device)로 변환.
        self.device = device

    @classmethod
    @abstractmethod
    def load(cls, device: str) -> "BaseMeshPredictor":
        """Load weights/checkpoints and return an instance ready for predict()."""

    @abstractmethod
    def predict(
        self, vertices: np.ndarray, faces: np.ndarray, action: dict
    ) -> np.ndarray:
        """Deform a mesh under an impact and return the new node coordinates.

        Args:
            vertices: (N, 3) float — node coordinates.
            faces:    (M, K) int   — face connectivity (surface triangles, K=3).
            action:   impact spec, e.g.
                      {"impact_node": 102, "force": [0.0, -10.0, 0.0]}.

        Returns:
            new_vertices: (N, 3) float — deformed node coordinates
            (정점 개수/순서는 입력과 동일해야 한다; faces는 그대로 재사용).

        Raises:
            ValueError: 잘못된 action(범위 밖 impact_node, 길이≠3 force 등).
                        router에서 400으로 매핑된다.
        """

    def simulate(
        self, vertices: np.ndarray, faces: np.ndarray, action: dict
    ) -> np.ndarray:
        """충격에 대한 시간에 따른 변형 궤적을 반환 (`POST /simulate`용).

        Returns:
            frames: (T, N, 3) float — T개 프레임의 노드 좌표. frames[0]은 보통
            원본에 가깝고 frames[-1]은 정착 상태다.

        기본 구현은 predict() 단일 프레임을 (1, N, 3)로 감싼다 — 정적 모델도
        그대로 동작한다. 시간 시퀀스를 내는 모델(예: MetalDentSimulator)은
        이 메서드를 오버라이드한다.
        """
        return np.asarray(self.predict(vertices, faces, action), dtype=np.float64)[None]
