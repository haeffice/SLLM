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
