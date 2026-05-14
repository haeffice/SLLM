"""Shared checkpoint-loading helper for the BAT package."""

from __future__ import annotations

import torch


def trusted_torch_load(path: str):
    """`torch.load` wrapper that disables the PyTorch 2.6+ safe-pickle default.

    SLAM-LLM/BAT 체크포인트는 가중치 외에 학습 시 사용된 omegaconf
    객체(`ListConfig`, `DictConfig`)를 메타데이터로 포함하고 있다. PyTorch 2.6
    부터 `torch.load`의 기본값이 `weights_only=True`로 바뀌면서 이런 임의
    클래스는 unpickling이 거부된다. 우리는 zhisheng01/SpatialAudio HF 데이터셋
    원본을 신뢰하므로 명시적으로 `weights_only=False`를 사용한다. 또한
    pickle이 ListConfig 같은 클래스를 객체로 복원하려면 그 모듈이 import
    가능해야 하므로 `requirements.txt`에 `omegaconf`를 유지한다.
    """
    return torch.load(path, map_location="cpu", weights_only=False)
