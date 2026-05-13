"""Common interface for audio-conditioned LLMs served via /inference.

Each concrete model lives in its own subdirectory under `models/` and
implements a subclass of [AudioLLM]. The class-level `model_id` is the
key under which the registry tracks the model and the URL query value
clients use to address it (`POST /inference?model=<id>`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class AudioLLM(ABC):
    model_id: str = ""

    def __init__(self, device: torch.device):
        self.device = device

    @classmethod
    @abstractmethod
    def load(cls, device: torch.device) -> "AudioLLM":
        """Load checkpoints and return an instance ready for inference."""

    @abstractmethod
    def infer(self, wav_bytes: bytes, question: str) -> dict:
        """Run inference on a single WAV chunk.

        Returns a dict with at least:
            - "response": str (model output text)
            - "model_id": str (this model's id)
        Additional keys (sample_rate, audio_samples, latency_ms, etc.)
        are encouraged but not required.
        """
