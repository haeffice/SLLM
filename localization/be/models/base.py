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
    def infer(self, wav_bytes: bytes, questions: list[str]) -> dict:
        """Run batch inference on a single WAV chunk with multiple prompts.

        The implementation should run the audio encoder once and broadcast
        its features to every question in `questions`, then return a dict
        containing at least:

            - "responses": list[str] — one answer per question, same order
            - "model_id":  str
        And ideally `"response"` (a joined string for backward-compat
        clients), `"questions"`, `"batch_size"`, and any other metadata
        the model produces.
        """
