"""Common interface for speech-translation models served via /translate.

Each concrete model lives in its own module under `models/` and implements a
subclass of [Translator]. The class-level `model_id` is the key under which the
registry tracks the model and the URL query value clients use to address it
(`POST /translate?model=<id>`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Translator(ABC):
    model_id: str = ""
    # Human-readable architecture tags shown in the client's chip row.
    # The BE is the single source of truth so the chips track whichever model
    # run.sh enables (see translator.tags_for / main.health).
    tags: list[str] = []

    def __init__(self, device: torch.device):
        self.device = device

    @classmethod
    @abstractmethod
    def load(cls, device: torch.device, ckpt_path: str | None) -> "Translator":
        """Load the checkpoint and return an instance ready for translation."""

    @abstractmethod
    def translate(self, wav_bytes: bytes, src: str, tgt: str) -> dict:
        """Translate one WAV chunk from `src` language to `tgt` language.

        The implementation should return a dict containing at least:

            - "text":     str — the translated text for this chunk
            - "model_id": str
        And ideally any metadata the model produces (sample_rate, duration, …).
        """
