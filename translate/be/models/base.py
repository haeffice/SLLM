"""Common interface for speech-translation models served via /ws.

Each concrete model lives in its own module under `models/` and implements a
subclass of [Translator]. The class-level `model_id` is the key under which the
registry tracks the model and the URL query value clients use to address it
(`/ws?model=<id>`).
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
    def stream_step(
        self, pcm: bytes, src: str, tgt: str, task: str, state: dict
    ) -> dict | None:
        """Feed one PCM chunk of a live stream and optionally emit an update.

        `pcm` is raw little-endian PCM16, mono, 16 kHz — the client's wire
        format (see translate/app/src/renderer/audio/capture.js).
        `task` is ``"translate"`` (src→tgt) or ``"transcribe"`` (verbatim src).
        `src`/`tgt`/`task` reflect the live session and may change between calls
        (the client switches them in place; see routers/ws.py).
        `state` is a per-connection dict the implementation may use to carry
        accumulated audio / decoding context across calls within one WebSocket;
        it is cleared on a direction/task switch or mic-off.

        Returns a dict with at least:

            - "confirmed":  str — finalized translation so far (rendered black)
            - "prediction": str — current tentative tail (rendered gray)

        or None when there is nothing new to emit for this chunk.
        """
