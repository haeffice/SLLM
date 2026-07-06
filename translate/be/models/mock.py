"""Mock translator — the swap point for a real streaming speech-translation LLM.

Replace the body of `stream_step()` (and `load()` if your model needs weights)
with a real Speech-LLM. The signature and the returned dict's "confirmed" /
"prediction" fields are the fixed contract the client renders (black / gray).
"""

from __future__ import annotations

import logging

import torch

from models.base import Translator
from preprocess import TARGET_SAMPLE_RATE, pcm16_to_float

log = logging.getLogger("be.mock")


class MockTranslator(Translator):
    model_id = "mock"
    tags = ["Speech-LLM", "EN<->KO", "mock"]

    @classmethod
    def load(cls, device: torch.device, ckpt_path: str | None) -> "MockTranslator":
        # No weights to load — just record where the real checkpoint would go.
        if ckpt_path:
            log.info("mock translator: ckpt_path=%s (ignored by mock)", ckpt_path)
        return cls(device)

    def stream_step(
        self, pcm: bytes, src: str, tgt: str, task: str, state: dict
    ) -> dict | None:
        # Accumulate streamed audio length; emit a fake confirmed/prediction.
        samples = pcm16_to_float(pcm)
        state["samples"] = state.get("samples", 0) + len(samples)
        secs = state["samples"] / TARGET_SAMPLE_RATE

        # Mock policy: "confirm" ~one token per elapsed second (cumulative),
        # and report the running duration as the tentative prediction. The tag
        # reflects the task — transcription stays in the source language.
        n = int(secs)
        tag = f"[{src}]" if task == "transcribe" else f"[{src}->{tgt}]"
        confirmed = " ".join(f"{tag}#{i + 1}" for i in range(n))
        prediction = f"…{secs:.1f}s"
        return {"confirmed": confirmed, "prediction": prediction}
