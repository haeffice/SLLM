"""Mock translator — the swap point for a real speech-translation LLM.

Replace the body of `translate()` (and `load()` if your model needs weights)
with a real Speech-LLM. The signatures and the returned dict's `"text"` field
are the fixed contract the client depends on.
"""

from __future__ import annotations

import logging

import torch

from models.base import Translator
from preprocess import decode_wav

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

    def translate(self, wav_bytes: bytes, src: str, tgt: str) -> dict:
        waveform, sample_rate = decode_wav(wav_bytes)
        num_samples = int(waveform.shape[-1])
        duration = num_samples / sample_rate
        return {
            "text": f"[{src}->{tgt}] mock translation ({duration:.2f}s)",
            "model_id": self.model_id,
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "duration_seconds": round(duration, 3),
        }
