"""Audio preprocessing shared by translation models.

Decodes raw WAV bytes to a mono 16 kHz float tensor — the canonical input the
client produces (see translate/app/src/renderer/audio/capture.js: encodeWAV
emits mono PCM16, resampled to 16 kHz).
"""

from __future__ import annotations

import io

import torch
import torchaudio

TARGET_SAMPLE_RATE = 16000


def decode_wav(wav_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Decode WAV bytes to a (mono) waveform resampled to 16 kHz.

    Returns (waveform[1, T], sample_rate). Raises ValueError on bad audio.
    """
    try:
        waveform, sample_rate = torchaudio.load_with_torchcodec(io.BytesIO(wav_bytes))
    except Exception as e:  # noqa: BLE001 — surfaced as 400 by the router
        raise ValueError(f"failed to decode audio: {e}") from e

    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, TARGET_SAMPLE_RATE
        )
        sample_rate = TARGET_SAMPLE_RATE

    return waveform, sample_rate
