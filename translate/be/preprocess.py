"""Audio helpers shared by translation models.

The client streams raw little-endian PCM16, mono, 16 kHz over the WebSocket
(see translate/app/src/renderer/audio/capture.js). `pcm16_to_float`
converts such a chunk to a normalized float32 array for model input.
"""

from __future__ import annotations

import numpy as np

TARGET_SAMPLE_RATE = 16000


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """Decode raw little-endian PCM16 bytes to a float32 array in [-1, 1]."""
    if len(pcm) % 2:
        pcm = pcm[:-1]  # drop a trailing odd byte if a chunk split a sample
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
