"""2-channel audio processor — PLACEHOLDER.

This file is the swap-in point for the user-implemented 2-channel processing
method. The TDOA utilities live in tdoa.py and can be imported here:

    from tdoa import gcc_phat, tau_to_azimuth, confidence_from_cc

Replace the body of `process_stereo` with the actual processing logic.
"""

from __future__ import annotations

import torch


def process_stereo(waveform: torch.Tensor, sample_rate: int) -> dict:
    num_channels = int(waveform.shape[0]) if waveform.dim() == 2 else 1
    num_samples = int(waveform.shape[-1])
    duration = num_samples / sample_rate if sample_rate else 0.0
    return {
        "sample_rate": sample_rate,
        "num_channels": num_channels,
        "num_samples": num_samples,
        "duration_seconds": round(duration, 3),
        "result": "[placeholder] implement 2-channel processing here",
    }
