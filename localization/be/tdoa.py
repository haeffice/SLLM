"""TDOA utilities: GCC-PHAT and azimuth conversion.

Pure numpy functions, no FastAPI/torch dependency. Intended to be imported
by processor.py (user-implemented 2-channel processing method).
"""

from __future__ import annotations

import numpy as np

DEFAULT_MIC_DISTANCE_M = 0.14
DEFAULT_SPEED_OF_SOUND = 343.0


def gcc_phat(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fs: int,
    max_tau: float | None = None,
    interp: int = 16,
) -> tuple[float, np.ndarray]:
    """Generalized Cross-Correlation with Phase Transform.

    Positive tau means sig_a leads sig_b (sound reached A before B).

    Args:
        sig_a, sig_b: 1-D float arrays, same sample rate.
        fs: sample rate in Hz.
        max_tau: clip search range to ±max_tau seconds. None = no clip.
        interp: integer upsampling factor for fractional-sample resolution.

    Returns:
        (tau_seconds, cc) — cc is the (cropped, shifted) correlation array
        centered on zero lag.
    """
    sig_a = np.asarray(sig_a, dtype=np.float64).ravel()
    sig_b = np.asarray(sig_b, dtype=np.float64).ravel()

    n = sig_a.size + sig_b.size
    n_fft = 1 << (n - 1).bit_length()

    A = np.fft.rfft(sig_a, n=n_fft)
    B = np.fft.rfft(sig_b, n=n_fft)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12

    cc = np.fft.irfft(R, n=n_fft * interp)

    max_shift = int(n_fft * interp / 2)
    if max_tau is not None:
        max_shift = min(int(fs * interp * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    peak = int(np.argmax(np.abs(cc))) - max_shift
    tau = peak / float(fs * interp)
    return tau, cc


def tau_to_azimuth(
    tau_seconds: float,
    mic_distance: float = DEFAULT_MIC_DISTANCE_M,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
) -> float:
    """Map TDOA to azimuth in degrees in the range [-90, +90].

    Positive azimuth = source on side of mic A (the one that received first).
    Saturates to ±90° when |tau| exceeds the physical limit (d / c).
    """
    max_tau = mic_distance / speed_of_sound
    tau_clipped = max(-max_tau, min(max_tau, float(tau_seconds)))
    sin_theta = (tau_clipped * speed_of_sound) / mic_distance
    sin_theta = max(-1.0, min(1.0, sin_theta))
    return float(np.degrees(np.arcsin(sin_theta)))


def confidence_from_cc(cc: np.ndarray) -> float:
    """Heuristic confidence: peak / mean(|cc|). Higher = sharper peak."""
    abs_cc = np.abs(cc)
    mean = float(abs_cc.mean()) + 1e-12
    peak = float(abs_cc.max())
    return peak / mean
