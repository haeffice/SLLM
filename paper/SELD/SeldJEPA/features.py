"""Shared 2-channel SELD front-end features (PyTorch 2.8).

Turns a **2-channel** waveform into a multichannel time-frequency feature stack
for both training stages of `SeldJEPA`:

    wav (2, T)  ──STFT per channel──►  X_0, X_1  (complex, F bins, T_frames)
      ├─ log-mel(ch0), log-mel(ch1)                 spectral content + implicit ILD
      └─ sin(IPD), cos(IPD)                         inter-channel phase (wrap-free)
    ────────────────────────────────────────────────────────────────────────
    feat  (C_feat = 4, T_frames, n_mels)

**Why binaural IPD+ILD and not FOA.** True First-Order Ambisonics / the
intensity vector needs **4** ambisonic channels (W, X, Y, Z); from **2** channels
a real FOA/intensity vector is mathematically unobtainable (no omni-W reference,
no height-Z) — elevation and front/back are not observable. We therefore model
the pair as binaural: spectral log-mel per channel (their difference is the ILD)
plus the inter-channel phase difference (IPD) as sin/cos. DOA is azimuth-only.

    IPD(t,f) = angle( X_0(t,f) · conj(X_1(t,f)) )      # auto-wraps to (-pi, pi]
    sinIPD = sin(IPD) ,  cosIPD = cos(IPD)             # avoid phase-wrap discontinuity

All four maps share the same (T_frames, n_mels) grid (IPD is mel-projected with a
row-normalised filterbank so its values stay in [-1, 1]). Implemented with
`torch.stft` + a hand-rolled HTK mel filterbank (no torchaudio — repo convention).

Augmentation helpers (shared, multichannel-aware):
  * `spec_augment`     — channel-CONSISTENT time/freq masking (one mask broadcast
                         across all channels so cross-channel ILD/IPD survives).
  * `lr_swap`          — swap ch0<->ch1 + negate sin(IPD); the geometry-exact
                         azimuth mirror phi -> -phi (caller flips the DOA label).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Defaults (DCASE Task-3 / SALSA-Lite style). 10 ms hop -> 100 fps STFT frames.
# -----------------------------------------------------------------------------
SAMPLE_RATE = 24_000        # Hz
N_FFT = 512                 # Hann window 512
WIN_LENGTH = 512
HOP_LENGTH = 240            # 10 ms hop @ 24 kHz  -> 100 frames / s
N_MELS = 64
N_CHANNELS = 2              # raw input channels (binaural pair)
C_FEAT = 4                  # [log-mel_0, log-mel_1, sinIPD, cosIPD]
EPS = 1e-6


# =============================================================================
# HTK mel filterbank (pure torch/numpy — no torchaudio/librosa)
# =============================================================================

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT,
                   n_mels: int = N_MELS, fmin: float = 0.0,
                   fmax: float | None = None) -> torch.Tensor:
    """Triangular HTK mel filterbank → (n_freq = n_fft//2 + 1, n_mels)."""
    fmax = fmax if fmax is not None else sample_rate / 2.0
    n_freq = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freq)
    mel_pts = torch.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)                                   # (n_mels + 2,)
    fb = torch.zeros(n_freq, n_mels)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        left = (fft_freqs - lo) / (ctr - lo + EPS)
        right = (hi - fft_freqs) / (hi - ctr + EPS)
        fb[:, m - 1] = torch.clamp(torch.minimum(left, right), min=0.0)
    return fb                                                      # (n_freq, n_mels)


# =============================================================================
# Feature extractor
# =============================================================================

@dataclass
class FeatureConfig:
    sample_rate: int = SAMPLE_RATE
    n_fft: int = N_FFT
    win_length: int = WIN_LENGTH
    hop_length: int = HOP_LENGTH
    n_mels: int = N_MELS


class AudioFeatureExtractor(nn.Module):
    """2-channel waveform → (4, T_frames, n_mels) log-mel + sin/cos-IPD stack.

    Buffers only (no learnable params); usable on CPU or GPU. Accepts a single
    clip ``(2, T)`` or a batch ``(B, 2, T)`` and returns ``(4, T_f, M)`` or
    ``(B, 4, T_f, M)`` respectively.
    """

    def __init__(self, cfg: FeatureConfig | None = None):
        super().__init__()
        self.cfg = cfg or FeatureConfig()
        c = self.cfg
        fb = mel_filterbank(c.sample_rate, c.n_fft, c.n_mels)      # (F, M)
        self.register_buffer("mel_fb", fb, persistent=False)
        # Row-sum-normalised filterbank for IPD averaging (keeps sin/cos in [-1,1]).
        self.register_buffer("mel_fb_norm", fb / (fb.sum(0, keepdim=True) + EPS),
                             persistent=False)
        self.register_buffer("window", torch.hann_window(c.win_length),
                             persistent=False)

    @property
    def c_feat(self) -> int:
        return C_FEAT

    def _stft(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (N, T) → complex STFT (N, F, T_frames)."""
        c = self.cfg
        return torch.stft(
            wav, n_fft=c.n_fft, hop_length=c.hop_length, win_length=c.win_length,
            window=self.window.to(wav.dtype), center=True, pad_mode="reflect",
            return_complex=True,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        single = wav.ndim == 2
        if single:
            wav = wav.unsqueeze(0)                                 # (1, 2, T)
        if wav.shape[1] != N_CHANNELS:
            raise ValueError(f"expected 2-channel audio, got shape {tuple(wav.shape)}")
        b = wav.shape[0]
        spec = self._stft(wav.reshape(b * N_CHANNELS, wav.shape[-1]))
        f, tf = spec.shape[-2], spec.shape[-1]
        spec = spec.reshape(b, N_CHANNELS, f, tf)                  # (B, 2, F, T_f)

        x0, x1 = spec[:, 0], spec[:, 1]                            # (B, F, T_f)
        # log-mel power per channel -> (B, T_f, M)
        mel_fb = self.mel_fb.to(spec.real.dtype)
        pow0 = (x0.abs() ** 2).transpose(1, 2) @ mel_fb           # (B, T_f, M)
        pow1 = (x1.abs() ** 2).transpose(1, 2) @ mel_fb
        logmel0 = torch.log(pow0 + EPS)
        logmel1 = torch.log(pow1 + EPS)

        # inter-channel phase difference -> sin/cos, mel-averaged -> (B, T_f, M)
        ipd = torch.angle(x0 * torch.conj(x1))                    # (B, F, T_f)
        mel_fb_norm = self.mel_fb_norm.to(spec.real.dtype)
        sin_ipd = torch.sin(ipd).transpose(1, 2) @ mel_fb_norm    # (B, T_f, M)
        cos_ipd = torch.cos(ipd).transpose(1, 2) @ mel_fb_norm

        feat = torch.stack([logmel0, logmel1, sin_ipd, cos_ipd], dim=1)  # (B, 4, T_f, M)
        return feat[0] if single else feat


# =============================================================================
# Augmentation (multichannel-aware) — operate on (C, T, M) or (B, C, T, M)
# =============================================================================

def spec_augment(feat: torch.Tensor, *, n_time_masks: int = 2, time_width: int = 16,
                 n_freq_masks: int = 2, freq_width: int = 8,
                 spectral_only: bool = False, generator: torch.Generator | None = None,
                 ) -> torch.Tensor:
    """Channel-CONSISTENT SpecAugment.

    One (time-band, freq-band) mask is sampled and broadcast across **all**
    feature channels, so within unmasked bins the cross-channel ILD/IPD
    relationship is preserved (per-channel-independent masking would corrupt the
    DOA cue). With ``spectral_only=True`` the spatial channels (sin/cos-IPD,
    indices >= 2) are left untouched (safer variant).
    """
    single = feat.ndim == 3
    if single:
        feat = feat.unsqueeze(0)
    b, c, t, m = feat.shape
    out = feat.clone()
    chan = slice(0, 2) if spectral_only else slice(0, c)

    def _randint(high: int) -> int:
        if high <= 0:
            return 0
        return int(torch.randint(0, high, (1,), generator=generator).item())

    for i in range(b):
        for _ in range(n_time_masks):
            w = min(time_width, t)
            if w > 0:
                t0 = _randint(t - w + 1)
                out[i, chan, t0:t0 + w, :] = 0.0
        for _ in range(n_freq_masks):
            w = min(freq_width, m)
            if w > 0:
                f0 = _randint(m - w + 1)
                out[i, chan, :, f0:f0 + w] = 0.0
    return out[0] if single else out


def lr_swap(feat: torch.Tensor) -> torch.Tensor:
    """Left/right mirror: swap ch0<->ch1 and negate sin(IPD).

    Corresponds to reflecting azimuth ``phi -> -phi`` (the only geometry-exact
    spatial augmentation for a 2-mic pair). cos(IPD) is even in phi and is left
    unchanged. The CALLER must mirror the azimuth DOA label to match.
    """
    out = feat.clone()
    out[..., 0, :, :], out[..., 1, :, :] = feat[..., 1, :, :], feat[..., 0, :, :]
    out[..., 2, :, :] = -feat[..., 2, :, :]                        # sin(IPD) -> -sin(IPD)
    return out
