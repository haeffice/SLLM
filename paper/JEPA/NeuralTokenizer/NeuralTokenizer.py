"""Standalone JEPA-Neural-Tokenizer module (PyTorch 2.8).

Self-contained reimplementation of **JEPA as a Neural Tokenizer** (*Learning
Robust Speech Representations with Density Adaptive Attention*, Ioannides
et al. incl. LeCun, arXiv:2512.07168, 2025-12). A two-stage, **reversible**
speech tokenizer:

  Stage 1 (SSL, `NeuralTokenizer_Trainer.Stage1Trainer`):
      raw 24 kHz waveform -> Conv1D down-sampling stack (hop 9600 -> 2.5 Hz)
      -> Density Adaptive Attention (Gaussian-mixture gating) -> 8 Conformer
      blocks. Trained by masked-latent JEPA prediction against an EMA teacher.

  Stage 2 (reconstruction, `NeuralTokenizer_Trainer.Stage2Trainer`):
      encoder features -> FSQ (finite scalar quantization, no codebook)
      -> mixed-radix token packing (19 tokens/frame -> 47.5 tok/s) -> HiFi-GAN
      decoder -> waveform. Trained with L1 + multi-resolution STFT + GAN.

This file holds the *deployable* tokenizer (encoder + FSQ + mixed-radix codec
+ HiFi-GAN decoder) and the shared building blocks reused by the trainer.
`from_checkpoint` loads a Stage-1 (`encoder.*`) and/or Stage-2
(`+ proj_in/proj_out/decoder`) checkpoint, aborting if nothing loads.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("NeuralTokenizer")


# =============================================================================
# Conv1D down-sampling front-end  (24 kHz -> 2.5 Hz at hop 9600)
# =============================================================================

class ConvFrontend(nn.Module):
    """Stem (stride 1) + strided Conv1D stack. `frames ≈ L / prod(strides)`."""

    def __init__(self, channels: tuple, strides: tuple, stem_kernel: int = 7):
        super().__init__()
        assert len(channels) == len(strides) + 1, \
            "channels must be len(strides)+1"
        self.stem = nn.Conv1d(1, channels[0], stem_kernel, 1,
                              padding=stem_kernel // 2)
        convs = []
        for i, s in enumerate(strides):
            k = 2 * s
            convs.append(nn.Conv1d(channels[i], channels[i + 1], k, s,
                                   padding=s // 2))
        self.convs = nn.ModuleList(convs)
        self.out_dim = channels[-1]
        self.hop = int(math.prod(strides))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, 1, L) -> features (B, out_dim, T)."""
        x = F.gelu(self.stem(wav))
        for c in self.convs:
            x = F.gelu(c(x))
        return x


# =============================================================================
# Density Adaptive Attention Mechanism (Gaussian-mixture gating)
# =============================================================================

class DAAM(nn.Module):
    """Gaussian-mixture density-adaptive gating over the temporal axis.

    For features x (B, C, T) with per-channel temporal mean μ and std σ, K
    learnable Gaussian components (mean offset δ_k, softplus log-scale ν_k)
    define a per-time log-density that gently modulates the features:
        z_{k,t} = (x_t - (μ + δ_k)) / (σ·σ̃_k + ε)
        logG    = logsumexp_k[ -½z² - log σ̃_k - ½log2π ] - log K
        y_t     = x_t · exp(α · logG)
    """

    def __init__(self, channels: int, k: int = 4, alpha: float = 0.05,
                 eps: float = 1e-3):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.eps = eps
        self.delta = nn.Parameter(torch.zeros(k, channels))
        self.nu = nn.Parameter(torch.full((k, channels), math.log(0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _ = x.shape
        mu = x.mean(dim=2, keepdim=True)                # (B, C, 1)
        sigma = x.std(dim=2, keepdim=True)              # (B, C, 1)
        scale = F.softplus(self.nu) + self.eps          # (K, C)
        log_two_pi = math.log(2.0 * math.pi)
        logps = []
        for k in range(self.k):
            d = self.delta[k].view(1, C, 1)
            s = scale[k].view(1, C, 1)
            z = (x - (mu + d)) / (sigma * s + self.eps)
            logps.append(-0.5 * z * z - torch.log(s) - 0.5 * log_two_pi)
        logp = torch.stack(logps, dim=0)                # (K, B, C, T)
        log_g = torch.logsumexp(logp, dim=0) - math.log(self.k)
        return x * torch.exp(self.alpha * log_g)


# =============================================================================
# Conformer block
# =============================================================================

class _FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * mult), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(dim * mult, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class _ConformerConv(nn.Module):
    def __init__(self, dim: int, kernel: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, 2 * dim, 1)
        self.dw = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.bn = nn.GroupNorm(1, dim)
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                               # (B, T, D)
        x = self.norm(x).transpose(1, 2)                # (B, D, T)
        x = F.glu(self.pw1(x), dim=1)
        x = self.pw2(F.silu(self.bn(self.dw(x))))
        return self.drop(x.transpose(1, 2))


class ConformerBlock(nn.Module):
    """Macaron Conformer: ½FFN + MHSA + ConvModule + ½FFN + LayerNorm."""

    def __init__(self, dim: int, heads: int, ff_mult: int = 4,
                 conv_kernel: int = 15, dropout: float = 0.0):
        super().__init__()
        self.ffn1 = _FeedForward(dim, ff_mult, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.conv = _ConformerConv(dim, conv_kernel, dropout)
        self.ffn2 = _FeedForward(dim, ff_mult, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        h = self.attn_norm(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


# =============================================================================
# JEPA encoder (front-end + DAAM + Conformer); split for masked SSL reuse
# =============================================================================

class JEPAEncoder(nn.Module):
    def __init__(self, cfg: "NeuralTokenizerConfig"):
        super().__init__()
        self.frontend = ConvFrontend(cfg.conv_channels, cfg.conv_strides,
                                     cfg.stem_kernel)
        d = self.frontend.out_dim
        self.daam = DAAM(d, cfg.daam_k, cfg.daam_alpha) if cfg.use_daam else None
        self.in_norm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([
            ConformerBlock(d, cfg.num_heads, cfg.ff_mult, cfg.conv_kernel,
                           cfg.dropout) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(d)
        self.out_dim = d
        self.hop = self.frontend.hop

    def frontend_features(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, 1, L) -> pre-Conformer features (B, T, d)."""
        f = self.frontend(wav)                          # (B, d, T)
        if self.daam is not None:
            f = self.daam(f)
        return self.in_norm(f.transpose(1, 2))          # (B, T, d)

    def encode_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Conformer over (B, T, d) features -> (B, T, d)."""
        x = feats
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Full encode: (B, 1, L) -> (B, T, d) at the encoder frame rate."""
        return self.encode_features(self.frontend_features(wav))


# =============================================================================
# Finite Scalar Quantization (no learnable codebook)
# =============================================================================

class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al. 2023). Bounds each dim with
    tanh, rounds to one of `level` values (straight-through), and exposes
    per-dim integer indices for the mixed-radix packer."""

    def __init__(self, dim: int, level: int = 4, eps: float = 1e-3):
        super().__init__()
        self.dim = dim
        self.register_buffer("levels",
                             torch.full((dim,), float(level)), persistent=False)
        self.eps = eps

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        L = self.levels
        half_l = (L - 1) * (1 - self.eps) / 2
        offset = torch.where(L % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z (..., dim) unbounded -> (z_norm (..., dim) in ~[-1,1], indices)."""
        zb = self._bound(z)
        quant = zb + (torch.round(zb) - zb).detach()    # round STE
        half = (self.levels // 2).clamp_min(1)
        z_norm = quant / half                           # ~[-1, 1] for decoder
        indices = (quant + half).round().long()         # [0, level-1]
        return z_norm, indices

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        half = (self.levels // 2).clamp_min(1)
        return (indices.float() - half) / half


# =============================================================================
# Mixed-radix token packing (reversible)
# =============================================================================

class MixedRadixCodec(nn.Module):
    """Pack per-dim FSQ indices into compact integer tokens and back.
    `dim=128, group=7, radix=4` -> 19 groups -> 47.5 tok/s at 2.5 Hz."""

    def __init__(self, dim: int = 128, group: int = 7, radix: int = 4):
        super().__init__()
        self.dim = dim
        self.group = group
        self.radix = radix
        self.n_groups = math.ceil(dim / group)
        self.padded = self.n_groups * group
        w = torch.tensor([radix ** (group - 1 - k) for k in range(group)],
                         dtype=torch.long)
        self.register_buffer("weights", w, persistent=False)  # (group,)
        self.vocab_size = radix ** group

    def pack(self, indices: torch.Tensor) -> torch.Tensor:
        """indices (..., dim) -> tokens (..., n_groups)."""
        *lead, d = indices.shape
        if d < self.padded:
            pad = torch.zeros(*lead, self.padded - d, dtype=indices.dtype,
                              device=indices.device)
            indices = torch.cat([indices, pad], dim=-1)
        groups = indices.reshape(*lead, self.n_groups, self.group)
        return (groups * self.weights).sum(dim=-1)            # (..., n_groups)

    def unpack(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (..., n_groups) -> indices (..., dim)."""
        *lead, _ = tokens.shape
        digits = (tokens.unsqueeze(-1) // self.weights) % self.radix
        indices = digits.reshape(*lead, self.padded)
        return indices[..., :self.dim]


# =============================================================================
# HiFi-GAN generator (decoder)
# =============================================================================

class _ResBlock(nn.Module):
    """Multi-receptive-field residual block (HiFi-GAN)."""

    def __init__(self, ch: int, kernel: int, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(ch, ch, kernel, 1, dilation=d,
                      padding=(kernel - 1) // 2 * d) for d in dilations])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(ch, ch, kernel, 1, padding=(kernel - 1) // 2)
            for _ in dilations])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            h = c2(F.leaky_relu(c1(F.leaky_relu(x, 0.1)), 0.1))
            x = x + h
        return x


class HiFiGANGenerator(nn.Module):
    """ConvTranspose1D upsampler with MRF residual blocks -> waveform."""

    def __init__(self, in_dim: int, upsample_strides: tuple,
                 channels: tuple, resblock_kernels=(3, 7, 11),
                 resblock_dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5))):
        super().__init__()
        assert len(channels) == len(upsample_strides)
        self.conv_pre = nn.Conv1d(in_dim, in_dim, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.kernels = resblock_kernels
        prev = in_dim
        for s, ch in zip(upsample_strides, channels):
            k = 2 * s
            self.ups.append(nn.ConvTranspose1d(prev, ch, k, s,
                                               padding=(k - s) // 2))
            for rk, rd in zip(resblock_kernels, resblock_dilations):
                self.resblocks.append(_ResBlock(ch, rk, rd))
            prev = ch
        self.conv_post = nn.Conv1d(prev, 1, 7, 1, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, in_dim, T) -> waveform (B, 1, L)."""
        x = self.conv_pre(x)
        nk = len(self.kernels)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, 0.1))
            xs = sum(self.resblocks[i * nk + j](x) for j in range(nk)) / nk
            x = xs
        return torch.tanh(self.conv_post(F.leaky_relu(x, 0.1)))


# =============================================================================
# Config
# =============================================================================

@dataclass
class NeuralTokenizerConfig:
    sample_rate: int = 24_000
    clip_seconds: float = 15.0

    # Conv front-end: hop = prod(strides). (8,8,5,5,6) -> 9600 -> 2.5 Hz.
    conv_channels: tuple = (64, 128, 256, 384, 512, 512)
    conv_strides: tuple = (8, 8, 5, 5, 6)
    stem_kernel: int = 7

    # DAAM
    use_daam: bool = True
    daam_k: int = 4
    daam_alpha: float = 0.05

    # Conformer encoder
    num_layers: int = 8
    num_heads: int = 16
    ff_mult: int = 4
    conv_kernel: int = 15
    dropout: float = 0.0

    # Stage-1 JEPA predictor + masking + EMA
    pred_layers: int = 2
    pred_heads: int = 8
    mask_ratio: float = 0.5
    mask_min_span: int = 2
    ema_decay: float = 0.996

    # Stage-2 FSQ + mixed-radix codec
    fsq_dim: int = 128
    fsq_level: int = 4
    pack_group: int = 7

    # Stage-2 HiFi-GAN decoder (channels auto-halved from d_model if empty)
    dec_channels: tuple = ()
    dec_min_channels: int = 16

    # Stage-2 losses
    lambda_stft: float = 2.0
    lambda_gan: float = 0.1
    disc_warmup: int = 5_000

    def __post_init__(self):
        for k in ("conv_channels", "conv_strides", "dec_channels"):
            v = getattr(self, k)
            if isinstance(v, list):
                setattr(self, k, tuple(v))

    @property
    def d_model(self) -> int:
        return self.conv_channels[-1]

    @property
    def clip_samples(self) -> int:
        return int(self.sample_rate * self.clip_seconds)

    def decoder_channels(self) -> tuple:
        """Upsample output channels mirroring the encoder strides (reversed)."""
        if self.dec_channels:
            return self.dec_channels
        n = len(self.conv_strides)
        d = self.d_model
        chans = [max(self.dec_min_channels, d // (2 ** (i + 1))) for i in range(n)]
        return tuple(chans)


# =============================================================================
# Deployable tokenizer (encoder + FSQ + codec + decoder)
# =============================================================================

class NeuralTokenizer(nn.Module):
    """Full reversible tokenizer: waveform <-> discrete tokens <-> waveform."""

    def __init__(self, config: Optional[NeuralTokenizerConfig] = None):
        super().__init__()
        cfg = config or NeuralTokenizerConfig()
        self.config = cfg
        d = cfg.d_model

        self.encoder = JEPAEncoder(cfg)
        self.proj_in = nn.Linear(d, cfg.fsq_dim)
        self.fsq = FSQ(cfg.fsq_dim, cfg.fsq_level)
        self.codec = MixedRadixCodec(cfg.fsq_dim, cfg.pack_group, cfg.fsq_level)
        self.proj_out = nn.Linear(cfg.fsq_dim, d)
        self.decoder = HiFiGANGenerator(
            d, tuple(reversed(cfg.conv_strides)), cfg.decoder_channels())

    @property
    def token_rate(self) -> float:
        return (self.config.sample_rate / self.encoder.hop) * self.codec.n_groups

    # -------------------------------------------------------------------------
    # Inference pipeline
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """waveform (B, 1, L) -> encoder features (B, T, d)."""
        return self.encoder(wav)

    @torch.inference_mode()
    def tokenize(self, wav: torch.Tensor) -> torch.Tensor:
        """waveform (B, 1, L) -> packed tokens (B, T, n_groups) Long."""
        _, indices = self.fsq(self.proj_in(self.encoder(wav)))
        return self.codec.pack(indices)

    @torch.inference_mode()
    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """packed tokens (B, T, n_groups) -> waveform (B, 1, L)."""
        indices = self.codec.unpack(tokens)
        z = self.proj_out(self.fsq.indices_to_codes(indices))
        return self.decoder(z.transpose(1, 2))

    @torch.inference_mode()
    def reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        """waveform -> tokens -> waveform (full round-trip)."""
        return self.detokenize(self.tokenize(wav))

    # -------------------------------------------------------------------------
    # Checkpoint loading
    # -------------------------------------------------------------------------

    _STRIP_PREFIXES = ("module.", "_orig_mod.", "model.", "gen.", "tok.")

    @classmethod
    def _unwrap(cls, obj) -> dict[str, torch.Tensor]:
        if isinstance(obj, dict):
            for key in ("state_dict", "model_state_dict", "model", "weights"):
                if key in obj and isinstance(obj[key], dict):
                    return cls._unwrap(obj[key])
            if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj
        raise ValueError(
            f"Could not find a tensor state-dict in checkpoint ({type(obj)})")

    @classmethod
    def _filter(cls, sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for raw_k, v in sd.items():
            k = raw_k
            changed = True
            while changed:
                changed = False
                for p in cls._STRIP_PREFIXES:
                    if k.startswith(p):
                        k, changed = k[len(p):], True
            out[k] = v
        return out

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        config: Optional[NeuralTokenizerConfig] = None,
        map_location: str = "cpu",
        device: Optional[torch.device] = None,
        strict: bool = False,
    ) -> "NeuralTokenizer":
        """Build a `NeuralTokenizer` and load Stage-1 (`encoder.*`) and/or
        Stage-2 (`+ proj_*`, `decoder.*`) weights. Aborts if nothing loads."""
        model = cls(config)
        try:
            blob = torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            blob = torch.load(path, map_location=map_location, weights_only=False)
        sd = cls._filter(cls._unwrap(blob))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        total = len(sd)
        applied = total - len(unexpected)
        if applied == 0:
            raise RuntimeError(
                f"NeuralTokenizer.from_checkpoint({path!r}): 0/{total} keys "
                "applied — checkpoint/architecture mismatch, aborting.")
        if not missing and not unexpected:
            logger.info("NeuralTokenizer loaded successfully (%d/%d keys)",
                        applied, total)
        else:
            logger.warning(
                "NeuralTokenizer partial load: %d/%d keys (missing=%d, "
                "unexpected=%d) — expected when loading a Stage-1-only ckpt.",
                applied, total, len(missing), len(unexpected))
            if strict and (missing or unexpected):
                raise RuntimeError(
                    f"strict=True but missing={missing} unexpected={unexpected}")
        if device is not None:
            model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model
