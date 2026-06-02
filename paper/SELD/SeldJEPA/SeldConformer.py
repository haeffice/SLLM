"""Causal-Conformer audio encoder + LeJEPA/SIGReg objective (PyTorch 2.8).

Stage-1 of `SeldJEPA`: a **~0.1B causal Conformer** encoder for real-time Sound
Event Localization & Detection (SELD), pre-trained **self-supervised** with the
**LeJEPA** objective (Balestriero & LeCun, arXiv:2511.08544) — a JEPA prediction
loss plus **SIGReg** (Sketched Isotropic Gaussian Regularization). There is no
EMA teacher, no stop-gradient, no scheduler: one trade-off hyper-parameter
`sigreg_coeff`. This mirrors the repo's `JEPA/LeJEPA/LeJEPA.py` (same `SIGReg`
class, same multi-crop predict-the-global-views recipe) with the ViT swapped for
a streaming-safe causal Conformer over the 4-channel binaural feature stack from
`features.py`.

Pipeline (one view):
    feat (4, T_f, M)  ──CausalConvSubsampling──►  (T_enc = T_f/2, D)    50 fps
      └─ + sinusoidal positional encoding
      └─ ConformerBlock x L      Macaron-FFN / chunk-causal MHSA / causal depthwise-conv
      └─ causal pool (last streaming state)  ──►  z in R^D
    predictor(z_view) predicts the global views' z ;  SIGReg({z}) -> N(0, I)
    L = L_pred + sigreg_coeff * SIGReg

**Causality (real-time / streaming-valid).** Three leak sources are all closed:
  * conv subsampling front-end: left-pad only in time (`right_pad=0`);
  * depthwise conv module: causal (left-pad `kernel-1`, no right context);
  * self-attention: **chunk-wise causal mask** — a frame attends to its own
    100 ms chunk (`chunk_frames` encoder steps) and all earlier chunks, never to
    the future, so the algorithmic latency is bounded by one chunk;
  * LayerNorm everywhere (no BatchNorm — streaming-safe);
  * pooling uses the last frame's state (depends only on the past).

Inference (Stage-2 / streaming): `forward_features(feat)` returns per-frame
encoder states `(B, T_enc, D)`; `embed(feat)` returns the pooled `(B, D)`.
`from_checkpoint` freezes params + `eval()` and aborts if nothing loads.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("SeldConformer")


# =============================================================================
# Causal Conv2d subsampling front-end  (4, T_f, M) -> (T_enc, D)
# =============================================================================

class CausalConvSubsampling(nn.Module):
    """2-D conv subsampling with **causal** (left-only) padding in time.

    Two conv layers downsample time x2 (100 fps -> 50 fps) and frequency x4,
    then the frequency axis is flattened and projected to `d_model`. Frequency
    padding is symmetric (frequency is not a causal axis); time padding is
    left-only so frame `t` never sees a future frame.
    """

    def __init__(self, in_chans: int, d_model: int, n_mels: int, hidden: int = 128):
        super().__init__()
        self.k = 3
        self.conv1 = nn.Conv2d(in_chans, hidden, kernel_size=3, stride=(2, 2))
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, stride=(1, 2))
        self.act = nn.SiLU()
        # frequency dim after two stride-2 convs (symmetric pad 1 each side):
        f1 = (n_mels + 2 - 3) // 2 + 1
        f2 = (f1 + 2 - 3) // 2 + 1
        self.freq_out = f2
        self.proj = nn.Linear(hidden * f2, d_model)

    def _causal_pad(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F). Pad time left-only by (k-1); pad freq symmetric by 1.
        return F.pad(x, (1, 1, self.k - 1, 0))                    # (Fl, Fr, Tl, Tr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, C_in, T_f, M) -> (B, T_enc, D)."""
        x = self.act(self.conv1(self._causal_pad(x)))            # (B, H, T/2, M/2)
        x = self.act(self.conv2(self._causal_pad(x)))            # (B, H, T/2, M/4)
        b, h, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, h * f)           # (B, T_enc, H*F)
        return self.proj(x)                                      # (B, T_enc, D)


# =============================================================================
# Positional encoding + Conformer block
# =============================================================================

def sinusoidal_pos_encoding(length: int, dim: int, device, dtype) -> torch.Tensor:
    """Standard 1-D absolute sinusoidal positional encoding -> (length, dim)."""
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    half = dim // 2
    div = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / max(half, 1)))
    ang = pos * div[None, :]
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(ang)[:, : pe[:, 0::2].shape[1]]
    pe[:, 1::2] = torch.cos(ang)[:, : pe[:, 1::2].shape[1]]
    return pe.to(dtype)


def chunk_causal_mask(length: int, chunk_frames: int, device) -> torch.Tensor:
    """Bool attention mask (length, length); True = NOT allowed to attend.

    Block-causal: frame `i` may attend frame `j` iff `j`'s chunk index is <= `i`'s
    chunk index, i.e. own chunk + all earlier chunks, never the future. Latency is
    bounded by one chunk (`chunk_frames` encoder steps == 100 ms).
    """
    idx = torch.arange(length, device=device)
    chunk = idx // max(chunk_frames, 1)
    return chunk[None, :] > chunk[:, None]                        # (L, L) True=masked


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class CausalConvModule(nn.Module):
    """Conformer convolution module with a **causal** depthwise conv + LayerNorm.

    LN -> pointwise(2D) -> GLU -> causal depthwise(kernel) -> LayerNorm -> SiLU
    -> pointwise(D) -> dropout. BatchNorm (non-streaming-safe) is replaced by
    LayerNorm over the channel axis.
    """

    def __init__(self, dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim)
        self.left_pad = kernel_size - 1
        self.dw_norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.norm(x).transpose(1, 2)                          # (B, D, T)
        x = F.glu(self.pw1(x), dim=1)                             # (B, D, T)
        x = F.pad(x, (self.left_pad, 0))                          # causal left-pad
        x = self.depthwise(x)                                     # (B, D, T)
        x = self.dw_norm(x.transpose(1, 2)).transpose(1, 2)       # LayerNorm over D
        x = self.act(x)
        x = self.pw2(x)
        return self.dropout(x.transpose(1, 2))                    # (B, T, D)


class ConformerBlock(nn.Module):
    """Macaron Conformer block: ½FFN → MHSA(chunk-causal) → ConvModule → ½FFN → LN."""

    def __init__(self, dim: int, n_heads: int, ffn_expansion: int,
                 conv_kernel: int, dropout: float):
        super().__init__()
        self.ffn1 = FeedForward(dim, ffn_expansion, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = CausalConvModule(dim, conv_kernel, dropout)
        self.ffn2 = FeedForward(dim, ffn_expansion, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.attn_drop(h)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


class ConformerEncoder(nn.Module):
    """Front-end + stack of causal Conformer blocks; returns per-frame states."""

    def __init__(self, in_chans: int, n_mels: int, d_model: int, n_layers: int,
                 n_heads: int, ffn_expansion: int, conv_kernel: int,
                 chunk_frames: int, dropout: float, frontend_hidden: int = 128):
        super().__init__()
        self.d_model = d_model
        self.chunk_frames = chunk_frames
        self.frontend = CausalConvSubsampling(in_chans, d_model, n_mels, frontend_hidden)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, ffn_expansion, conv_kernel, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat (B, C_in, T_f, M) -> per-frame states (B, T_enc, D)."""
        x = self.frontend(feat)                                  # (B, T_enc, D)
        t = x.shape[1]
        x = x + sinusoidal_pos_encoding(t, self.d_model, x.device, x.dtype)[None]
        x = self.dropout(x)
        mask = chunk_causal_mask(t, self.chunk_frames, x.device)
        for blk in self.blocks:
            x = blk(x, mask)
        return x


# =============================================================================
# SIGReg — Sketched Isotropic Gaussian Regularization (verbatim from LeJEPA)
# =============================================================================

class SIGReg(nn.Module):
    """Penalise the deviation of embeddings from an isotropic Gaussian.

    Sketch the embeddings onto `num_slices` random unit directions, then for each
    1-D projection measure the Epps-Pulley empirical-characteristic-function
    distance to N(0,1) via `num_points`-node Gauss-Hermite quadrature of
    ``integral |phi_n(t) - e^{-t^2/2}|^2 e^{-t^2/2} dt``. (Same implementation as
    `JEPA/LeJEPA/LeJEPA.py` — modality-agnostic, it only sees the (N, K) matrix.)
    """

    def __init__(self, num_slices: int = 1024, num_points: int = 17):
        super().__init__()
        self.num_slices = num_slices
        self.num_points = num_points
        nodes, weights = np.polynomial.hermite.hermgauss(num_points)
        self.register_buffer("t", torch.tensor(np.sqrt(2.0) * nodes, dtype=torch.float32))
        self.register_buffer("w", torch.tensor(np.sqrt(2.0) * weights, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        v = torch.randn(d, self.num_slices, device=z.device, dtype=z.dtype)
        v = v / v.norm(dim=0, keepdim=True).clamp_min(1e-12)      # unit directions
        y = z @ v                                                 # (n, S)
        t = self.t.to(z.dtype)
        ty = y.unsqueeze(-1) * t.view(1, 1, -1)                   # (n, S, P)
        re = torch.cos(ty).mean(dim=0)                            # (S, P) Re phi_n
        im = torch.sin(ty).mean(dim=0)                            # (S, P) Im phi_n
        phi0 = torch.exp(-0.5 * t * t).view(1, -1)                # (1, P) N(0,1) CF
        integrand = (re - phi0) ** 2 + im ** 2
        per_slice = (integrand * self.w.to(z.dtype).view(1, -1)).sum(dim=-1)
        return per_slice.mean()


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.GELU()]
    return nn.Sequential(*layers)


# =============================================================================
# Config + model
# =============================================================================

@dataclass
class SeldConformerConfig:
    # input features (must match features.py / config)
    in_chans: int = 4                          # [log-mel x2, sinIPD, cosIPD]
    n_mels: int = 64
    # causal Conformer (~0.1B at d=512, L=16)
    encoder_dim: int = 512                     # d_model
    num_layers: int = 16
    num_heads: int = 8
    ffn_expansion: int = 4
    conv_kernel: int = 31                      # causal depthwise kernel
    chunk_frames: int = 5                      # encoder frames / 100 ms chunk (50 fps)
    frontend_hidden: int = 128
    dropout: float = 0.1
    pool: str = "last"                         # "last" (streaming) | "mean"
    # LeJEPA objective
    pred_hidden: int = 1024                    # predictor MLP hidden width
    sigreg_coeff: float = 1.0                  # lambda: L = L_pred + lambda * SIGReg
    num_slices: int = 1024                     # SIGReg sketch width
    num_points: int = 17                       # Epps-Pulley quadrature knots
    # multi-crop view layout (set by the dataset; recorded for bookkeeping)
    num_global: int = 2
    num_local: int = 4


class SeldConformer(nn.Module):
    """Causal-Conformer encoder trained with LeJEPA (predictor + SIGReg).

    Training: `compute_loss(globals, locals)`.
    Inference: `forward_features(feat)` -> (B, T_enc, D); `embed(feat)` -> (B, D).
    """

    def __init__(self, config: Optional[SeldConformerConfig] = None):
        super().__init__()
        cfg = config or SeldConformerConfig()
        self.config = cfg
        self.encoder = ConformerEncoder(
            in_chans=cfg.in_chans, n_mels=cfg.n_mels, d_model=cfg.encoder_dim,
            n_layers=cfg.num_layers, n_heads=cfg.num_heads,
            ffn_expansion=cfg.ffn_expansion, conv_kernel=cfg.conv_kernel,
            chunk_frames=cfg.chunk_frames, dropout=cfg.dropout,
            frontend_hidden=cfg.frontend_hidden,
        )
        d = cfg.encoder_dim
        self.predictor = _mlp([d, cfg.pred_hidden, d])
        self.sigreg = SIGReg(cfg.num_slices, cfg.num_points)

    @property
    def out_dim(self) -> int:
        return self.config.encoder_dim

    # -------------------------------------------------------------------------
    # Encoding
    # -------------------------------------------------------------------------

    def forward_features(self, feat: torch.Tensor) -> torch.Tensor:
        """feat (B, C_in, T_f, M) -> per-frame encoder states (B, T_enc, D)."""
        return self.encoder(feat)

    def _pool(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T_enc, D) -> (B, D). 'last' = streaming-valid causal state."""
        if self.config.pool == "mean":
            return frames.mean(dim=1)
        return frames[:, -1]                                     # last causal state

    def embed(self, feat: torch.Tensor) -> torch.Tensor:
        """Pooled embedding (B, D) for one view / clip."""
        return self._pool(self.forward_features(feat))

    def _encode_group(self, views: torch.Tensor) -> torch.Tensor:
        """views (B, V, C, T, M) -> pooled embeddings (B, V, D)."""
        b, v = views.shape[0], views.shape[1]
        z = self.embed(views.reshape(b * v, *views.shape[2:]))   # (B*V, D)
        return z.view(b, v, -1)

    # -------------------------------------------------------------------------
    # LeJEPA training objective (mirrors JEPA/LeJEPA/LeJEPA.py)
    # -------------------------------------------------------------------------

    def compute_loss(self, globals: torch.Tensor,
                     locals: Optional[torch.Tensor] = None) -> dict:
        """One step. `globals` (B, ng, C, Tg, M); `locals` (B, nl, C, Tl, M)
        optional. Global-view embeddings are the prediction targets; SIGReg is
        applied to all view embeddings pooled together. No EMA / stop-gradient."""
        zg = self._encode_group(globals)                         # (B, ng, D)
        ng = zg.shape[1]
        z_src = [zg]
        if locals is not None and locals.numel() > 0:
            z_src.append(self._encode_group(locals))             # (B, nl, D)
        z_all = torch.cat(z_src, dim=1)                          # (B, V, D)
        V = z_all.shape[1]

        # JEPA prediction: every source view predicts every global target (skip self).
        p_all = self.predictor(z_all)                            # (B, V, D)
        pred_terms, n_pairs = 0.0, 0
        for s in range(V):
            for g in range(ng):
                if s == g:
                    continue
                pred_terms = pred_terms + F.mse_loss(p_all[:, s], zg[:, g])
                n_pairs += 1
        pred = pred_terms / max(n_pairs, 1)

        reg = self.sigreg(z_all.reshape(-1, z_all.shape[-1]))     # embeddings -> N(0, I)
        loss = pred + self.config.sigreg_coeff * reg
        return {"loss": loss, "pred_loss": pred.detach(), "reg_loss": reg.detach()}

    # -------------------------------------------------------------------------
    # Checkpoint loading (abort if nothing matched; freeze + eval)
    # -------------------------------------------------------------------------

    _STRIP_PREFIXES = ("module.", "_orig_mod.", "model.", "le.", "seld_conformer.",
                       "encoder_model.")
    _KEEP_PREFIXES = ("encoder.", "predictor.")

    @classmethod
    def _unwrap(cls, obj) -> dict:
        if isinstance(obj, dict):
            for key in ("state_dict", "model_state_dict", "model", "weights"):
                if key in obj and isinstance(obj[key], dict):
                    return cls._unwrap(obj[key])
            if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj
        raise ValueError(f"Could not find a tensor state-dict in checkpoint ({type(obj)})")

    @classmethod
    def _filter(cls, sd: dict) -> dict:
        out: dict = {}
        for raw_k, v in sd.items():
            k, changed = raw_k, True
            while changed:
                changed = False
                for p in cls._STRIP_PREFIXES:
                    if k.startswith(p):
                        k, changed = k[len(p):], True
            if any(k.startswith(p) for p in cls._KEEP_PREFIXES):
                out[k] = v
        return out

    @classmethod
    def from_checkpoint(cls, path: str, config: Optional[SeldConformerConfig] = None,
                        map_location: str = "cpu", device=None,
                        freeze: bool = True) -> "SeldConformer":
        """Build a `SeldConformer` and load weights. Aborts (per CLAUDE.md) if not
        a single tensor loads. With `freeze=True` sets `eval()` + frozen params."""
        model = cls(config)
        try:
            blob = torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            blob = torch.load(path, map_location=map_location, weights_only=False)
        sd = cls._filter(cls._unwrap(blob))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        applied = len(sd) - len(unexpected)
        if applied == 0:
            raise RuntimeError(
                f"SeldConformer.from_checkpoint({path!r}): 0/{len(sd)} keys applied "
                "— checkpoint/architecture mismatch, aborting.")
        logger.info("SeldConformer loaded %d/%d keys (missing=%d, unexpected=%d)",
                    applied, len(sd), len(missing), len(unexpected))
        if device is not None:
            model = model.to(device)
        if freeze:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
        return model
