"""Standalone LeJEPA module (PyTorch 2.8).

Self-contained reimplementation of **LeJEPA** (Balestriero & LeCun,
arXiv:2511.08544, ref impl: github.com/rbalestr-lab/lejepa). LeJEPA proves
that the **isotropic Gaussian** is the optimal distribution for a JEPA's
embeddings (it minimises downstream prediction risk), and reaches it with a
single, heuristics-free regulariser — **SIGReg** (Sketched Isotropic
Gaussian Regularization). There is **no EMA teacher, no stop-gradient, no
schedulers, no predictor-collapse tricks**; the whole objective has **one**
trade-off hyper-parameter.

    z_v   = Enc(view_v)                           # ViT embedding per view
    p_v   = Pred(z_v)                             # JEPA predictor
    L_pred= mean_{s, g!=s} || Pred(z_s) - z_g ||^2  # predict global embeddings
    L_reg = SIGReg({ z_v })                        # embeddings -> N(0, I)
    L     = L_pred + lambda * L_reg                # single hyper-parameter

**SIGReg** projects the embeddings onto `num_slices` random unit directions
(the "sketch") and, for each 1-D projection, measures how far its
distribution is from N(0,1) with the **Epps-Pulley** empirical-characteristic
-function goodness-of-fit statistic, evaluated by `num_points`-node
Gauss-Hermite quadrature:

    phi_n(t) = (1/n) sum_k exp(i t y_k)            # empirical char. function
    T_slice  = integral |phi_n(t) - e^{-t^2/2}|^2 e^{-t^2/2} dt
             ~ sum_m w_m [ (Re phi_n(t_m) - e^{-t_m^2/2})^2 + Im phi_n(t_m)^2 ]
    SIGReg   = mean_slice T_slice

`T >= 0`, and is 0 iff every 1-D projection matches N(0,1) (i.e. embeddings
are isotropic Gaussian). Differentiable, linear in `n` and dimension, CPU
-friendly. Defaults follow the paper: `num_slices=1024`, `num_points=17`.
The encoder is architecture-agnostic in the paper (ResNets / ViTs / ConvNets);
here we use a compact ViT so multi-crop views of different resolutions share
one network and the linear probe can read the CLS token of the last layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("LeJEPA")


# =============================================================================
# Compact ViT encoder (variable input size via pos-embed interpolation)
# =============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)                                  # (B, D, gh, gw)
        gh, gw = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)                  # (B, gh*gw, D)
        return x, (gh, gw)


class Block(nn.Module):
    """Pre-norm transformer block (MHSA + MLP)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """CLS-token ViT. `forward` returns the final-layer normed CLS embedding;
    `forward(..., return_hidden=True)` also returns the per-block CLS history
    so the linear probe can concatenate the last layers (LeJEPA protocol)."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int,
                 embed_dim: int, depth: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        base_grid = img_size // patch_size
        self.base_grid = base_grid
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + base_grid * base_grid, embed_dim))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _interp_pos(self, grid: tuple[int, int]) -> torch.Tensor:
        """Interpolate the patch positional embedding to grid (gh, gw)."""
        gh, gw = grid
        if gh == self.base_grid and gw == self.base_grid:
            return self.pos_embed
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        b = self.base_grid
        patch_pos = patch_pos.reshape(1, b, b, self.embed_dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(gh, gw), mode="bicubic",
                                  align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gh * gw, self.embed_dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        B = x.shape[0]
        tokens, grid = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self._interp_pos(grid)
        cls_hist = []
        for blk in self.blocks:
            x = blk(x)
            cls_hist.append(x[:, 0])                      # CLS token per block
        z = self.norm(cls_hist[-1])
        if return_hidden:
            return z, cls_hist
        return z


# =============================================================================
# SIGReg — Sketched Isotropic Gaussian Regularization (Epps-Pulley + slicing)
# =============================================================================

class SIGReg(nn.Module):
    """Penalises the deviation of embeddings from an isotropic Gaussian.

    Sketch the embeddings onto `num_slices` random unit directions, then for
    each 1-D projection measure the Epps-Pulley empirical-characteristic
    -function distance to N(0,1) via `num_points`-node Gauss-Hermite
    quadrature of  integral |phi_n(t) - e^{-t^2/2}|^2 e^{-t^2/2} dt.
    """

    def __init__(self, num_slices: int = 1024, num_points: int = 17):
        super().__init__()
        self.num_slices = num_slices
        self.num_points = num_points
        # Gauss-Hermite (weight e^{-x^2}); fold the Epps-Pulley weight
        # e^{-t^2/2} in via t = sqrt(2) x:  int g(t) e^{-t^2/2} dt
        #   = sqrt(2) sum_m W_m g(sqrt(2) x_m).
        nodes, weights = np.polynomial.hermite.hermgauss(num_points)
        self.register_buffer("t",
                             torch.tensor(np.sqrt(2.0) * nodes, dtype=torch.float32))
        self.register_buffer("w",
                             torch.tensor(np.sqrt(2.0) * weights, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        # Fresh random sketch each call (unbiased "sketched" estimator).
        v = torch.randn(d, self.num_slices, device=z.device, dtype=z.dtype)
        v = v / v.norm(dim=0, keepdim=True).clamp_min(1e-12)   # unit directions
        y = z @ v                                              # (n, S)
        t = self.t.to(z.dtype)                                 # (P,)
        ty = y.unsqueeze(-1) * t.view(1, 1, -1)                # (n, S, P)
        re = torch.cos(ty).mean(dim=0)                         # (S, P) Re phi_n
        im = torch.sin(ty).mean(dim=0)                         # (S, P) Im phi_n
        phi0 = torch.exp(-0.5 * t * t).view(1, -1)             # (1, P) N(0,1) CF
        integrand = (re - phi0) ** 2 + im ** 2                 # |phi_n - phi_0|^2
        per_slice = (integrand * self.w.to(z.dtype).view(1, -1)).sum(dim=-1)
        return per_slice.mean()


# =============================================================================
# MLP predictor
# =============================================================================

def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.GELU()]
    return nn.Sequential(*layers)


# =============================================================================
# Config
# =============================================================================

@dataclass
class LeJEPAConfig:
    in_chans: int = 3
    img_size: int = 224                       # base grid for pos-embed
    patch_size: int = 16
    embed_dim: int = 384                       # ViT-S
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    pred_hidden: int = 1024                    # predictor MLP hidden width

    # multi-crop view layout (DINO-style): predict global embeddings
    num_global: int = 2
    num_local: int = 6

    # SIGReg — the embeddings -> isotropic Gaussian regulariser
    sigreg_coeff: float = 1.0                  # the single trade-off hyper-param
    num_slices: int = 1024
    num_points: int = 17
    probe_last_layers: int = 2                 # CLS tokens concatenated at probe


# =============================================================================
# Model
# =============================================================================

class LeJEPA(nn.Module):
    """LeJEPA image SSL: ViT encoder + JEPA predictor, trained with the
    predictive loss + SIGReg (no EMA / stop-gradient / scheduler).

    Inference: `encode(x)` returns the frozen probe feature — the LayerNorm'd
    CLS tokens of the last `probe_last_layers` blocks, concatenated.
    """

    def __init__(self, config: Optional[LeJEPAConfig] = None):
        super().__init__()
        cfg = config or LeJEPAConfig()
        self.config = cfg
        self.encoder = ViTEncoder(cfg.img_size, cfg.patch_size, cfg.in_chans,
                                  cfg.embed_dim, cfg.depth, cfg.num_heads,
                                  cfg.mlp_ratio)
        d = self.encoder.out_dim
        self.predictor = _mlp([d, cfg.pred_hidden, d])
        self.sigreg = SIGReg(cfg.num_slices, cfg.num_points)

    # -------------------------------------------------------------------------
    # Training objective
    # -------------------------------------------------------------------------

    def _encode_group(self, views: torch.Tensor) -> torch.Tensor:
        """views (B, V, C, H, W) -> embeddings (B, V, D)."""
        b, v = views.shape[0], views.shape[1]
        z = self.encoder(views.flatten(0, 1))               # (B*V, D)
        return z.view(b, v, -1)

    def compute_loss(self, globals: torch.Tensor,
                     locals: Optional[torch.Tensor] = None) -> dict:
        """One step. `globals` (B, ng, C, Hg, Wg); `locals` (B, nl, C, Hl, Wl)
        optional. Global embeddings are the prediction targets; SIGReg is
        applied to all view embeddings pooled together."""
        zg = self._encode_group(globals)                    # (B, ng, D)
        ng = zg.shape[1]
        z_src = [zg]
        if locals is not None and locals.numel() > 0:
            z_src.append(self._encode_group(locals))        # (B, nl, D)
        z_all = torch.cat(z_src, dim=1)                     # (B, V, D)
        V = z_all.shape[1]

        # JEPA prediction: every source view predicts every global target
        # (skip the self-pair). Energy-based — no stop-gradient on targets.
        p_all = self.predictor(z_all)                       # (B, V, D)
        pred_terms, n_pairs = 0.0, 0
        for s in range(V):
            for g in range(ng):
                if s == g:                                  # skip self-pair
                    continue
                pred_terms = pred_terms + F.mse_loss(p_all[:, s], zg[:, g])
                n_pairs += 1
        pred = pred_terms / max(n_pairs, 1)

        # SIGReg over the pooled embeddings of all views in the batch.
        reg = self.sigreg(z_all.reshape(-1, z_all.shape[-1]))

        loss = pred + self.config.sigreg_coeff * reg
        return {"loss": loss, "pred_loss": pred.detach(),
                "reg_loss": reg.detach()}

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Probe feature: concat of LayerNorm'd CLS tokens from the last
        `probe_last_layers` blocks (LeJEPA linear-probe protocol)."""
        _, cls_hist = self.encoder(x, return_hidden=True)
        k = max(1, min(self.config.probe_last_layers, len(cls_hist)))
        feats = [self.encoder.norm(c) for c in cls_hist[-k:]]
        return torch.cat(feats, dim=-1)                     # (B, k*D)

    @torch.inference_mode()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Single final-layer normed CLS embedding (B, D) — the SSL target."""
        return self.encoder(x)

    # -------------------------------------------------------------------------
    # Checkpoint loading
    # -------------------------------------------------------------------------

    _STRIP_PREFIXES = ("module.", "_orig_mod.", "model.", "backbone.")
    _KEEP_PREFIXES = ("encoder.", "predictor.")

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
            if any(k.startswith(p) for p in cls._KEEP_PREFIXES):
                out[k] = v
        return out

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        config: Optional[LeJEPAConfig] = None,
        map_location: str = "cpu",
        device: Optional[torch.device] = None,
        strict: bool = False,
    ) -> "LeJEPA":
        """Build a `LeJEPA` and load weights. Aborts (per paper/CLAUDE.md) if
        not a single tensor loads."""
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
                f"LeJEPA.from_checkpoint({path!r}): 0/{total} keys applied "
                "— checkpoint/architecture mismatch, aborting.")
        if not missing and not unexpected:
            logger.info("LeJEPA loaded successfully (%d/%d keys)", applied, total)
        else:
            logger.warning(
                "LeJEPA partial load: %d/%d keys (missing=%d, unexpected=%d)",
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
