"""Standalone LeWorldModel (LeWM) module (PyTorch 2.8).

Self-contained reimplementation of LeWorldModel (arXiv:2603.19312, ref
impl: github.com/lucas-maes/le-wm, MIT). LeWM is the first Joint-Embedding
Predictive Architecture that trains *stably end-to-end from raw pixels*
using only TWO loss terms and NO EMA teacher / stop-gradient / pretrained
encoder:

    z_t        = Enc(o_t)                       (ViT frame encoder)
    z_hat_{t+1}= Pred(z_{<=t}, a_{<=t})         (causal AR predictor)
    L_pred     = || z_hat_{t+1} - z_{t+1} ||^2  (BOTH sides carry gradient
                                                 — end-to-end, no stop-grad)
    L_reg      = SIGReg({z})                     (Sketched Isotropic Gaussian
                                                 Regularization, LeJEPA-style:
                                                 random 1-D projections tested
                                                 for N(0,1) via Epps-Pulley)
    L          = L_pred + lambda * L_reg

`L_reg` is what prevents representation collapse, replacing the EMA/
stop-gradient machinery used by I-/V-/Point-JEPA. This cuts the tunable
loss hyperparameters from six to one (`lambda`).

Reference: LeJEPA (Balestriero & LeCun 2025) introduced SIGReg; LeWM
applies it to action-conditioned world models.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("LeWorldModel")

SQRT_2PI = math.sqrt(2.0 * math.pi)
SQRT_PI = math.sqrt(math.pi)


# =============================================================================
# ViT frame encoder
# =============================================================================

class PatchEmbed2D(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int,
                 embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must divide patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)        # (B, N, D)


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, eps: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.norm1(x)
        x = x + self.attn(y, y, y, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """ViT that maps an image to a single frame embedding (mean-pooled
    tokens -> projector). No [CLS] token."""

    def __init__(self, cfg: "LeWMConfig"):
        super().__init__()
        self.patch_embed = PatchEmbed2D(cfg.img_size, cfg.patch_size,
                                        cfg.in_chans, cfg.embed_dim)
        n = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            _Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio,
                   cfg.layer_norm_eps)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        self.proj = nn.Linear(cfg.embed_dim, cfg.latent_dim)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(imgs) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(dim=1)                          # (B, D) mean-pool
        return self.proj(x)                                   # (B, latent_dim)


# =============================================================================
# Action embedder + autoregressive predictor
# =============================================================================

class ActionEmbedder(nn.Module):
    def __init__(self, action_dim: int, latent_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, latent_dim))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.mlp(a)


class ARPredictor(nn.Module):
    """Causal Transformer predicting z_{t+1} from (z_<=t, a_<=t).

    Tokens fed are (z_t + a_emb_t); a causal mask makes prediction at
    position t depend only on <= t. Output at t is the predicted z_{t+1}.
    """

    def __init__(self, cfg: "LeWMConfig"):
        super().__init__()
        d = cfg.latent_dim
        self.time_pos = nn.Parameter(torch.zeros(1, cfg.max_seq_len, d))
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        self.blocks = nn.ModuleList([
            _Block(d, cfg.pred_num_heads, cfg.mlp_ratio, cfg.layer_norm_eps)
            for _ in range(cfg.pred_depth)
        ])
        self.norm = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        self.head = nn.Linear(d, d)

    def forward(self, z: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        """z, a_emb: (B, T, D) -> predicted next-step embeddings (B, T, D)."""
        B, T, D = z.shape
        x = z + a_emb + self.time_pos[:, :T, :]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=z.device),
                          diagonal=1)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        return self.head(self.norm(x))                        # (B, T, D)


# =============================================================================
# SIGReg — Sketched Isotropic Gaussian Regularization (Epps-Pulley)
# =============================================================================

def epps_pulley_statistic(y: torch.Tensor) -> torch.Tensor:
    """Closed-form Epps-Pulley goodness-of-fit of 1-D samples `y` (n,) to
    N(0, 1), using the empirical characteristic function with Gaussian
    weight exp(-t^2/2):

        T = sqrt(2pi)/n^2 * sum_jk exp(-(y_j-y_k)^2/2)
            - 2*sqrt(pi)/n * sum_k exp(-y_k^2/4)
            + sqrt(pi)

    T >= 0, and T = 0 iff the empirical CF matches N(0,1)'s. Differentiable.
    """
    n = y.shape[0]
    diff = y.unsqueeze(0) - y.unsqueeze(1)                    # (n, n)
    term1 = SQRT_2PI * torch.exp(-0.5 * diff ** 2).mean()     # mean = sum/n^2
    term2 = 2.0 * SQRT_PI * torch.exp(-0.25 * y ** 2).mean()
    return term1 - term2 + SQRT_PI


class SIGReg(nn.Module):
    """Random-projection ("sketched") isotropic-Gaussian regularizer.

    Sample `num_proj` random unit directions, project the embeddings onto
    each (after centering+standardizing per-dim so the test targets the
    *isotropic standard* Gaussian), and average the Epps-Pulley statistic.
    Minimizing this drives the embedding distribution toward an isotropic
    Gaussian, preventing collapse without EMA/stop-gradient.
    """

    def __init__(self, num_proj: int = 64, eps: float = 1e-5):
        super().__init__()
        self.num_proj = num_proj
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., D) -> scalar regularization loss."""
        z = z.reshape(-1, z.shape[-1])                        # (N, D)
        z = (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + self.eps)
        dirs = F.normalize(
            torch.randn(z.shape[-1], self.num_proj, device=z.device,
                        dtype=z.dtype), dim=0)                # (D, P)
        proj = z @ dirs                                       # (N, P)
        stats = [epps_pulley_statistic(proj[:, p])
                 for p in range(self.num_proj)]
        return torch.stack(stats).mean()


# =============================================================================
# Config
# =============================================================================

@dataclass
class LeWMConfig:
    # observation / action
    img_size: int = 64
    patch_size: int = 8
    in_chans: int = 3
    action_dim: int = 2
    max_seq_len: int = 16

    # encoder ViT
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-6

    # latent + predictor
    latent_dim: int = 384
    pred_depth: int = 4
    pred_num_heads: int = 6

    # objective (single tunable loss hyperparameter: reg_weight)
    reg_weight: float = 1.0
    num_proj: int = 64


# =============================================================================
# Model
# =============================================================================

class LeWorldModel(nn.Module):
    """LeWM world model: ViT encoder + action embedder + AR predictor,
    trained with prediction loss + SIGReg (no EMA / stop-grad).

    Inference: `encode(obs)` for frame embeddings, `rollout(obs0, actions)`
    to imagine future latents. Training: `compute_loss(obs, actions)`.
    """

    def __init__(self, config: Optional[LeWMConfig] = None):
        super().__init__()
        cfg = config or LeWMConfig()
        self.config = cfg
        self.encoder = ViTEncoder(cfg)
        self.action_embedder = ActionEmbedder(cfg.action_dim, cfg.latent_dim)
        self.predictor = ARPredictor(cfg)
        self.sigreg = SIGReg(cfg.num_proj)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # -------------------------------------------------------------------------
    def _encode_seq(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, T, C, H, W) -> (B, T, latent_dim)."""
        B, T = obs.shape[:2]
        z = self.encoder(obs.reshape(B * T, *obs.shape[2:]))
        return z.reshape(B, T, -1)

    def compute_loss(self, obs: torch.Tensor,
                     actions: torch.Tensor) -> dict:
        """One training step.

        obs    : (B, T, C, H, W) float in [0, 1]
        actions: (B, T, action_dim)
        """
        z = self._encode_seq(obs)                             # (B, T, D), grad
        a_emb = self.action_embedder(actions)                 # (B, T, D)
        z_hat = self.predictor(z[:, :-1], a_emb[:, :-1])      # predict z_{1..T-1}
        # End-to-end: the target z[:, 1:] is NOT detached (collapse is held
        # off by SIGReg, not stop-gradient).
        pred_loss = F.mse_loss(z_hat, z[:, 1:])
        reg_loss = self.sigreg(z)
        loss = pred_loss + self.config.reg_weight * reg_loss
        return {"loss": loss, "pred_loss": pred_loss.detach(),
                "reg_loss": reg_loss.detach()}

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Frame embeddings. obs (B, T, C, H, W) or (B, C, H, W) -> latents."""
        if obs.ndim == 4:
            return self.encoder(obs)
        return self._encode_seq(obs)

    @torch.inference_mode()
    def rollout(self, obs0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Imagine future latents from the first frame and an action plan.

        obs0   : (B, C, H, W) first observation
        actions: (B, T, action_dim) planned actions
        Returns predicted latents (B, T, latent_dim).
        """
        B, T = actions.shape[:2]
        a_emb = self.action_embedder(actions)
        z = self.encoder(obs0).unsqueeze(1)                   # (B, 1, D)
        for t in range(T):
            z_hat = self.predictor(z, a_emb[:, :t + 1])       # (B, t+1, D)
            z = torch.cat([z, z_hat[:, -1:, :]], dim=1)
        return z[:, 1:, :]                                    # (B, T, D)

    # -------------------------------------------------------------------------
    # Checkpoint loading
    # -------------------------------------------------------------------------

    _STRIP_PREFIXES = ("module.", "_orig_mod.", "model.", "net.")
    _KEEP_PREFIXES = ("encoder.", "action_embedder.", "predictor.")

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
        config: Optional[LeWMConfig] = None,
        map_location: str = "cpu",
        device: Optional[torch.device] = None,
        strict: bool = False,
    ) -> "LeWorldModel":
        """Build a `LeWorldModel` and load weights from a checkpoint. Aborts
        (per paper/CLAUDE.md) if not a single tensor loads."""
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
                f"LeWorldModel.from_checkpoint({path!r}): 0/{total} keys "
                "applied — checkpoint/architecture mismatch, aborting.")
        if not missing and not unexpected:
            logger.info("LeWorldModel loaded successfully (%d/%d keys)",
                        applied, total)
        else:
            logger.warning(
                "LeWorldModel partial load: %d/%d keys (missing=%d, unexpected=%d)",
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
