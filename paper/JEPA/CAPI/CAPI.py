"""Standalone CAPI module (PyTorch 2.8).

Self-contained reimplementation of **CAPI** (Darcet, Baldassarre, Oquab,
Mairal, Bojanowski; Meta FAIR, arXiv:2502.08769, ref impl:
github.com/facebookresearch/capi, Apache-2.0). CAPI is a *latent* masked
image model — the JEPA-family synthesis of I-JEPA (predict in latent space)
and iBOT/DINO (online-clustered targets). Instead of regressing teacher
features (I-JEPA) or pixels (MAE), CAPI predicts, for each masked patch, the
**cluster assignment** that an EMA teacher produces, turning collapse
prevention into a balanced clustering problem solved with Sinkhorn-Knopp.

    h_vis      = Enc(visible patches + registers)        # student, masked input
    z_pred     = Pred(mask-queries  x-attend  h_vis)     # cross-attn predictor
    t_full     = EMA-Enc(all patches)                    # teacher, full input
    target_i   = SK( prototypes . normalize(t_full[i]) ) # balanced soft target
    a_i        = softmax( prototypes . normalize(z_pred[i]) / tau )
    L          = - mean_i  sum_k  target_i(k) log a_i(k) # cross-entropy

Key properties vs. the other JEPAs in this repo: masked-latent prediction
(not augmentation views), **online clustering** targets (16k prototypes,
Sinkhorn-balanced), an **EMA teacher** (mu = 1 - lr), and a **cross-attention
predictor** (mask tokens attend to the encoder output; no self-attention).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("CAPI")


# =============================================================================
# Transformer blocks
# =============================================================================

class SelfBlock(nn.Module):
    """Pre-norm self-attention block (encoder)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class CrossBlock(nn.Module):
    """Pre-norm cross-attention block (predictor): queries attend to context;
    no self-attention (per CAPI)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, q: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        q = q + self.attn(self.norm_q(q), self.norm_ctx(ctx),
                          self.norm_ctx(ctx), need_weights=False)[0]
        q = q + self.mlp(self.norm2(q))
        return q


# =============================================================================
# ViT encoder with register tokens (visible-only or full forward)
# =============================================================================

class ViTEncoder(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int,
                 embed_dim: int, depth: int, num_heads: int, mlp_ratio: float,
                 num_registers: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must divide patch_size"
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size,
                                     stride=patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim))
        self.registers = nn.Parameter(torch.zeros(1, num_registers, embed_dim))
        self.blocks = nn.ModuleList(
            [SelfBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.num_registers = num_registers
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.registers, std=0.02)
        self.apply(_init_weights)

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        t = self.patch_embed(x).flatten(2).transpose(1, 2)    # (B, N, D)
        return t + self.pos_embed

    def _run(self, h: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)

    def forward_visible(self, x: torch.Tensor,
                        visible_idx: torch.Tensor) -> torch.Tensor:
        """Encode only the visible patches (+ registers). Returns the visible
        patch embeddings (B, K, D)."""
        tok = self._tokens(x)
        B, _, D = tok.shape
        idx = visible_idx.unsqueeze(-1).expand(-1, -1, D)
        vis = torch.gather(tok, 1, idx)                       # (B, K, D)
        reg = self.registers.expand(B, -1, -1)
        h = self._run(torch.cat([reg, vis], dim=1))
        return h[:, self.num_registers:]                      # drop registers

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """Encode all patches (+ registers). Returns all patch embeddings
        (B, N, D)."""
        tok = self._tokens(x)
        B = tok.shape[0]
        reg = self.registers.expand(B, -1, -1)
        h = self._run(torch.cat([reg, tok], dim=1))
        return h[:, self.num_registers:]


def _init_weights(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# =============================================================================
# Cross-attention predictor
# =============================================================================

class Predictor(nn.Module):
    """Mask queries (mask-token + masked-position embedding) cross-attend to
    the encoder's visible-patch output."""

    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(
            [CrossBlock(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def forward(self, ctx: torch.Tensor, masked_pos: torch.Tensor) -> torch.Tensor:
        """ctx (B, K, D) visible-patch embeddings; masked_pos (B, M, D) the
        positional embeddings of the masked patches. Returns (B, M, D)."""
        q = self.mask_token + masked_pos                      # (B, M, D)
        for blk in self.blocks:
            q = blk(q, ctx)
        return self.norm(q)


# =============================================================================
# Sinkhorn-Knopp balanced assignment (no grad)
# =============================================================================

@torch.no_grad()
def sinkhorn_knopp(scores: torch.Tensor, eps: float, n_iters: int) -> torch.Tensor:
    """scores (n, k) prototype logits -> balanced soft assignments (n, k),
    rows sum to 1. Standard SwAV/DINO Sinkhorn over the batch."""
    q = torch.exp(scores / eps).t()                           # (k, n)
    q = q / q.sum().clamp_min(1e-12)
    k, n = q.shape
    for _ in range(n_iters):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12) / k   # rows (clusters)
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12) / n   # cols (samples)
    q = q * n                                                 # cols sum to 1
    return q.t()                                              # (n, k)


# =============================================================================
# Config
# =============================================================================

@dataclass
class CAPIConfig:
    in_chans: int = 3
    img_size: int = 224
    patch_size: int = 14
    embed_dim: int = 384                       # ViT-S default (paper: ViT-L)
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    num_registers: int = 16

    pred_depth: int = 12                        # cross-attention predictor
    pred_heads: int = 6

    num_prototypes: int = 16384                 # online clustering targets
    mask_ratio: float = 0.65
    sinkhorn_eps: float = 0.05                  # SK regularisation (teacher)
    sinkhorn_iters: int = 3
    student_temp: float = 0.12                  # student softmax temperature
    ema_momentum: float = 0.999                 # teacher EMA (paper: 1 - lr)


# =============================================================================
# Model
# =============================================================================

class CAPI(nn.Module):
    """CAPI latent masked image model: student ViT encoder + cross-attention
    predictor + learnable prototypes, with an EMA teacher producing
    Sinkhorn-balanced cluster targets (no pixel decoder, no contrastive pairs).

    Inference: `encode(x)` runs the (student) encoder over the full image and
    returns mean-pooled patch features — the linear-probe target.
    """

    def __init__(self, config: Optional[CAPIConfig] = None):
        super().__init__()
        cfg = config or CAPIConfig()
        self.config = cfg
        self.encoder = ViTEncoder(cfg.img_size, cfg.patch_size, cfg.in_chans,
                                  cfg.embed_dim, cfg.depth, cfg.num_heads,
                                  cfg.mlp_ratio, cfg.num_registers)
        self.predictor = Predictor(cfg.embed_dim, cfg.pred_depth,
                                   cfg.pred_heads, cfg.mlp_ratio)
        self.prototypes = nn.Parameter(
            torch.randn(cfg.num_prototypes, cfg.embed_dim) * 0.02)
        # EMA teacher: frozen deep copy of the encoder.
        self.teacher = copy.deepcopy(self.encoder)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    # -------------------------------------------------------------------------
    # EMA teacher
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def update_teacher(self, momentum: Optional[float] = None):
        m = self.config.ema_momentum if momentum is None else momentum
        for ps, pt in zip(self.encoder.parameters(), self.teacher.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)

    # -------------------------------------------------------------------------
    # Training objective
    # -------------------------------------------------------------------------

    def _proto_logits(self, z: torch.Tensor) -> torch.Tensor:
        """L2-normalise embeddings and prototypes, return cosine logits."""
        z = F.normalize(z, dim=-1)
        c = F.normalize(self.prototypes, dim=-1)
        return z @ c.t()                                      # (..., K)

    def compute_loss(self, image: torch.Tensor, visible_idx: torch.Tensor,
                     masked_idx: torch.Tensor) -> dict:
        """One step. `image` (B, C, H, W); `visible_idx` (B, K) / `masked_idx`
        (B, M) index patches (row-major grid). Predicts the teacher's cluster
        assignment of each masked patch."""
        cfg = self.config
        B, M = masked_idx.shape
        D = cfg.embed_dim

        # Student: encode visible patches, predict masked-patch latents.
        h_vis = self.encoder.forward_visible(image, visible_idx)    # (B, K, D)
        pos = self.encoder.pos_embed.expand(B, -1, -1)
        masked_pos = torch.gather(pos, 1,
                                  masked_idx.unsqueeze(-1).expand(-1, -1, D))
        z_pred = self.predictor(h_vis, masked_pos)                  # (B, M, D)

        # Teacher: full-image latents at the masked positions -> SK targets.
        with torch.no_grad():
            t_full = self.teacher.forward_full(image)               # (B, N, D)
            t_masked = torch.gather(
                t_full, 1, masked_idx.unsqueeze(-1).expand(-1, -1, D))
            t_logits = self._proto_logits(t_masked).reshape(B * M, -1)
            target = sinkhorn_knopp(t_logits, cfg.sinkhorn_eps,
                                    cfg.sinkhorn_iters)             # (B*M, K)

        s_logits = self._proto_logits(z_pred).reshape(B * M, -1)
        log_p = F.log_softmax(s_logits / cfg.student_temp, dim=-1)
        loss = -(target * log_p).sum(dim=-1).mean()

        # Diagnostics: target cluster entropy / usage (collapse monitor).
        with torch.no_grad():
            usage = target.mean(dim=0)                              # (K,)
            ent = -(usage * (usage + 1e-12).log()).sum()
        return {"loss": loss, "ce_loss": loss.detach(),
                "target_entropy": ent}

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Probe feature: mean-pooled student patch embeddings (B, D)."""
        return self.encoder.forward_full(x).mean(dim=1)

    @torch.inference_mode()
    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Dense student patch embeddings (B, N, D)."""
        return self.encoder.forward_full(x)

    # -------------------------------------------------------------------------
    # Checkpoint loading
    # -------------------------------------------------------------------------

    _STRIP_PREFIXES = ("module.", "_orig_mod.", "model.")
    _KEEP_PREFIXES = ("encoder.", "predictor.", "teacher.", "prototypes")

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
        config: Optional[CAPIConfig] = None,
        map_location: str = "cpu",
        device: Optional[torch.device] = None,
        strict: bool = False,
    ) -> "CAPI":
        """Build a `CAPI` and load weights. Aborts (per paper/CLAUDE.md) if not
        a single tensor loads."""
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
                f"CAPI.from_checkpoint({path!r}): 0/{total} keys applied "
                "— checkpoint/architecture mismatch, aborting.")
        if not missing and not unexpected:
            logger.info("CAPI loaded successfully (%d/%d keys)", applied, total)
        else:
            logger.warning(
                "CAPI partial load: %d/%d keys (missing=%d, unexpected=%d)",
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
