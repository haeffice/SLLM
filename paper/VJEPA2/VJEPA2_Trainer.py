"""Standalone V-JEPA 2 trainer module (PyTorch 2.8).

`VJEPA2Trainer` is the JEPA self-supervised pre-training wrapper. It owns

    * the *student* path the inference model (`VJEPA2`) reuses —
      `self.patch_embed`, `self.blocks`, `self.norm`.
    * a *teacher* path — `self.teacher`, a frozen EMA copy of the full
      student encoder that produces the prediction targets over ALL tokens.
    * a *predictor* — `self.predictor`, a narrow Transformer that, given the
      context-token embeddings (at their grid positions) plus a learned
      mask token at every masked position, predicts the teacher features
      at the masked positions. Encoder<->predictor dim mappers included.

The class is self-contained: 3D multi-block masking, EMA scheduling/step,
teacher target construction (final-LayerNorm features) and the masked L1
prediction loss all live inside the module. The training loop only needs to
call `forward(video)` (masks are sampled internally) and `ema_step(step)`
after each optimizer step.

References:
    * I-JEPA  (Assran et al. 2023, arXiv:2301.08243)
    * V-JEPA  (Bardes et al. 2024, arXiv:2404.08471)
    * V-JEPA 2 (Assran et al. 2025, arXiv:2506.09985,
                github.com/facebookresearch/vjepa2)

The student-side state-dict keys match `VJEPA2` exactly, so a checkpoint
saved here loads with `VJEPA2.from_checkpoint(...)`.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from VJEPA2 import VARIANTS, Block, PatchEmbed3D, RoPE3D, VJEPA2Config


logger = logging.getLogger("VJEPA2_Trainer")


# =============================================================================
# Trainer config
# =============================================================================

@dataclass
class VJEPA2TrainerConfig(VJEPA2Config):
    # predictor (narrow ViT)
    predictor_embed_dim: int = 384
    predictor_depth: int = 12
    predictor_num_heads: int = 12
    predictor_mlp_ratio: float = 4.0

    # EMA teacher schedule (linear tau anneal)
    ema_decay: float = 0.998
    ema_end_decay: float = 1.0
    ema_anneal_end_step: int = 100_000

    # 3D multi-block masking (V-JEPA 2 "multiblock-3d").
    # `mask_ratios` are *target* coverage fractions; one block group is
    # sampled per ratio and unioned. The complement is the context.
    mask_ratios: tuple[float, ...] = (0.15, 0.7)
    mask_blocks_per_ratio: tuple[int, ...] = (8, 2)
    min_context_ratio: float = 0.1


# =============================================================================
# 3D multi-block masker
# =============================================================================

class _MultiBlock3DMasker(nn.Module):
    """Sample spatio-temporal block masks on the (T', H', W') token grid.

    For every (ratio, n_blocks) pair we drop `n_blocks` random 3D blocks
    sized to roughly cover `ratio` of the grid; the union over all pairs is
    the target set, its complement (clamped to `>= min_context_ratio`) is
    the context. Returns flat token indices.

    Output (per batch item, lists since lengths differ across the batch is
    avoided by equalising counts — see `forward`):
        ctx_idx : (B, n_ctx)  Long  visible token indices
        tgt_idx : (B, n_tgt)  Long  masked (predicted) token indices
    """

    def __init__(self, cfg: VJEPA2TrainerConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = cfg.grid_size                       # (T', H', W')

    def _sample_block(self, ratio: float, gen: torch.Generator) -> torch.Tensor:
        T, H, W = self.grid
        # Cubic-ish block: side scales with ratio**(1/3) per axis.
        side = min(1.0, max(1e-3, ratio)) ** (1.0 / 3.0)
        bt = max(1, min(T, round(T * side)))
        bh = max(1, min(H, round(H * side)))
        bw = max(1, min(W, round(W * side)))
        t0 = int(torch.randint(0, T - bt + 1, (1,), generator=gen).item())
        h0 = int(torch.randint(0, H - bh + 1, (1,), generator=gen).item())
        w0 = int(torch.randint(0, W - bw + 1, (1,), generator=gen).item())
        m = torch.zeros(T, H, W, dtype=torch.bool)
        m[t0:t0 + bt, h0:h0 + bh, w0:w0 + bw] = True
        return m.flatten()

    def _sample_one(self, gen: torch.Generator) -> torch.Tensor:
        N = self.cfg.num_patches
        tgt = torch.zeros(N, dtype=torch.bool)
        for ratio, nb in zip(self.cfg.mask_ratios, self.cfg.mask_blocks_per_ratio):
            for _ in range(nb):
                tgt |= self._sample_block(ratio, gen)
        # Guarantee a minimum visible context.
        min_ctx = max(1, int(self.cfg.min_context_ratio * N))
        if (~tgt).sum().item() < min_ctx:
            free = torch.where(tgt)[0]
            keep = free[torch.randperm(len(free), generator=gen)[:min_ctx]]
            tgt[keep] = False
        # Guarantee at least one target.
        if tgt.sum().item() == 0:
            tgt[torch.randint(0, N, (1,), generator=gen).item()] = True
        return tgt

    @torch.no_grad()
    def forward(self, batch_size: int,
                device: Optional[torch.device] = None,
                seed: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        masks = [self._sample_one(gen) for _ in range(batch_size)]
        # Equalise counts across the batch so we can stack into tensors.
        n_tgt = min(int(m.sum()) for m in masks)
        n_ctx = min(int((~m).sum()) for m in masks)
        ctx_idx, tgt_idx = [], []
        for m in masks:
            t = torch.where(m)[0]
            c = torch.where(~m)[0]
            t = t[torch.randperm(len(t), generator=gen)[:n_tgt]].sort().values
            c = c[torch.randperm(len(c), generator=gen)[:n_ctx]].sort().values
            tgt_idx.append(t)
            ctx_idx.append(c)
        ctx = torch.stack(ctx_idx)
        tgt = torch.stack(tgt_idx)
        if device is not None:
            ctx, tgt = ctx.to(device), tgt.to(device)
        return ctx, tgt


# =============================================================================
# Predictor (narrow ViT over context + mask tokens)
# =============================================================================

class _Predictor(nn.Module):
    def __init__(self, cfg: VJEPA2TrainerConfig):
        super().__init__()
        d = cfg.predictor_embed_dim
        self.embed = nn.Linear(cfg.embed_dim, d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        rope = RoPE3D(d // cfg.predictor_num_heads, cfg.grid_size)
        self.blocks = nn.ModuleList([
            Block(d, cfg.predictor_num_heads, cfg.predictor_mlp_ratio,
                  True, cfg.layer_norm_eps, rope=rope)
            for _ in range(cfg.predictor_depth)
        ])
        self.norm = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        self.out = nn.Linear(d, cfg.embed_dim)

    def forward(self, ctx_tokens: torch.Tensor, ctx_idx: torch.Tensor,
                tgt_idx: torch.Tensor, num_patches: int) -> torch.Tensor:
        """ctx_tokens: (B, n_ctx, D_enc) student features at `ctx_idx`.
        Returns predicted encoder-dim features at `tgt_idx`: (B, n_tgt, D_enc)."""
        B = ctx_tokens.shape[0]
        d = self.mask_token.shape[-1]
        # Scatter context + mask tokens back onto the full grid so RoPE
        # (which is position-indexed) sees correct token positions.
        full = self.mask_token.expand(B, num_patches, d).clone()
        full = full.scatter(
            1, ctx_idx.unsqueeze(-1).expand(-1, -1, d), self.embed(ctx_tokens)
        )
        for blk in self.blocks:
            full = blk(full)
        full = self.norm(full)
        pred = full.gather(1, tgt_idx.unsqueeze(-1).expand(-1, -1, d))
        return self.out(pred)                            # (B, n_tgt, D_enc)


# =============================================================================
# Trainer model
# =============================================================================

class VJEPA2Trainer(nn.Module):
    """JEPA trainer: student encoder + EMA teacher + predictor.

    `self.patch_embed` / `self.blocks` / `self.norm` are the deployable
    encoder (student). After training, save `self.state_dict()` and load
    it with `VJEPA2.from_checkpoint(...)` (it keeps only `patch_embed.*`,
    `blocks.*`, `norm.*`).
    """

    def __init__(self, *, variant: str = "vit_large", **overrides):
        super().__init__()
        depth, dim, heads, mlp = VARIANTS[variant]
        base = dict(variant=variant, embed_dim=dim, depth=depth,
                    num_heads=heads, mlp_ratio=mlp)
        base.update(overrides)
        self.config = VJEPA2TrainerConfig(**base)
        cfg = self.config

        # ---- student (shared with VJEPA2) ----
        self.patch_embed = PatchEmbed3D(
            patch_size=cfg.patch_size, tubelet_size=cfg.tubelet_size,
            in_chans=cfg.in_chans, embed_dim=cfg.embed_dim,
        )
        enc_rope = RoPE3D(cfg.embed_dim // cfg.num_heads, cfg.grid_size)
        self.blocks = nn.ModuleList([
            Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio,
                  cfg.qkv_bias, cfg.layer_norm_eps, rope=enc_rope)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)

        # ---- predictor ----
        self.predictor = _Predictor(cfg)

        # ---- pixel normalisation buffers (match VJEPA2) ----
        from VJEPA2 import IMAGENET_MEAN, IMAGENET_STD
        self.register_buffer(
            "_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1),
            persistent=False)
        self.register_buffer(
            "_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1),
            persistent=False)

        self.masker = _MultiBlock3DMasker(cfg)
        self.apply(self._init_weights)
        self._init_teacher()

    # -------------------------------------------------------------------------
    # Init
    # -------------------------------------------------------------------------

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _student_modules(self) -> nn.ModuleList:
        return nn.ModuleList([self.patch_embed, *self.blocks, self.norm])

    def _init_teacher(self):
        # Frozen EMA copy of the full student encoder (patch_embed+blocks+norm).
        self.teacher = copy.deepcopy(self._student_modules())
        self.teacher.requires_grad_(False)

    # -------------------------------------------------------------------------
    # EMA
    # -------------------------------------------------------------------------

    def _ema_decay(self, step: int) -> float:
        cfg = self.config
        if step >= cfg.ema_anneal_end_step:
            return cfg.ema_end_decay
        r = cfg.ema_end_decay - cfg.ema_decay
        pct_remaining = 1.0 - (step / cfg.ema_anneal_end_step)
        return cfg.ema_end_decay - r * pct_remaining

    @torch.no_grad()
    def ema_step(self, step: int) -> float:
        r = self._ema_decay(step)
        student = list(self.patch_embed.parameters()) \
            + [p for b in self.blocks for p in b.parameters()] \
            + list(self.norm.parameters())
        for sp, tp in zip(student, self.teacher.parameters()):
            tp.data.mul_(r).add_(sp.detach().data, alpha=(1.0 - r))
        return r

    # -------------------------------------------------------------------------
    # Forward paths
    # -------------------------------------------------------------------------

    def _normalize(self, video: torch.Tensor) -> torch.Tensor:
        return (video - self._mean) / self._std

    def _encode_context(self, tokens: torch.Tensor,
                        ctx_idx: torch.Tensor) -> torch.Tensor:
        """Student encoder over *visible* context tokens only."""
        d = tokens.shape[-1]
        x = tokens.gather(1, ctx_idx.unsqueeze(-1).expand(-1, -1, d))
        for blk in self.blocks:
            x = blk(x, ctx_idx)                           # RoPE by true pos
        return self.norm(x)                              # (B, n_ctx, D)

    @torch.no_grad()
    def _teacher_targets(self, tokens: torch.Tensor,
                         tgt_idx: torch.Tensor) -> torch.Tensor:
        x = tokens
        for mod in self.teacher[1:-1]:                   # skip patch_embed/norm
            x = mod(x)
        x = self.teacher[-1](x)                          # final LayerNorm
        d = x.shape[-1]
        tgt = x.gather(1, tgt_idx.unsqueeze(-1).expand(-1, -1, d))
        # data2vec/V-JEPA-style per-token instance norm of the targets.
        return F.layer_norm(tgt, (d,))

    def generate_masks(self, batch_size: int,
                       device: Optional[torch.device] = None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.masker(batch_size, device=device)

    def forward(self, video: torch.Tensor,
                ctx_idx: Optional[torch.Tensor] = None,
                tgt_idx: Optional[torch.Tensor] = None) -> dict:
        """One training step. `video`: (B, 3, T, H, W) in [0, 1]."""
        video = self._normalize(video)
        tokens = self.patch_embed(video)                 # (B, N, D)
        if ctx_idx is None or tgt_idx is None:
            ctx_idx, tgt_idx = self.generate_masks(tokens.shape[0],
                                                   device=tokens.device)

        ctx_feats = self._encode_context(tokens, ctx_idx)
        preds = self.predictor(ctx_feats, ctx_idx, tgt_idx,
                               self.config.num_patches)

        with torch.no_grad():
            t_tokens = self.teacher[0](video)            # teacher patch_embed
            targets = self._teacher_targets(t_tokens, tgt_idx)

        loss = F.smooth_l1_loss(preds, targets)
        return {"loss": loss, "preds": preds, "targets": targets}
