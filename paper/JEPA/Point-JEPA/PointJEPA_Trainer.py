"""Standalone Point-JEPA trainer module (PyTorch 2.8).

`PointJEPATrainer` is the JEPA self-supervised pre-training wrapper. It owns

    * the *student* path the inference model (`PointJEPA`) reuses —
      `self.tokenizer`, `self.pos_embed`, `self.encoder`.
    * a *teacher* path — `self.teacher_encoder`, a frozen EMA copy of the
      student encoder that produces the prediction targets over ALL tokens.
    * a *predictor* — `self.predictor`, a narrow Transformer (depth 6,
      dim 192) that, from the context-token reps + positional embeddings of
      the target centers + a learned mask token, predicts the EMA-teacher
      features at the target tokens. Encoder<->predictor dim mappers
      included.

Context / target blocks are *contiguous spans of the sequenced order*
(`PointJEPA.greedy_sequence`) — that is exactly why the sequencer exists:
proximity-based block selection becomes a simple slice. Defaults follow the
released config: 4 target blocks ratio∈(0.15,0.2), context ratio∈(0.4,0.75),
smooth-L1 loss (β=2), EMA decay 0.995→1.0.

References:
    * I-JEPA (Assran et al. 2023, arXiv:2301.08243)
    * Point-MAE (Pang et al. 2022, arXiv:2203.06604) — tokenizer
    * Point-JEPA (Saito et al. WACV 2025, arXiv:2404.16432,
                  github.com/Ayumu-J-S/Point-JEPA)

Student-side state-dict keys match `PointJEPA`, so a checkpoint saved here
loads with `PointJEPA.from_checkpoint(...)`.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from PointJEPA import (
    PointJEPAConfig,
    PosEmbed,
    TransformerEncoder,
    apply_sequence,
    greedy_sequence,
)


logger = logging.getLogger("PointJEPA_Trainer")


# =============================================================================
# Trainer config
# =============================================================================

@dataclass
class PointJEPATrainerConfig(PointJEPAConfig):
    predictor_dim: int = 192
    predictor_depth: int = 6

    ema_decay: float = 0.995
    ema_end_decay: float = 1.0
    ema_anneal_end_step: int = 100_000

    num_target_blocks: int = 4
    target_ratio_min: float = 0.15
    target_ratio_max: float = 0.20
    context_ratio_min: float = 0.40
    context_ratio_max: float = 0.75
    smooth_l1_beta: float = 2.0


# =============================================================================
# Contiguous block masker over the sequenced token order
# =============================================================================

class _BlockMasker:
    """Sample context / target index sets as contiguous spans of the
    sequenced (proximity-ordered) tokens.

    Returns, per batch item, equal-length index tensors (counts equalised
    across the batch so they stack):
        ctx_idx : (B, n_ctx)               visible context tokens
        tgt_idx : (B, M, n_tgt)            M target blocks
    """

    def __init__(self, cfg: PointJEPATrainerConfig):
        self.cfg = cfg

    def _span(self, C: int, lo: float, hi: float) -> tuple[int, int]:
        size = max(1, int(round(random.uniform(lo, hi) * C)))
        size = min(size, C)
        start = random.randint(0, C - size)
        return start, size

    def __call__(self, B: int, C: int,
                 device: Optional[torch.device] = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        M = cfg.num_target_blocks
        ctx_all, tgt_all = [], []
        for _ in range(B):
            t_start, t_size = self._span(C, cfg.target_ratio_min,
                                         cfg.target_ratio_max)
            blocks = []
            tgt_union = torch.zeros(C, dtype=torch.bool)
            for _m in range(M):
                s, sz = self._span(C, cfg.target_ratio_min,
                                   cfg.target_ratio_max)
                blk = torch.arange(s, s + sz)
                blocks.append(blk)
                tgt_union[s:s + sz] = True
            cs, csz = self._span(C, cfg.context_ratio_min,
                                 cfg.context_ratio_max)
            ctx_mask = torch.zeros(C, dtype=torch.bool)
            ctx_mask[cs:cs + csz] = True
            ctx_mask &= ~tgt_union                       # remove target overlap
            if ctx_mask.sum() == 0:                      # guarantee >=1 context
                free = torch.where(~tgt_union)[0]
                ctx_mask[free[0] if len(free) else 0] = True
            ctx_all.append(torch.where(ctx_mask)[0])
            tgt_all.append(blocks)

        n_ctx = min(len(c) for c in ctx_all)
        n_tgt = min(min(len(b) for b in blks) for blks in tgt_all)
        ctx_idx = torch.stack([c[:n_ctx] for c in ctx_all])
        tgt_idx = torch.stack([
            torch.stack([b[:n_tgt] for b in blks]) for blks in tgt_all
        ])                                               # (B, M, n_tgt)
        if device is not None:
            ctx_idx, tgt_idx = ctx_idx.to(device), tgt_idx.to(device)
        return ctx_idx, tgt_idx


# =============================================================================
# Predictor (narrow transformer over context + mask tokens)
# =============================================================================

class _Predictor(nn.Module):
    def __init__(self, cfg: PointJEPATrainerConfig):
        super().__init__()
        d = cfg.predictor_dim
        self.embed = nn.Linear(cfg.embed_dim, d)
        self.pos = PosEmbed(d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.transformer = TransformerEncoder(cfg, depth=cfg.predictor_depth,
                                              dim=d)
        self.out = nn.Linear(d, cfg.embed_dim)

    def forward(self, ctx_tokens: torch.Tensor, ctx_centers: torch.Tensor,
                tgt_centers: torch.Tensor) -> torch.Tensor:
        """ctx_tokens (B, n_ctx, D_enc), *_centers (B, *, 3).
        Returns predicted encoder-dim features at the target tokens
        (B, n_tgt, D_enc)."""
        B, n_tgt, _ = tgt_centers.shape
        c = self.embed(ctx_tokens) + self.pos(ctx_centers)     # (B, n_ctx, d)
        m = self.mask_token.expand(B, n_tgt, -1) + self.pos(tgt_centers)
        x = torch.cat([c, m], dim=1)
        x = self.transformer(x)
        return self.out(x[:, -n_tgt:, :])                      # (B, n_tgt, D_enc)


# =============================================================================
# Trainer model
# =============================================================================

class PointJEPATrainer(nn.Module):
    """JEPA trainer: student encoder + EMA teacher + narrow predictor.

    `self.tokenizer` / `self.pos_embed` / `self.encoder` are the deployable
    encoder (student). After training, save `self.state_dict()` and load it
    with `PointJEPA.from_checkpoint(...)`.
    """

    def __init__(self, **overrides):
        super().__init__()
        self.config = PointJEPATrainerConfig(**overrides)
        cfg = self.config

        from PointJEPA import PointTokenizer
        self.tokenizer = PointTokenizer(cfg.num_groups, cfg.group_size,
                                        cfg.embed_dim)
        self.pos_embed = PosEmbed(cfg.embed_dim)
        self.encoder = TransformerEncoder(cfg)
        self.predictor = _Predictor(cfg)

        self.masker = _BlockMasker(cfg)
        self.apply(self._init_weights)
        self._init_teacher()

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_teacher(self):
        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_encoder.requires_grad_(False)

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
        for sp, tp in zip(self.encoder.parameters(),
                          self.teacher_encoder.parameters()):
            tp.data.mul_(r).add_(sp.detach().data, alpha=(1.0 - r))
        for sb, tb in zip(self.encoder.buffers(),
                          self.teacher_encoder.buffers()):
            tb.data.copy_(sb.data)
        return r

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def generate_masks(self, B: int, device=None):
        return self.masker(B, self.config.num_groups, device=device)

    @staticmethod
    def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """x (B, C, D), idx (B, n) -> (B, n, D)."""
        d = x.shape[-1]
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, d))

    def forward(self, points: torch.Tensor,
                ctx_idx: Optional[torch.Tensor] = None,
                tgt_idx: Optional[torch.Tensor] = None) -> dict:
        """One training step. `points`: (B, N, 3)."""
        tokens, centers = self.tokenizer(points)
        order = greedy_sequence(centers)
        tokens = apply_sequence(tokens, order)              # (B, C, D)
        centers = apply_sequence(centers, order)            # (B, C, 3)
        B, C, D = tokens.shape

        if ctx_idx is None or tgt_idx is None:
            ctx_idx, tgt_idx = self.generate_masks(B, device=tokens.device)
        M = tgt_idx.shape[1]

        # ---- student over context tokens ----
        ctx_tok = self._gather(tokens, ctx_idx)
        ctx_ctr = self._gather(centers, ctx_idx)
        ctx_feat = self.encoder(ctx_tok + self.pos_embed(ctx_ctr))

        # ---- teacher over ALL tokens (targets read off) ----
        with torch.no_grad():
            t_all = self.teacher_encoder(tokens + self.pos_embed(centers))
            t_all = F.layer_norm(t_all, (D,))               # normalised targets

        # ---- predict each target block, smooth-L1 against teacher ----
        loss = tokens.new_zeros(())
        for m in range(M):
            ti = tgt_idx[:, m, :]
            tgt_ctr = self._gather(centers, ti)
            pred = self.predictor(ctx_feat, ctx_ctr, tgt_ctr)
            target = self._gather(t_all, ti)
            loss = loss + F.smooth_l1_loss(
                pred, target, beta=self.config.smooth_l1_beta)
        loss = loss / max(1, M)
        return {"loss": loss}
