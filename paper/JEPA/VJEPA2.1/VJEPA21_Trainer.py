"""Standalone V-JEPA 2.1 trainer module (PyTorch 2.8).

V-JEPA 2.1 (arXiv:2603.14482, *Unlocking Dense Features in Video
Self-Supervised Learning*, Mur-Labadia et al., Meta, 2026-03) keeps the
V-JEPA 2 student/teacher/predictor stack but upgrades the objective so the
encoder learns high-quality, temporally-consistent **dense** features. The
two ingredients that *are* the SSL novelty are implemented here on top of
the sibling `VJEPA2Trainer` (`../VJEPA2`):

  (i) Dense Predictive Loss — V-JEPA 2 only penalised predictions at the
      *masked* tokens, so visible context tokens had no pressure to keep
      local spatial fidelity and degenerated into global aggregators. 2.1
      makes the predictor reconstruct the EMA-teacher features at **every**
      grid position (context ∪ masked); the masked region keeps weight 1.0
      while context is down-weighted by `context_loss_weight`.

  (ii) Deep Self-Supervision — the JEPA objective is also applied against
      teacher features taken at several **intermediate encoder depths**
      (not just the final layer). A lightweight linear head per supervised
      depth maps the predictor's hidden state to that depth's
      (instance-normed) teacher features, over all positions.

The other two paper ingredients are non-architectural: multi-modal training
(images enter as 1-frame clips — handled by the dataset) and scaling (config
knobs). The deployed student (`patch_embed`/`blocks`/`norm`) is byte-for-byte
the `VJEPA2` encoder, so checkpoints saved here load with
`VJEPA21.from_checkpoint(...)` (or `VJEPA2.from_checkpoint`).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- sibling import: the V-JEPA 2 trainer + its building blocks --------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_VJEPA2_DIR = os.path.join(os.path.dirname(_HERE), "VJEPA2")
for _p in (_HERE, _VJEPA2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from VJEPA2_Trainer import VJEPA2Trainer, VJEPA2TrainerConfig  # noqa: E402


logger = logging.getLogger("VJEPA21_Trainer")


# =============================================================================
# Trainer config (extends the V-JEPA 2 trainer config with the 2.1 recipe)
# =============================================================================

@dataclass
class VJEPA21TrainerConfig(VJEPA2TrainerConfig):
    # (i) Dense Predictive Loss — all tokens contribute; context is down-weighted.
    dense_loss: bool = True
    context_loss_weight: float = 0.1
    # (ii) Deep Self-Supervision — 1-indexed encoder depths whose teacher
    # features are additionally predicted (e.g. (8, 16, 24) for vit_large).
    deep_supervision_layers: tuple = ()
    deep_supervision_weight: float = 1.0


# =============================================================================
# Trainer model
# =============================================================================

class VJEPA21Trainer(VJEPA2Trainer):
    """V-JEPA 2 trainer + dense predictive loss + deep self-supervision.

    Reuses the parent's student encoder, EMA teacher, predictor, 3D
    multi-block masker, EMA schedule and weight init unchanged; only the
    config is upgraded to `VJEPA21TrainerConfig`, the per-depth heads are
    added, and `forward` is overridden.
    """

    _EXTRA_KEYS = ("dense_loss", "context_loss_weight",
                   "deep_supervision_layers", "deep_supervision_weight")

    def __init__(self, *, variant: str = "vit_large", **overrides):
        # Split off the 2.1-only knobs so the parent builds the (identical)
        # student/teacher/predictor stack, then re-stamp the richer config.
        extras = {k: overrides.pop(k) for k in self._EXTRA_KEYS if k in overrides}
        super().__init__(variant=variant, **overrides)

        base = {f.name: getattr(self.config, f.name) for f in fields(self.config)}
        base.update(extras)
        self.config = VJEPA21TrainerConfig(**base)
        cfg = self.config

        # Sanitise the supervised depths to the valid [1, depth] range.
        ds = tuple(int(x) for x in (cfg.deep_supervision_layers or ()))
        self._ds_layers = tuple(sorted({x for x in ds if 1 <= x <= cfg.depth}))
        # One linear head per supervised depth: predictor hidden -> encoder dim.
        self.deep_heads = nn.ModuleList([
            nn.Linear(cfg.predictor_embed_dim, cfg.embed_dim)
            for _ in self._ds_layers
        ])
        for h in self.deep_heads:
            nn.init.trunc_normal_(h.weight, std=0.02)
            nn.init.constant_(h.bias, 0)

    # -------------------------------------------------------------------------
    # Dense predictor pass (predict teacher features at ALL grid positions)
    # -------------------------------------------------------------------------

    def _dense_predict(self, ctx_feats: torch.Tensor, ctx_idx: torch.Tensor,
                       num_patches: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse the parent predictor's submodules but read out every token.

        Returns (pred_all, hidden):
            pred_all : (B, N, D_enc) predicted teacher features at all tokens
            hidden   : (B, N, D_pred) predictor hidden state (for deep heads)
        """
        p = self.predictor
        B = ctx_feats.shape[0]
        d = p.mask_token.shape[-1]
        full = p.mask_token.expand(B, num_patches, d).clone()
        full = full.scatter(
            1, ctx_idx.unsqueeze(-1).expand(-1, -1, d), p.embed(ctx_feats)
        )
        for blk in p.blocks:                       # RoPE over the full grid
            full = blk(full)
        hidden = p.norm(full)                      # (B, N, D_pred)
        return p.out(hidden), hidden               # (B, N, D_enc), (B, N, D_pred)

    # -------------------------------------------------------------------------
    # Teacher targets — final features (all tokens) + intermediate depths
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _teacher_final_and_depths(self, t_tokens: torch.Tensor,
                                  depths: set) -> tuple[torch.Tensor, dict]:
        x = t_tokens
        depth_feats: dict[int, torch.Tensor] = {}
        blocks = self.teacher[1:-1]                # encoder blocks (skip embed/norm)
        for i, blk in enumerate(blocks, start=1):
            x = blk(x)
            if i in depths:
                depth_feats[i] = F.layer_norm(x, (x.shape[-1],))
        x = self.teacher[-1](x)                    # final LayerNorm
        final = F.layer_norm(x, (x.shape[-1],))    # data2vec-style target norm
        return final, depth_feats

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(self, video: torch.Tensor,
                ctx_idx: torch.Tensor = None,
                tgt_idx: torch.Tensor = None) -> dict:
        """One 2.1 training step. `video`: (B, 3, T, H, W) in [0, 1]."""
        cfg = self.config
        video = self._normalize(video)
        tokens = self.patch_embed(video)           # (B, N, D)
        B, N, _ = tokens.shape
        if ctx_idx is None or tgt_idx is None:
            ctx_idx, tgt_idx = self.generate_masks(B, device=tokens.device)

        ctx_feats = self._encode_context(tokens, ctx_idx)        # (B, n_ctx, D)
        pred_all, hidden = self._dense_predict(ctx_feats, ctx_idx, N)

        with torch.no_grad():
            t_tokens = self.teacher[0](video)                    # teacher patch_embed
            target_all, depth_feats = self._teacher_final_and_depths(
                t_tokens, set(self._ds_layers))

        # ---- (i) main prediction loss ----
        if cfg.dense_loss:
            # All tokens contribute; masked region weight 1.0, context down-weighted.
            w = torch.ones(B, N, device=tokens.device)
            w.scatter_(1, ctx_idx, float(cfg.context_loss_weight))
            per_tok = F.smooth_l1_loss(
                pred_all, target_all, reduction="none").mean(-1)  # (B, N)
            main_loss = (per_tok * w).sum() / w.sum().clamp_min(1.0)
        else:
            d = pred_all.shape[-1]
            gather = lambda t: t.gather(1, tgt_idx.unsqueeze(-1).expand(-1, -1, d))
            main_loss = F.smooth_l1_loss(gather(pred_all), gather(target_all))

        # ---- (ii) deep self-supervision loss (over all tokens) ----
        deep_loss = pred_all.new_zeros(())
        for head, layer in zip(self.deep_heads, self._ds_layers):
            deep_loss = deep_loss + F.smooth_l1_loss(head(hidden), depth_feats[layer])
        if self.deep_heads:
            deep_loss = deep_loss / len(self.deep_heads)

        loss = main_loss + cfg.deep_supervision_weight * deep_loss
        return {
            "loss": loss,
            "main_loss": main_loss.detach(),
            "deep_loss": deep_loss.detach(),
        }
