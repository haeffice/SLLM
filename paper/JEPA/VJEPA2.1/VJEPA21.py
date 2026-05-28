"""Standalone V-JEPA 2.1 inference module (PyTorch 2.8).

V-JEPA 2.1 (arXiv:2603.14482, Meta, 2026-03) keeps the *deployed* video
encoder architecturally identical to V-JEPA 2 — it changes only the
self-supervised *training recipe* (Dense Predictive Loss + Deep
Self-Supervision; see `VJEPA21_Trainer.py`). The recipe yields much
higher-quality, temporally-consistent **dense per-token features**, so the
only inference-time addition here is a convenience accessor for that dense
token grid. Everything else (3D ViT + RoPE encoder, attentive probe,
`from_checkpoint`) is inherited unchanged from the sibling `../VJEPA2`.

Because the student encoder is identical, a 2.1 student checkpoint
(`vjepa21_student_step*.pt`) and an original V-JEPA 2 checkpoint are
mutually loadable via `from_checkpoint`.
"""

from __future__ import annotations

import os
import sys

import torch

# --- sibling import: reuse the V-JEPA 2 encoder verbatim --------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_VJEPA2_DIR = os.path.join(os.path.dirname(_HERE), "VJEPA2")
for _p in (_HERE, _VJEPA2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from VJEPA2 import VARIANTS, VJEPA2, VJEPA2Config  # noqa: E402,F401


class VJEPA21(VJEPA2):
    """Inference-only V-JEPA 2.1 video encoder.

    Identical deployed network to `VJEPA2`; the 2.1 recipe lives in the
    trainer. `get_dense_features` is the intended entry point — it returns
    the per-token grid `(B, N, D)` whose quality 2.1 specifically targets.
    """

    @torch.inference_mode()
    def get_dense_features(self, video: torch.Tensor,
                           normalize: bool = True) -> torch.Tensor:
        """Dense per-token features `(B, N, D)` (no pooling).

        These are the spatio-temporally grounded features that the V-JEPA 2.1
        Dense Predictive Loss + Deep Self-Supervision recipe sharpens.
        """
        if normalize:
            video = self.normalize_pixels(video)
        return self.forward_features(video)


# =============================================================================
# Convenience constructors (mirror VJEPA2's)
# =============================================================================

def vjepa21_vit_large(**kw) -> VJEPA21:
    return VJEPA21(VJEPA2Config.from_variant("vit_large", **kw))


def vjepa21_vit_huge(**kw) -> VJEPA21:
    return VJEPA21(VJEPA2Config.from_variant("vit_huge", **kw))


def vjepa21_vit_giant(**kw) -> VJEPA21:
    return VJEPA21(VJEPA2Config.from_variant("vit_giant", **kw))


def vjepa21_vit_gigantic(**kw) -> VJEPA21:
    return VJEPA21(VJEPA2Config.from_variant("vit_gigantic", **kw))


__all__ = [
    "VJEPA21", "VJEPA2Config", "VARIANTS",
    "vjepa21_vit_large", "vjepa21_vit_huge",
    "vjepa21_vit_giant", "vjepa21_vit_gigantic",
]
