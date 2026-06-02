"""Multi-ACCDOA SELD classifier on the frozen causal-Conformer encoder (PyTorch 2.8).

Stage-2 of `SeldJEPA`: a **Multi-ACCDOA** head with **N = 3 tracks/frame** on top of
the Stage-1 `SeldConformer` encoder (loaded frozen by default), trained with the
**ADPIT** (Auxiliary Duplicating Permutation Invariant Training) loss. DOA is
**2-D azimuth (x, y)** — elevation is unobservable from a 2-channel pair — so the
per-frame output is `(N, C, 2)`.

Frame-rate alignment (real-time): the encoder runs at 50 fps (STFT 100 fps →
front-end ×2); the SELD label rate is 10 Hz (100 ms). A temporal average-pool of
`pool_factor = 5` encoder frames → 1 label frame, i.e. **1 chunk (100 ms) == 1
label frame**. `pool_factor = label_hop_ms / (stft_hop_ms · frontend_downsample)`.

Output / decoding (Multi-ACCDOA, Shimada et al. arXiv:2110.07124):
    pred (B, T, N=3, C, 2)
    activity_{n,c} = || pred_{n,c} ||_2 ,  active iff > activity_threshold (0.5)
    azimuth_{n,c}  = atan2(y, x)
    track unification: same-class active tracks within `unify_angle_deg` (30°) are averaged.

`predict` applies `eval()` + `no_grad`; `from_checkpoint` freezes params and aborts
if nothing loads. `from_pretrained_encoder` builds Stage-2 from a Stage-1 `.pt`.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from SeldConformer import ConformerEncoder  # noqa: E402

logger = logging.getLogger("SeldMultiACCDOA")


# =============================================================================
# ADPIT loss — fixed 13 duplicated-permutation layouts for N = 3 tracks
# =============================================================================
# Per (class, frame) with M active sources (sources sorted active-first into
# slots 0..M-1, inactive slots zero), the ADPIT permutation set is:
#   M<=1 : layout (0,0,0)                            -> duplicate the lone source
#   M==2 : 6 surjections of 3 tracks onto slots {0,1} (== the paper's 12, de-duped)
#   M==3 : 6 permutations of slots (0,1,2)
# 1 + 6 + 6 = 13 layout evaluations (the canonical DCASE count). The min is taken
# over the layouts valid for each cell's M; mean over (B, T, C). (Eqs. 4-6.)
_LAYOUT_000 = (0, 0, 0)
_LAYOUTS_M2 = ((0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
_LAYOUTS_M3 = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def adpit_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ADPIT MSE loss.

    pred   : (B, T, N=3, C, D)        network output  (D = 2 for azimuth xy)
    target : (B, T, C, S=3, D)        up-to-3 per-class source vectors, sorted
                                       active-first, inactive slots zero.
    """
    tt = min(pred.shape[1], target.shape[1])
    pred = pred[:, :tt].permute(0, 1, 3, 2, 4)            # (B, T, C, N=3, D)
    target = target[:, :tt]                               # (B, T, C, S=3, D)
    m = (target.norm(dim=-1) > 0).sum(-1)                 # (B, T, C) active count

    def layout_loss(slots) -> torch.Tensor:
        tgt = torch.stack([target[..., s, :] for s in slots], dim=-2)  # (B,T,C,3,D)
        return ((pred - tgt) ** 2).mean(dim=(-1, -2))     # (B, T, C)

    l000 = layout_loss(_LAYOUT_000)
    l_m2 = torch.stack([layout_loss(s) for s in _LAYOUTS_M2], 0).amin(0)
    l_m3 = torch.stack([layout_loss(s) for s in _LAYOUTS_M3], 0).amin(0)
    loss = torch.where(m <= 1, l000, torch.where(m == 2, l_m2, l_m3))
    return loss.mean()


# =============================================================================
# Helpers
# =============================================================================

def conformer_encoder_kwargs(model_kwargs: dict) -> dict:
    """Map Stage-1 `SeldConformerConfig` kwargs -> `ConformerEncoder` kwargs."""
    mk = model_kwargs
    return dict(
        in_chans=int(mk.get("in_chans", 4)),
        n_mels=int(mk.get("n_mels", 64)),
        d_model=int(mk.get("encoder_dim", 512)),
        n_layers=int(mk.get("num_layers", 16)),
        n_heads=int(mk.get("num_heads", 8)),
        ffn_expansion=int(mk.get("ffn_expansion", 4)),
        conv_kernel=int(mk.get("conv_kernel", 31)),
        chunk_frames=int(mk.get("chunk_frames", 5)),
        dropout=float(mk.get("dropout", 0.1)),
        frontend_hidden=int(mk.get("frontend_hidden", 128)),
    )


def temporal_avg_pool(frames: torch.Tensor, factor: int) -> torch.Tensor:
    """(B, T_enc, D) -> (B, T_enc//factor, D) by non-overlapping average pooling."""
    if factor <= 1:
        return frames
    b, t, d = frames.shape
    tl = t // factor
    if tl == 0:                                           # clip shorter than one label hop
        return frames.mean(dim=1, keepdim=True)
    return frames[:, :tl * factor].reshape(b, tl, factor, d).mean(dim=2)


# =============================================================================
# Model
# =============================================================================

class SeldMultiACCDOA(nn.Module):
    """Frozen causal-Conformer encoder + Multi-ACCDOA head (N tracks, 2-D azimuth)."""

    DOA_DIM = 2                                           # (x, y) azimuth

    def __init__(self, encoder_kwargs: dict, num_tracks: int = 3, num_classes: int = 12,
                 pool_factor: int = 5, activity_threshold: float = 0.5,
                 unify_angle_deg: float = 30.0):
        super().__init__()
        self.encoder = ConformerEncoder(**conformer_encoder_kwargs(encoder_kwargs))
        self.encoder_kwargs = encoder_kwargs
        self.num_tracks = num_tracks
        self.num_classes = num_classes
        self.pool_factor = pool_factor
        self.activity_threshold = activity_threshold
        self.unify_angle_deg = unify_angle_deg
        self.head = nn.Linear(self.encoder.d_model, num_tracks * num_classes * self.DOA_DIM)

    # -- core -----------------------------------------------------------------
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat (B, C_in, T_f, M) -> Multi-ACCDOA pred (B, T_label, N, C, 2)."""
        frames = self.encoder(feat)                       # (B, T_enc, D)
        pooled = temporal_avg_pool(frames, self.pool_factor)
        logits = self.head(pooled)                        # (B, T_label, N*C*2)
        b, tl = logits.shape[:2]
        return logits.reshape(b, tl, self.num_tracks, self.num_classes, self.DOA_DIM)

    def compute_loss(self, feat: torch.Tensor, target: torch.Tensor) -> dict:
        pred = self.forward(feat)
        loss = adpit_loss(pred, target)
        return {"loss": loss, "pred": pred, "adpit_loss": loss.detach()}

    def freeze_encoder(self):
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self):
        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def predict(self, feat: torch.Tensor) -> torch.Tensor:
        """eval-mode forward -> Multi-ACCDOA pred (B, T_label, N, C, 2)."""
        self.eval()
        return self.forward(feat)

    def decode(self, pred: torch.Tensor) -> dict:
        """pred (B, T, N, C, 2) -> per-track activity (bool) + azimuth (deg)."""
        norm = pred.norm(dim=-1)                          # (B, T, N, C)
        active = norm > self.activity_threshold
        azimuth = torch.rad2deg(torch.atan2(pred[..., 1], pred[..., 0]))  # (B, T, N, C)
        return {"active": active, "azimuth_deg": azimuth, "norm": norm}

    # -- checkpoints ----------------------------------------------------------
    @staticmethod
    def read_encoder_kwargs(encoder_ckpt: str, map_location: str = "cpu") -> dict:
        """Read the Stage-1 `.pt`'s stored `model_kwargs` (encoder architecture)."""
        blob = torch.load(encoder_ckpt, map_location=map_location, weights_only=False)
        if not (isinstance(blob, dict) and "state_dict" in blob):
            raise ValueError(f"{encoder_ckpt!r}: expected a Stage-1 dump with "
                             "'state_dict'/'model_kwargs'.")
        return blob.get("model_kwargs", {})

    def load_encoder_weights(self, encoder_ckpt: str, freeze: bool = True,
                             map_location: str = "cpu") -> None:
        """Load Stage-1 `encoder.*` weights into this model's encoder; abort if
        none match; optionally freeze (`eval()` + `requires_grad_(False)`)."""
        blob = torch.load(encoder_ckpt, map_location=map_location, weights_only=False)
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        applied = len(enc_sd) - len(unexpected)
        if applied == 0:
            raise RuntimeError(
                f"load_encoder_weights({encoder_ckpt!r}): 0/{len(enc_sd)} encoder keys "
                "applied — Stage-1/Stage-2 architecture mismatch, aborting.")
        logger.info("Stage-1 encoder loaded %d/%d keys (missing=%d, unexpected=%d) — %s",
                    applied, len(enc_sd), len(missing), len(unexpected),
                    "frozen" if freeze else "trainable")
        if freeze:
            self.freeze_encoder()

    @classmethod
    def from_pretrained_encoder(cls, encoder_ckpt: str, *, num_tracks: int = 3,
                                num_classes: int = 12, pool_factor: int = 5,
                                activity_threshold: float = 0.5,
                                unify_angle_deg: float = 30.0,
                                freeze_encoder: bool = True,
                                map_location: str = "cpu") -> "SeldMultiACCDOA":
        """Build Stage-2 from a Stage-1 `.pt` (rebuilds the encoder from its stored
        `model_kwargs`, loads `encoder.*` weights, aborts if none match)."""
        enc_kwargs = cls.read_encoder_kwargs(encoder_ckpt, map_location)
        model = cls(enc_kwargs, num_tracks=num_tracks, num_classes=num_classes,
                    pool_factor=pool_factor, activity_threshold=activity_threshold,
                    unify_angle_deg=unify_angle_deg)
        model.load_encoder_weights(encoder_ckpt, freeze=freeze_encoder,
                                   map_location=map_location)
        return model

    def model_kwargs(self) -> dict:
        return dict(encoder_kwargs=self.encoder_kwargs, num_tracks=self.num_tracks,
                    num_classes=self.num_classes, pool_factor=self.pool_factor,
                    activity_threshold=self.activity_threshold,
                    unify_angle_deg=self.unify_angle_deg)

    @classmethod
    def from_checkpoint(cls, path: str, map_location: str = "cpu", device=None,
                        **overrides) -> "SeldMultiACCDOA":
        """Load a full Stage-2 checkpoint (encoder + head) for evaluation. Aborts
        if nothing loads; then `eval()` + frozen params."""
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(blob, dict) and "state_dict" in blob:
            sd = blob["state_dict"]
            kwargs = {**blob.get("model_kwargs", {}), **overrides}
        else:
            sd = blob
            kwargs = overrides
        if "encoder_kwargs" not in kwargs:
            raise ValueError(f"{path!r}: missing 'encoder_kwargs' in checkpoint/overrides.")
        model = cls(**kwargs)
        sd = {k[len("seld_accdoa."):] if k.startswith("seld_accdoa.") else k: v
              for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        applied = len(sd) - len(unexpected)
        if applied == 0:
            raise RuntimeError(
                f"from_checkpoint({path!r}): 0/{len(sd)} tensors matched — aborting.")
        logger.info("SeldMultiACCDOA loaded %d/%d tensors (missing=%d, unexpected=%d)",
                    applied, len(sd), len(missing), len(unexpected))
        if device is not None:
            model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model
