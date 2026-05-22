"""Standalone Point-JEPA inference module (PyTorch 2.8).

Self-contained reimplementation of the deployable half of Point-JEPA
(WACV 2025, arXiv:2404.16432, ref impl: github.com/Ayumu-J-S/Point-JEPA,
MIT). Includes only what is needed to encode a point cloud into per-token
embeddings — the EMA teacher / predictor used during JEPA pre-training
live in `PointJEPA_Trainer.py`.

Pipeline (matches the released config):
    1024 points
      -> FPS to C=64 centers, KNN k=32 grouping
      -> mini-PointNet patch embedding  (per group -> D=384)
      -> greedy *sequencer* (orders the C tokens so adjacent indices are
         spatially proximate; start = center with lowest coord sum, then
         iteratively append the nearest unvisited center)
      -> + MLP positional embedding of the (sequenced) centers
      -> 12-layer ViT encoder (D=384, heads=6)  ->  (B, C, 384)

The sequencer is the paper's core contribution: it makes proximity-based
context/target block selection a simple contiguous-span operation in the
trainer (`PointJEPA_Trainer.py`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


logger = logging.getLogger("PointJEPA")


# =============================================================================
# Geometry ops (pure torch, CPU-friendly — no torch_cluster dependency)
# =============================================================================

def farthest_point_sample(xyz: torch.Tensor, n_sample: int) -> torch.Tensor:
    """FPS center indices.

    xyz: (B, N, 3) -> (B, n_sample) long indices.
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, n_sample, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch = torch.arange(B, device=device)
    for i in range(n_sample):
        centroids[:, i] = farthest
        centroid = xyz[batch, farthest, :].view(B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(dim=-1).indices
    return centroids


def knn_group(xyz: torch.Tensor, centers: torch.Tensor,
              k: int) -> torch.Tensor:
    """k nearest neighbours of every center.

    xyz: (B, N, 3), centers: (B, C, 3) -> (B, C, k) long indices.
    """
    dist = torch.cdist(centers, xyz)                  # (B, C, N)
    return dist.topk(k, dim=-1, largest=False).indices


def _gather_xyz(xyz: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """xyz: (B, N, 3), idx: (B, M[, k]) -> gathered (B, M[, k], 3)."""
    B = xyz.shape[0]
    flat = idx.reshape(B, -1)
    out = torch.gather(xyz, 1, flat.unsqueeze(-1).expand(-1, -1, 3))
    return out.reshape(*idx.shape, 3)


# =============================================================================
# Tokenizer: FPS + KNN groups + mini-PointNet patch embedding
# =============================================================================

class PatchEmbed(nn.Module):
    """Point-BERT / Point-MAE style mini-PointNet group encoder.

    Input  : groups (B, C, k, 3) — k neighbours per center, center-relative.
    Output : tokens (B, C, embed_dim)
    """

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, embed_dim, 1),
        )

    def forward(self, groups: torch.Tensor) -> torch.Tensor:
        B, C, k, _ = groups.shape
        x = groups.reshape(B * C, k, 3).transpose(1, 2)        # (BC, 3, k)
        f = self.first(x)                                      # (BC, 256, k)
        g = f.max(dim=-1, keepdim=True).values                 # (BC, 256, 1)
        f = torch.cat([g.expand(-1, -1, k), f], dim=1)         # (BC, 512, k)
        f = self.second(f)                                     # (BC, D, k)
        return f.max(dim=-1).values.reshape(B, C, -1)          # (B, C, D)


class PointTokenizer(nn.Module):
    """Raw points -> (tokens, centers). FPS centers, KNN groups (made
    center-relative), mini-PointNet embedding."""

    def __init__(self, num_groups: int = 64, group_size: int = 32,
                 embed_dim: int = 384):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.embed = PatchEmbed(embed_dim)

    def forward(self, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        center_idx = farthest_point_sample(pts, self.num_groups)   # (B, C)
        centers = _gather_xyz(pts, center_idx)                     # (B, C, 3)
        nn_idx = knn_group(pts, centers, self.group_size)          # (B, C, k)
        groups = _gather_xyz(pts, nn_idx)                          # (B, C, k, 3)
        groups = groups - centers.unsqueeze(2)                     # center-relative
        return self.embed(groups), centers                         # tokens, centers


# =============================================================================
# Greedy sequencer (the Point-JEPA contribution)
# =============================================================================

@torch.no_grad()
def greedy_sequence(centers: torch.Tensor) -> torch.Tensor:
    """Order tokens so adjacent indices are spatially proximate.

    Start from the center with the lowest coordinate sum, then iteratively
    append the nearest not-yet-visited center.

    centers: (B, C, 3) -> permutation (B, C) long  (apply via gather).
    """
    B, C, _ = centers.shape
    device = centers.device
    dist = torch.cdist(centers, centers)                  # (B, C, C)
    order = torch.zeros(B, C, dtype=torch.long, device=device)
    visited = torch.zeros(B, C, dtype=torch.bool, device=device)
    cur = centers.sum(-1).min(dim=-1).indices             # (B,)
    batch = torch.arange(B, device=device)
    order[:, 0] = cur
    visited[batch, cur] = True
    for i in range(1, C):
        d = dist[batch, cur].clone()                      # (B, C)
        d[visited] = float("inf")
        cur = d.min(dim=-1).indices
        order[:, i] = cur
        visited[batch, cur] = True
    return order


def apply_sequence(x: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Reorder (B, C, D) (or (B, C, 3)) tokens by `order` (B, C)."""
    d = x.shape[-1]
    return torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, d))


# =============================================================================
# Config + encoder
# =============================================================================

@dataclass
class PointJEPAConfig:
    num_points: int = 1024
    num_groups: int = 64
    group_size: int = 32
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-6


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float,
                 qkv_bias: bool, eps: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = nn.MultiheadAttention(dim, heads, bias=qkv_bias,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(),
                                 nn.Linear(h, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: PointJEPAConfig, depth: Optional[int] = None,
                 dim: Optional[int] = None):
        super().__init__()
        d = dim or cfg.embed_dim
        self.blocks = nn.ModuleList([
            _Block(d, cfg.num_heads, cfg.mlp_ratio, cfg.qkv_bias,
                   cfg.layer_norm_eps)
            for _ in range(depth or cfg.depth)
        ])
        self.norm = nn.LayerNorm(d, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor, return_layers: bool = False):
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        if return_layers:
            return outs
        return self.norm(x)


class PosEmbed(nn.Module):
    """MLP positional embedding of the (sequenced) center coordinates."""

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim))

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        return self.mlp(centers)


class PointJEPA(nn.Module):
    """Inference-only Point-JEPA encoder.

    `self.tokenizer` + `self.pos_embed` + `self.encoder` are the deployed
    student. The EMA teacher / predictor used during JEPA pre-training are
    NOT here (see `PointJEPA_Trainer.py`).
    """

    def __init__(self, config: Optional[PointJEPAConfig] = None):
        super().__init__()
        cfg = config or PointJEPAConfig()
        self.config = cfg
        self.tokenizer = PointTokenizer(cfg.num_groups, cfg.group_size,
                                        cfg.embed_dim)
        self.pos_embed = PosEmbed(cfg.embed_dim)
        self.encoder = TransformerEncoder(cfg)

    def forward_tokens(self, pts: torch.Tensor):
        """-> (sequenced tokens (B,C,D), sequenced centers (B,C,3))."""
        tokens, centers = self.tokenizer(pts)
        order = greedy_sequence(centers)
        tokens = apply_sequence(tokens, order)
        centers = apply_sequence(centers, order)
        return tokens, centers

    @torch.inference_mode()
    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """Encode a point cloud into per-token features.

        pts: (B, N, 3) -> (B, num_groups, embed_dim).
        """
        if pts.ndim != 3 or pts.shape[-1] != 3:
            raise ValueError(
                f"pts must be (B, N, 3); got {tuple(pts.shape)}")
        tokens, centers = self.forward_tokens(pts)
        x = tokens + self.pos_embed(centers)
        return self.encoder(x)

    @torch.inference_mode()
    def get_shape_representation(self, pts: torch.Tensor) -> torch.Tensor:
        """Global shape descriptor: concat(mean, max) over tokens -> (B, 2D)
        (the Point-MAE/Point-JEPA SVM-eval feature)."""
        feats = self.forward(pts)
        return torch.cat([feats.mean(1), feats.max(1).values], dim=-1)

    # -------------------------------------------------------------------------
    # Checkpoint loading
    # -------------------------------------------------------------------------

    _STRIP_PREFIXES = ("module.", "_orig_mod.", "model.", "student.",
                       "encoder_student.", "backbone.")
    _KEEP_PREFIXES = ("tokenizer.", "pos_embed.", "encoder.")

    @classmethod
    def _unwrap(cls, obj) -> dict[str, torch.Tensor]:
        if isinstance(obj, dict):
            for key in ("student", "state_dict", "model_state_dict",
                        "model", "weights"):
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
        config: Optional[PointJEPAConfig] = None,
        map_location: str = "cpu",
        device: Optional[torch.device] = None,
        strict: bool = False,
    ) -> "PointJEPA":
        """Build a `PointJEPA` and load weights from a `.pt`/`.ckpt`.

        Auto-detects raw / Lightning `state_dict` / `model` blobs and the
        trainer's `{"student": ...}` dump, strips wrapper prefixes and keeps
        only the deployed encoder weights. Aborts (per paper/CLAUDE.md) if
        not a single tensor loads.
        """
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
                f"PointJEPA.from_checkpoint({path!r}): 0/{total} keys applied "
                "— checkpoint/architecture mismatch, aborting.")
        if not missing and not unexpected:
            logger.info("PointJEPA loaded successfully (%d/%d keys)",
                        applied, total)
        else:
            logger.warning(
                "PointJEPA partial load: %d/%d keys (missing=%d, unexpected=%d)",
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
