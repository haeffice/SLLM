"""Standalone V-JEPA 2 inference module (PyTorch 2.8).

Self-contained reimplementation of the deployable half of V-JEPA 2
(arXiv:2506.09985, ref impl: github.com/facebookresearch/vjepa2,
ckpts: huggingface.co/facebook/vjepa2-* and the torch.hub weights).
Includes only what is needed to encode a video clip into per-token
embeddings plus an attentive-probe classifier head — the EMA teacher,
predictor and mask tokens used during JEPA pre-training live in
`VJEPA2_Trainer.py`.

Architecture (matches the released ViT-* encoders):
    PatchEmbed3D : Conv3d(3 -> D, k=(tubelet, P, P), stride=same)
        -> N = (T/tubelet) * (H/P) * (W/P) tokens, NO [CLS] token
        -> L pre-norm Transformer blocks (3D-RoPE self-attention, GELU MLP)
        -> final LayerNorm
    Input  : video (B, C=3, T, H, W), pixels in [0, 1] then ImageNet-norm
    Output : (B, N, D) per-token features

Encoder variants (depth / embed_dim / heads / mlp_ratio):
    vit_large    24 / 1024 / 16 / 4
    vit_huge     32 / 1280 / 16 / 4
    vit_giant    40 / 1408 / 16 / 48/11
    vit_gigantic 48 / 1664 / 16 / 64/13
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("VJEPA2")


# =============================================================================
# Constants — V-JEPA 2 default video front-end
# =============================================================================

IN_CHANS = 3
PATCH_SIZE = 16
TUBELET_SIZE = 2
# ImageNet mean/std (V-JEPA 2 normalises pixels with these after /255).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# Patch embedding (Conv3d over a tubelet)
# =============================================================================

class PatchEmbed3D(nn.Module):
    """Video -> token sequence via a single Conv3d over (tubelet, P, P).

    Input  : (B, C, T, H, W)
    Output : (B, N, D) with N = (T/tubelet)*(H/P)*(W/P)
    """

    def __init__(self, patch_size: int = PATCH_SIZE, tubelet_size: int = TUBELET_SIZE,
                 in_chans: int = IN_CHANS, embed_dim: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                       # (B, D, T', H', W')
        return x.flatten(2).transpose(1, 2)    # (B, N, D)


# =============================================================================
# 3D rotary position embedding (the V-JEPA 2 "_rope" encoders)
# =============================================================================

class RoPE3D(nn.Module):
    """Axial 3D rotary embedding.

    The per-head channel block is split into three equal thirds rotated by
    the temporal / height / width grid coordinates respectively (any
    leftover channels are passed through un-rotated). Matches the axial
    factorisation used by the V-JEPA 2 reference RoPE attention.
    """

    def __init__(self, head_dim: int, grid_size: tuple[int, int, int],
                 base: float = 10_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.grid_size = grid_size
        # per-axis rotary width (must be even); remainder left unrotated.
        d3 = head_dim // 3
        self.d_rot = (d3 // 2) * 2
        cos, sin = self._build_tables(grid_size, base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _axis_freqs(self, n: int, base: float) -> torch.Tensor:
        half = self.d_rot // 2
        inv = 1.0 / (base ** (torch.arange(half, dtype=torch.float64) / max(1, half)))
        pos = torch.arange(n, dtype=torch.float64)
        return torch.outer(pos, inv)                       # (n, half)

    def _build_tables(self, grid: tuple[int, int, int], base: float):
        T, H, W = grid
        ft = self._axis_freqs(T, base)                     # (T, half)
        fh = self._axis_freqs(H, base)
        fw = self._axis_freqs(W, base)
        # Broadcast each axis over the full (T,H,W) grid, then flatten to N.
        ang_t = ft[:, None, None, :].expand(T, H, W, -1)
        ang_h = fh[None, :, None, :].expand(T, H, W, -1)
        ang_w = fw[None, None, :, :].expand(T, H, W, -1)
        ang = torch.cat([ang_t, ang_h, ang_w], dim=-1)     # (T,H,W, 3*half)
        ang = ang.reshape(T * H * W, -1)                   # (N, 3*half)
        ang = torch.cat([ang, ang], dim=-1)                # duplicate halves
        return ang.cos().float(), ang.sin().float()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor,
                pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, heads, N, head_dim) -> rotated tensor, same shape.

        `pos` (B, N) Long, optional: true grid token index of every
        position in `x`. Needed when `x` is a *gathered subset* of the
        full token grid (the JEPA context encoder) so each token still
        rotates by its real spatio-temporal coordinate. When omitted the
        first `N` table rows are used (correct for the full grid)."""
        rot = 3 * self.d_rot
        if rot == 0:
            return x
        x_rot, x_pass = x[..., :rot], x[..., rot:]
        if pos is None:
            cos = self.cos[: x.shape[-2]].to(x.dtype)
            sin = self.sin[: x.shape[-2]].to(x.dtype)
        else:
            # (B, N, rot) -> (B, 1, N, rot) to broadcast over heads.
            cos = self.cos[pos].to(x.dtype).unsqueeze(1)
            sin = self.sin[pos].to(x.dtype).unsqueeze(1)
        x_rot = x_rot * cos + self._rotate_half(x_rot) * sin
        return torch.cat([x_rot, x_pass], dim=-1)


# =============================================================================
# Transformer block (pre-norm, 3D-RoPE MHSA, GELU MLP)
# =============================================================================

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 rope: Optional[RoPE3D] = None):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.rope = rope

    def forward(self, x: torch.Tensor,
                pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                    # (3, B, h, N, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope is not None:
            q, k = self.rope(q, pos), self.rope(k, pos)
        x = F.scaled_dot_product_attention(q, k, v)         # (B, h, N, hd)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float,
                 qkv_bias: bool, layer_norm_eps: float,
                 rope: Optional[RoPE3D] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias, rope=rope)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor,
                pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), pos)
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# Config + variants
# =============================================================================

# (depth, embed_dim, num_heads, mlp_ratio) — matches the reference impl.
VARIANTS: dict[str, tuple[int, int, int, float]] = {
    "vit_large":    (24, 1024, 16, 4.0),
    "vit_huge":     (32, 1280, 16, 4.0),
    "vit_giant":    (40, 1408, 16, 48 / 11),
    "vit_gigantic": (48, 1664, 16, 64 / 13),
}


@dataclass
class VJEPA2Config:
    # video window
    img_size: int = 256
    num_frames: int = 16
    in_chans: int = IN_CHANS
    patch_size: int = PATCH_SIZE
    tubelet_size: int = TUBELET_SIZE

    # encoder
    variant: str = "vit_large"
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-6
    use_rope: bool = True

    # attentive-probe classifier head (set num_classes>0 to build it)
    num_classes: int = 0
    probe_num_heads: int = 16

    @classmethod
    def from_variant(cls, variant: str, **overrides) -> "VJEPA2Config":
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
        depth, dim, heads, mlp = VARIANTS[variant]
        kw = dict(variant=variant, embed_dim=dim, depth=depth,
                  num_heads=heads, mlp_ratio=mlp)
        kw.update(overrides)
        return cls(**kw)

    @property
    def grid_size(self) -> tuple[int, int, int]:
        return (
            self.num_frames // self.tubelet_size,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
        )

    @property
    def num_patches(self) -> int:
        t, h, w = self.grid_size
        return t * h * w


# =============================================================================
# Attentive probe (frozen-feature classifier head)
# =============================================================================

class AttentiveProbe(nn.Module):
    """Single learnable query cross-attends over the frozen token features,
    then a linear classifier. This is the V-JEPA 2 attentive-probe protocol
    used for all frozen-encoder downstream evaluations."""

    def __init__(self, dim: int, num_classes: int, num_heads: int = 16):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.head(self.norm(pooled.squeeze(1)))


# =============================================================================
# Inference model
# =============================================================================

class VJEPA2(nn.Module):
    """Inference-only V-JEPA 2 video encoder (+ optional attentive probe).

    `self.patch_embed` + `self.blocks` + `self.norm` are exactly the
    deployed student encoder. The EMA teacher / predictor / mask tokens
    used during JEPA pre-training are NOT here (see `VJEPA2_Trainer.py`).
    """

    def __init__(self, config: Optional[VJEPA2Config] = None):
        super().__init__()
        cfg = config or VJEPA2Config()
        self.config = cfg

        self.patch_embed = PatchEmbed3D(
            patch_size=cfg.patch_size, tubelet_size=cfg.tubelet_size,
            in_chans=cfg.in_chans, embed_dim=cfg.embed_dim,
        )

        rope = (
            RoPE3D(cfg.embed_dim // cfg.num_heads, cfg.grid_size)
            if cfg.use_rope else None
        )
        if not cfg.use_rope:
            # Fixed 3D sin-cos fallback (kept un-trained, like the ref impl).
            self.register_buffer(
                "pos_embed",
                _sincos_pos_embed_3d(cfg.embed_dim, cfg.grid_size),
                persistent=True,
            )
        else:
            self.pos_embed = None

        self.blocks = nn.ModuleList([
            Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio,
                  cfg.qkv_bias, cfg.layer_norm_eps, rope=rope)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)

        self.probe: Optional[AttentiveProbe] = (
            AttentiveProbe(cfg.embed_dim, cfg.num_classes, cfg.probe_num_heads)
            if cfg.num_classes > 0 else None
        )

        self.register_buffer(
            "_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1),
            persistent=False,
        )

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def normalize_pixels(self, video: torch.Tensor) -> torch.Tensor:
        """(B, 3, T, H, W) in [0, 1] -> ImageNet-normalised."""
        return (video - self._mean) / self._std

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(
                f"video must be 5D (B, C, T, H, W); got {tuple(video.shape)}"
            )
        x = self.patch_embed(video)                     # (B, N, D)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)                             # (B, N, D)

    @torch.inference_mode()
    def forward(self, video: torch.Tensor,
                normalize: bool = True) -> torch.Tensor:
        """Encode a clip into per-token features (B, N, D), or class logits
        (B, num_classes) when an attentive probe is attached."""
        if normalize:
            video = self.normalize_pixels(video)
        feats = self.forward_features(video)
        if self.probe is not None:
            return self.probe(feats)
        return feats

    @torch.inference_mode()
    def get_video_representation(self, video: torch.Tensor,
                                 normalize: bool = True) -> torch.Tensor:
        """Clip-level embedding: mean-pooled token features (B, D)."""
        if normalize:
            video = self.normalize_pixels(video)
        return self.forward_features(video).mean(dim=1)

    # -------------------------------------------------------------------------
    # Checkpoint loading
    # -------------------------------------------------------------------------

    # Wrapper prefixes peeled off in order (DDP `module.`, compile
    # `_orig_mod.`, the ref impl's `backbone.`/`encoder.`, HF
    # `vjepa2.`/`model.`). The HF `transformers` port renames a few
    # submodules — handled by `_remap_hf_keys`.
    _STRIP_PREFIXES = ("module.", "_orig_mod.", "backbone.", "encoder.",
                       "model.", "vjepa2.")

    @classmethod
    def _unwrap_state_dict(cls, obj) -> dict[str, torch.Tensor]:
        if isinstance(obj, dict):
            # The torch.hub V-JEPA 2 ckpt stores the encoder under "encoder"
            # / "target_encoder"; prefer the EMA target encoder for inference.
            for key in ("target_encoder", "encoder", "state_dict",
                        "model_state_dict", "model", "weights"):
                if key in obj and isinstance(obj[key], dict):
                    return cls._unwrap_state_dict(obj[key])
            if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj
        raise ValueError(
            f"Could not find a tensor state-dict in checkpoint ({type(obj)})"
        )

    @staticmethod
    def _remap_hf_keys(k: str) -> str:
        """Map HuggingFace `VJEPA2Model` names onto this module's names."""
        k = k.replace("patch_embeddings.projection.", "patch_embed.proj.")
        k = k.replace("layernorm.", "norm.")
        k = k.replace("layer.", "blocks.")
        k = k.replace(".attention.qkv.", ".attn.qkv.")
        k = k.replace(".attention.proj.", ".attn.proj.")
        k = k.replace(".mlp.fc1.", ".mlp.fc1.").replace(".mlp.fc2.", ".mlp.fc2.")
        return k

    @classmethod
    def _filter_keys(cls, sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for raw_k, v in sd.items():
            k = raw_k
            changed = True
            while changed:
                changed = False
                for p in cls._STRIP_PREFIXES:
                    if k.startswith(p):
                        k, changed = k[len(p):], True
            out[cls._remap_hf_keys(k)] = v
        return out

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        config: Optional[VJEPA2Config] = None,
        map_location: str = "cpu",
        device: Optional[torch.device] = None,
        strict: bool = False,
    ) -> "VJEPA2":
        """Build a `VJEPA2` and load weights from a `.pt`/`.pth` checkpoint.

        Auto-detects raw state dicts, Lightning `state_dict`, training
        scripts' `model_state_dict`/`model`, and the official V-JEPA 2
        `{"encoder": ..., "target_encoder": ..., "predictor": ...}` blob
        (the EMA `target_encoder` is preferred for inference). Wrapper
        prefixes are peeled and HF names remapped automatically. Aborts
        (per paper/CLAUDE.md) if not a single tensor loads.
        """
        model = cls(config)
        try:
            blob = torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            blob = torch.load(path, map_location=map_location, weights_only=False)

        sd = cls._unwrap_state_dict(blob)
        filtered = cls._filter_keys(sd)
        missing, unexpected = model.load_state_dict(filtered, strict=False)

        total = len(filtered)
        applied = total - len(unexpected)
        if applied == 0:
            raise RuntimeError(
                f"VJEPA2.from_checkpoint({path!r}): 0/{total} keys applied — "
                "checkpoint/architecture mismatch, aborting."
            )
        if not missing and not unexpected:
            logger.info("VJEPA2 loaded successfully (%d/%d keys)", applied, total)
        else:
            logger.warning(
                "VJEPA2 partial load: %d/%d keys (missing=%d, unexpected=%d)",
                applied, total, len(missing), len(unexpected),
            )
            if strict and (missing or unexpected):
                raise RuntimeError(
                    f"strict=True but missing={missing} unexpected={unexpected}"
                )

        if device is not None:
            model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model


# =============================================================================
# Fixed 3D sin-cos positional embedding (use_rope=False fallback)
# =============================================================================

def _sincos_1d(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega = 1.0 / (10_000.0 ** (omega / (embed_dim / 2.0)))
    out = torch.outer(pos.double(), omega)
    return torch.cat([out.sin(), out.cos()], dim=1)


def _sincos_pos_embed_3d(embed_dim: int,
                         grid: tuple[int, int, int]) -> torch.Tensor:
    """(1, T*H*W, embed_dim) factorised t/h/w sin-cos table."""
    T, H, W = grid
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 3D sincos"
    dt = embed_dim // 2                        # half the channels for time
    dhw = embed_dim - dt                       # rest split over h, w
    dh = (dhw // 2 // 2) * 2
    dw = dhw - dh
    gt = torch.arange(T)
    gh = torch.arange(H)
    gw = torch.arange(W)
    et = _sincos_1d(dt, gt)[:, None, None, :].expand(T, H, W, dt)
    eh = _sincos_1d(dh, gh)[None, :, None, :].expand(T, H, W, dh)
    ew = _sincos_1d(dw, gw)[None, None, :, :].expand(T, H, W, dw)
    emb = torch.cat([et, eh, ew], dim=-1).reshape(T * H * W, embed_dim)
    return emb.float().unsqueeze(0)


# =============================================================================
# Convenience constructors
# =============================================================================

def vjepa2_vit_large(**kw) -> VJEPA2:
    return VJEPA2(VJEPA2Config.from_variant("vit_large", **kw))


def vjepa2_vit_huge(**kw) -> VJEPA2:
    return VJEPA2(VJEPA2Config.from_variant("vit_huge", **kw))


def vjepa2_vit_giant(**kw) -> VJEPA2:
    return VJEPA2(VJEPA2Config.from_variant("vit_giant", **kw))


def vjepa2_vit_gigantic(**kw) -> VJEPA2:
    return VJEPA2(VJEPA2Config.from_variant("vit_gigantic", **kw))
