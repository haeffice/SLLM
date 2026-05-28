"""Standalone Causal-JEPA module (PyTorch 2.8).

Self-contained reimplementation of **Causal-JEPA** (*Causal-JEPA: Learning
World Models through Object-Level Latent Interventions*, arXiv:2602.11389,
2026). Causal-JEPA moves masked joint-embedding prediction from image
*patches* to **object slots**: each frame is encoded into a small set of
object-centric latent slots, and the JEPA predictor must infer **masked
whole objects** from the remaining objects. Masking a full object is a
*latent intervention* — its current state is removed while its identity is
preserved — which provably forces the predictor to reason about
inter-object interactions (the paper's causal inductive bias) instead of
taking per-patch shortcuts.

    S_{1:T}    = SlotEnc(frames)                         # object slots (B,T,N,d)
    Z~_t^i     = proj(S_{t0}^i) + mask_token + e_t       # identity anchor for masked objects
    Ŝ_{1:T}    = Pred( visible slots ∪ anchors )         # ViT over the slot sequence
    L_pred     = ‖Ŝ - sg(S)‖²  over masked-history ∪ future positions
    L_recon    = ‖Dec(S) - frames‖²                      # slots stay object-like (self-contained)

Because every other directory in `paper/JEPA` is self-contained and
CPU-smoke-testable (no external DINOv2/VideoSAUR download), the frozen
object-centric encoder `g` of the paper is realised here as a **trainable
Slot Attention** (Locatello et al. 2020) auto-encoder: a spatial-broadcast
decoder reconstruction loss makes the slots object-like from scratch, and
`freeze_encoder` + `init_from` reproduce the paper's exact frozen-`g`
protocol when a pretrained slot encoder is available.

Inference (`from_checkpoint` → `eval()`): `encode_slots(video)` and
`rollout(history_video)` for latent-space planning / MPC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("CausalJEPA")


# =============================================================================
# Soft positional embedding (shared by slot encoder + broadcast decoder)
# =============================================================================

def _build_grid(resolution: int) -> torch.Tensor:
    """(1, res, res, 4) normalised coordinate grid [x, y, 1-x, 1-y]."""
    r = torch.linspace(0.0, 1.0, resolution)
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    grid = torch.stack([xx, yy, 1.0 - xx, 1.0 - yy], dim=-1)   # (res, res, 4)
    return grid.unsqueeze(0)


class SoftPositionEmbed(nn.Module):
    """Add a learned linear projection of a coordinate grid to a feature map."""

    def __init__(self, dim: int, resolution: int):
        super().__init__()
        self.proj = nn.Linear(4, dim)
        self.register_buffer("grid", _build_grid(resolution), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim, res, res); grid: (1, res, res, 4)
        pos = self.proj(self.grid).permute(0, 3, 1, 2)         # (1, dim, res, res)
        return x + pos


# =============================================================================
# Slot Attention (Locatello et al. 2020)
# =============================================================================

class SlotAttention(nn.Module):
    """Iterative object-centric grouping: `num_slots` slots compete to explain
    the input feature tokens. Returns (B, num_slots, slot_dim)."""

    def __init__(self, num_slots: int, slot_dim: int, input_dim: int,
                 iters: int = 3, hidden_dim: int = 128, eps: float = 1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.slots_mu = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.trunc_normal_(self.slots_mu, std=0.02)
        nn.init.trunc_normal_(self.slots_logsigma, std=0.02)

        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(input_dim, slot_dim)
        self.to_v = nn.Linear(input_dim, slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, slot_dim))
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, _, _ = inputs.shape
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots)) * self.scale
            attn_logits = torch.einsum("bid,bjd->bij", q, k)   # (B, slots, N)
            attn = attn_logits.softmax(dim=1) + self.eps       # compete over slots
            attn = attn / attn.sum(dim=2, keepdim=True)        # mean over inputs
            updates = torch.einsum("bij,bjd->bid", attn, v)    # (B, slots, d)
            slots = self.gru(updates.reshape(-1, self.slot_dim),
                             slots_prev.reshape(-1, self.slot_dim))
            slots = slots.reshape(B, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots


# =============================================================================
# CNN frame encoder + spatial-broadcast decoder
# =============================================================================

class FrameEncoder(nn.Module):
    """Per-frame CNN (/8 downsample) + soft position embedding -> tokens."""

    def __init__(self, in_chans: int, img_size: int, channels: tuple):
        super().__init__()
        assert img_size % 8 == 0, "img_size must be divisible by 8"
        chs = [in_chans] + list(channels)
        strides = [2, 2, 2, 1]                  # /8 spatial downsample
        assert len(channels) == 4, "expect 4 encoder channel widths"
        layers: list[nn.Module] = []
        for i in range(4):
            layers += [nn.Conv2d(chs[i], chs[i + 1], 5, stride=strides[i],
                                 padding=2), nn.GELU()]
        self.cnn = nn.Sequential(*layers)
        self.enc_dim = channels[-1]
        self.feat_size = img_size // 8
        self.pos = SoftPositionEmbed(self.enc_dim, self.feat_size)
        self.norm = nn.LayerNorm(self.enc_dim)
        self.mlp = nn.Sequential(nn.Linear(self.enc_dim, self.enc_dim), nn.GELU(),
                                 nn.Linear(self.enc_dim, self.enc_dim))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (M, C, H, W) -> tokens (M, feat_size**2, enc_dim)."""
        x = self.cnn(frames)                    # (M, enc_dim, h, w)
        x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)        # (M, h*w, enc_dim)
        return self.mlp(self.norm(x))


class BroadcastDecoder(nn.Module):
    """Spatial-broadcast decoder: each slot -> RGBA over the grid; alpha-
    composited reconstruction. Provides the (self-contained) object-learning
    signal so slots are meaningful without external pretrained weights."""

    def __init__(self, slot_dim: int, out_chans: int, resolution: int,
                 hidden: int = 64):
        super().__init__()
        self.resolution = resolution
        self.out_chans = out_chans
        self.pos = SoftPositionEmbed(slot_dim, resolution)
        self.net = nn.Sequential(
            nn.Conv2d(slot_dim, hidden, 5, padding=2), nn.GELU(),
            nn.Conv2d(hidden, hidden, 5, padding=2), nn.GELU(),
            nn.Conv2d(hidden, hidden, 5, padding=2), nn.GELU(),
            nn.Conv2d(hidden, out_chans + 1, 3, padding=1),
        )

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """slots (B, K, slot_dim) -> (recon (B, C, R, R), masks (B, K, 1, R, R))."""
        B, K, D = slots.shape
        R = self.resolution
        x = slots.reshape(B * K, D, 1, 1).expand(-1, -1, R, R)
        x = self.pos(x)
        out = self.net(x)                       # (B*K, C+1, R, R)
        out = out.reshape(B, K, self.out_chans + 1, R, R)
        rgb, alpha = out[:, :, :self.out_chans], out[:, :, self.out_chans:]
        masks = alpha.softmax(dim=1)            # over slots
        recon = (rgb.sigmoid() * masks).sum(dim=1)
        return recon, masks


# =============================================================================
# Predictor — ViT over the (object-masked) slot sequence
# =============================================================================

class _SelfBlock(nn.Module):
    """Pre-norm self-attention transformer block."""

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


class SlotPredictor(nn.Module):
    """Masked transformer over T*N slot tokens. Visible slots enter as
    `proj(slot)`; masked slots enter as the **identity anchor**
    `proj(slot@t0) + mask_token` (state removed, identity kept). Temporal and
    slot-identity embeddings are added to every token."""

    def __init__(self, slot_dim: int, num_slots: int, num_frames: int,
                 dim: int, depth: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.proj = nn.Linear(slot_dim, dim)
        self.out = nn.Linear(dim, slot_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.temporal_emb = nn.Embedding(num_frames, dim)
        self.slot_id_emb = nn.Embedding(num_slots, dim)
        self.blocks = nn.ModuleList(
            [_SelfBlock(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, slots: torch.Tensor, mask: torch.Tensor,
                anchor: torch.Tensor) -> torch.Tensor:
        """slots (B,T,N,Ds); mask (B,T,N) bool (True=masked); anchor (B,N,Ds)
        identity slots at t0. Returns predicted slots (B,T,N,Ds)."""
        B, T, N, _ = slots.shape
        content = self.proj(slots)                              # (B,T,N,D)
        anchored = self.proj(anchor).unsqueeze(1).expand(-1, T, -1, -1)
        m = mask.unsqueeze(-1)
        tok = torch.where(m, anchored + self.mask_token, content)

        t_idx = torch.arange(T, device=slots.device)
        s_idx = torch.arange(N, device=slots.device)
        tok = tok + self.temporal_emb(t_idx).view(1, T, 1, -1)
        tok = tok + self.slot_id_emb(s_idx).view(1, 1, N, -1)

        tok = tok.reshape(B, T * N, -1)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)
        return self.out(tok).reshape(B, T, N, -1)


# =============================================================================
# Config
# =============================================================================

@dataclass
class CausalJEPAConfig:
    in_chans: int = 3
    img_size: int = 64
    num_frames: int = 16                 # window length T
    history_len: int = 6                 # H; frames [H, T) are the future

    # object-centric encoder / decoder
    enc_channels: tuple = (32, 64, 64, 64)
    num_slots: int = 7
    slot_dim: int = 128
    slot_iters: int = 3
    slot_hidden: int = 128
    dec_hidden: int = 64

    # slot predictor (ViT)
    pred_dim: int = 256
    pred_depth: int = 6
    pred_heads: int = 8
    pred_mlp_ratio: float = 4.0

    # object-level masking (latent intervention budget per history frame)
    max_masked_slots: int = 4

    # loss weights
    recon_weight: float = 1.0
    history_weight: float = 1.0
    future_weight: float = 1.0

    # reproduce the paper's frozen-g protocol when a pretrained encoder exists
    freeze_encoder: bool = False

    def __post_init__(self):
        if isinstance(self.enc_channels, list):
            self.enc_channels = tuple(self.enc_channels)


# =============================================================================
# Model
# =============================================================================

class CausalJEPA(nn.Module):
    """Object-centric JEPA: Slot-Attention encoder + broadcast decoder +
    masked slot predictor. `forward(video)` returns the training losses;
    `encode_slots` / `rollout` are the inference entry points."""

    def __init__(self, config: Optional[CausalJEPAConfig] = None):
        super().__init__()
        cfg = config or CausalJEPAConfig()
        self.config = cfg

        self.encoder = FrameEncoder(cfg.in_chans, cfg.img_size, cfg.enc_channels)
        self.slot_attn = SlotAttention(
            cfg.num_slots, cfg.slot_dim, self.encoder.enc_dim,
            iters=cfg.slot_iters, hidden_dim=cfg.slot_hidden)
        self.decoder = BroadcastDecoder(
            cfg.slot_dim, cfg.in_chans, cfg.img_size, hidden=cfg.dec_hidden)
        self.predictor = SlotPredictor(
            cfg.slot_dim, cfg.num_slots, cfg.num_frames, cfg.pred_dim,
            cfg.pred_depth, cfg.pred_heads, cfg.pred_mlp_ratio)

        if cfg.freeze_encoder:
            for mod in (self.encoder, self.slot_attn, self.decoder):
                for p in mod.parameters():
                    p.requires_grad_(False)

    # -------------------------------------------------------------------------
    # Slot encoding / reconstruction
    # -------------------------------------------------------------------------

    def encode_slots(self, video: torch.Tensor) -> torch.Tensor:
        """video (B, T, C, H, W) in [0, 1] -> slots (B, T, N, slot_dim)."""
        B, T, C, H, W = video.shape
        tokens = self.encoder(video.reshape(B * T, C, H, W))
        slots = self.slot_attn(tokens)                          # (B*T, N, d)
        return slots.reshape(B, T, self.config.num_slots, self.config.slot_dim)

    def reconstruct(self, slots: torch.Tensor) -> torch.Tensor:
        """slots (B, T, N, d) -> reconstructed frames (B, T, C, H, W)."""
        B, T, N, d = slots.shape
        recon, _ = self.decoder(slots.reshape(B * T, N, d))
        return recon.reshape(B, T, self.config.in_chans,
                             self.config.img_size, self.config.img_size)

    # -------------------------------------------------------------------------
    # Object-level masking (latent intervention)
    # -------------------------------------------------------------------------

    def _sample_mask(self, B: int, device: torch.device) -> torch.Tensor:
        cfg = self.config
        T, N, H = cfg.num_frames, cfg.num_slots, cfg.history_len
        mask = torch.zeros(B, T, N, dtype=torch.bool, device=device)
        mask[:, H:, :] = True                                   # predict the future
        # History frames: intervene on 0..max_masked_slots random whole objects.
        k_max = min(cfg.max_masked_slots, N)
        for b in range(B):
            for t in range(H):
                k = int(torch.randint(0, k_max + 1, (1,), device=device).item())
                if k > 0:
                    sel = torch.randperm(N, device=device)[:k]
                    mask[b, t, sel] = True
        return mask

    # -------------------------------------------------------------------------
    # Training objective
    # -------------------------------------------------------------------------

    def forward(self, video: torch.Tensor) -> dict:
        cfg = self.config
        slots = self.encode_slots(video)                        # (B,T,N,d)
        B, T, N, _ = slots.shape

        # (a) reconstruction aux loss — keeps slots object-like (self-contained).
        recon = self.reconstruct(slots)
        recon_loss = F.mse_loss(recon, video)

        # (b) object-level masked prediction (latent intervention).
        mask = self._sample_mask(B, video.device)               # (B,T,N)
        anchor = slots[:, 0]                                     # identity anchor @ t0
        pred = self.predictor(slots, mask, anchor)              # (B,T,N,d)
        target = slots.detach()                                 # stop-grad targets
        err = ((pred - target) ** 2).mean(-1)                   # (B,T,N)

        is_future = (torch.arange(T, device=video.device) >= cfg.history_len)
        is_future = is_future.view(1, T, 1).float()
        base_w = is_future * cfg.future_weight + (1.0 - is_future) * cfg.history_weight
        w = mask.float() * base_w                               # (B,T,N)
        pred_loss = (err * w).sum() / w.sum().clamp_min(1.0)

        loss = pred_loss + cfg.recon_weight * recon_loss
        with torch.no_grad():
            hist_m = mask & (is_future == 0)
            fut_m = mask & (is_future == 1)
            history_loss = (err * hist_m).sum() / hist_m.sum().clamp_min(1)
            future_loss = (err * fut_m).sum() / fut_m.sum().clamp_min(1)
        return {
            "loss": loss,
            "pred_loss": pred_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "history_loss": history_loss.detach(),
            "future_loss": future_loss.detach(),
        }

    # -------------------------------------------------------------------------
    # Inference — latent rollout / MPC
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def rollout(self, history_video: torch.Tensor) -> torch.Tensor:
        """Predict future slots from a history clip.

        history_video (B, H, C, H, W) with H = `config.history_len`.
        Returns predicted future slots (B, T-H, N, slot_dim).
        """
        cfg = self.config
        slots_hist = self.encode_slots(history_video)           # (B,H,N,d)
        B, H, N, d = slots_hist.shape
        T = cfg.num_frames
        pad = slots_hist.new_zeros(B, T - H, N, d)
        slots = torch.cat([slots_hist, pad], dim=1)             # (B,T,N,d)
        mask = torch.zeros(B, T, N, dtype=torch.bool, device=slots.device)
        mask[:, H:, :] = True
        pred = self.predictor(slots, mask, slots_hist[:, 0])
        return pred[:, H:]                                      # (B, T-H, N, d)

    @torch.inference_mode()
    def goal_distance(self, history_video: torch.Tensor,
                      goal_frame: torch.Tensor) -> torch.Tensor:
        """MPC-style latent goal distance ‖Ŝ_T − S_goal‖² (mean over slots).

        history_video (B, H, C, H, W); goal_frame (B, C, H, W).
        """
        pred_future = self.rollout(history_video)               # (B, T-H, N, d)
        s_goal = self.encode_slots(goal_frame.unsqueeze(1))[:, 0]   # (B, N, d)
        s_pred = pred_future[:, -1]                             # (B, N, d)
        return ((s_pred - s_goal) ** 2).mean(dim=(-1, -2))      # (B,)

    # -------------------------------------------------------------------------
    # Checkpoint loading
    # -------------------------------------------------------------------------

    _STRIP_PREFIXES = ("module.", "_orig_mod.", "model.", "cjepa.")

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
            out[k] = v
        return out

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        config: Optional[CausalJEPAConfig] = None,
        map_location: str = "cpu",
        device: Optional[torch.device] = None,
        strict: bool = False,
    ) -> "CausalJEPA":
        """Build a `CausalJEPA` and load weights. Aborts (per paper/CLAUDE.md)
        if not a single tensor loads."""
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
                f"CausalJEPA.from_checkpoint({path!r}): 0/{total} keys applied "
                "— checkpoint/architecture mismatch, aborting.")
        if not missing and not unexpected:
            logger.info("CausalJEPA loaded successfully (%d/%d keys)", applied, total)
        else:
            logger.warning(
                "CausalJEPA partial load: %d/%d keys (missing=%d, unexpected=%d)",
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
