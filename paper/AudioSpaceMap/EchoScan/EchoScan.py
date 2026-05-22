"""EchoScan — room geometry (space map) inference from acoustic echoes.

Paper-faithful re-implementation of

    Inmo Yeon, Iljoo Jeong, Seungchul Lee, Jung-Woo Choi,
    "EchoScan: Scanning Complex Room Geometries via Acoustic Echoes",
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024.
    arXiv:2310.11728  —  https://arxiv.org/abs/2310.11728

EchoScan turns multi-channel room impulse responses (RIRs) — the acoustic
echoes captured by a small mic array next to a loudspeaker — into a *space
map*: a 2-D **floorplan** segmentation image plus a 1-D **height** profile.
The two are multiplied (outer product) to recover the full 3-D room volume,
so arbitrary (non-convex, curved) shapes are handled without assuming the
number of walls.

Architecture (Sec. III of the paper):

    RIR  X in R^{M x N}              (M=6 omni mics, N=1024 @ 8 kHz, direct sound removed)
      └─ Encoder (1-D ResNet)        stem Conv1d(k=9,/2) + 6 residual conv blocks
            → F in R^{C_L x D_L}     C_L=1024 channels, D_L=16 time steps
      └─ Multi-Aggregation (MA)      AvgPool(rho=1) ‖ GeM(rho=3) over time,
            → m in R^{512}             each → Linear(1024→256) + L2-norm, concat
      ├─ Floorplan decoder           reshape m→(2,16,16), k UpConv blocks (×2 each)
      │     → Y^LW in R^{b x b}        b=1024  (±10.24 m, 2 cm/pixel), sigmoid
      └─ Height decoder              single Linear
            → y^H in R^{h}             h=512   (±5.12 m, 2 cm/pixel), sigmoid
    3-D volume  Y^3D = Y^LW ⊗ y^H

This module is the **inference** model + loss helpers. Training lives in
`train_echoscan.py` (transformers `Trainer`); evaluation in
`eval_echoscan.py`. No official code release was found at the time of
writing, so the network follows the paper's textual/figure description; a
few under-specified details (decoder channel schedule, "projective skip
connections") are simplified and flagged in the README.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Paper defaults (Sec. III / IV). Everything downstream reads these.
# -----------------------------------------------------------------------------
SAMPLE_RATE = 8000          # Hz
N_MICS = 6                  # circular array channels
RIR_LENGTH = 1024           # samples (~128 ms ≈ 44 m sound travel)
ARRAY_RADIUS = 0.05         # m  (5 cm circular array, source at centre)

PIXEL_M = 0.02              # 2 cm / pixel for both maps
FLOORPLAN_SIZE = 1024       # b — floorplan is b×b, covers ±10.24 m
HEIGHT_SIZE = 512           # h — height vector, covers ±5.12 m

ENCODER_STEM_CH = 32        # channels after the stem conv
ENCODER_CHANNELS = (32, 64, 128, 256, 512, 1024)   # per residual block (C_L=1024)
DECODER_INIT = 16           # floorplan decoder starts from a 16×16 grid


# =============================================================================
# Encoder — 1-D ResNet over the M-channel RIR
# =============================================================================

class ResBlock1D(nn.Module):
    """ResNet basic block (1-D). Two Conv1d(k=5) + BN + ReLU with a residual
    add; the first conv carries the (optional) stride / channel change and the
    skip path is projected by a 1×1 conv when shape changes."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=stride,
                               padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)        # ReLU at the block exit → F ≥ 0


class EchoEncoder(nn.Module):
    """RIR (B, M, N) → latent F (B, C_L, D_L). C_L=1024, D_L=16 for N=1024."""

    def __init__(self, n_mics: int = N_MICS,
                 channels=ENCODER_CHANNELS, stem_ch: int = ENCODER_STEM_CH):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_mics, stem_ch, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(stem_ch),
            nn.ReLU(inplace=True),
        )
        blocks = []
        in_ch = stem_ch
        for i, out_ch in enumerate(channels):
            stride = 1 if i == 0 else 2          # first block keeps the length
            blocks.append(ResBlock1D(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))         # (B, C_L, D_L)


# =============================================================================
# Multi-Aggregation module — Avg (ρ=1) ‖ GeM (ρ=3) over the time axis
# =============================================================================

class MultiAggregation(nn.Module):
    """Compress F (B, C_L, D_L) → m (B, 2*proj_dim) by two generalised-mean
    poolings over time. ρ=1 is plain average pooling; ρ=3 is GeM, which
    emphasises the salient (high-order reflection) responses. Each pooled
    1024-vector is linearly reduced to `proj_dim` and ℓ2-normalised, then the
    two are concatenated (Eq. 1 in the paper)."""

    def __init__(self, in_ch: int, proj_dim: int = 256, gem_p: float = 3.0,
                 eps: float = 1e-6):
        super().__init__()
        self.gem_p = gem_p
        self.eps = eps
        self.fc_avg = nn.Linear(in_ch, proj_dim)
        self.fc_gem = nn.Linear(in_ch, proj_dim)
        self.out_dim = 2 * proj_dim

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        avg = f.mean(dim=2)                                      # ρ=1
        gem = f.clamp(min=self.eps).pow(self.gem_p).mean(dim=2).pow(1.0 / self.gem_p)
        a = F.normalize(self.fc_avg(avg), dim=1)
        g = F.normalize(self.fc_gem(gem), dim=1)
        return torch.cat([a, g], dim=1)                          # (B, 2*proj_dim)


# =============================================================================
# Decoders
# =============================================================================

class UpConvBlock(nn.Module):
    """Nearest-neighbour ×2 upsample + Conv + BN + ReLU (a UCB)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(self.up(x))))


class FloorplanDecoder(nn.Module):
    """m (B, D) → floorplan logits (B, 1, b, b). Reshape to a (2,16,16) grid
    then upsample ×2 `k` times where 16·2^k = b (k=6 for b=1024)."""

    def __init__(self, in_dim: int, out_size: int = FLOORPLAN_SIZE,
                 init_size: int = DECODER_INIT, init_ch: int = 2):
        super().__init__()
        if out_size % init_size != 0 or (out_size // init_size) & (out_size // init_size - 1):
            raise ValueError(
                f"floorplan_size ({out_size}) must be init_size ({init_size}) × 2^k"
            )
        self.init_size = init_size
        self.init_ch = init_ch
        self.fc = nn.Linear(in_dim, init_ch * init_size * init_size)
        n_up = int(round(math.log2(out_size // init_size)))
        widths = [max(256 >> i, 16) for i in range(n_up)]        # 256,128,64,32,16,16,...
        ucbs, ch = [], init_ch
        for w in widths:
            ucbs.append(UpConvBlock(ch, w))
            ch = w
        self.ucbs = nn.Sequential(*ucbs)
        self.head = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        x = self.fc(m).view(-1, self.init_ch, self.init_size, self.init_size)
        return self.head(self.ucbs(x))                           # (B, 1, b, b) logits


class HeightDecoder(nn.Module):
    """m (B, D) → height logits (B, h). Single linear layer (paper Sec. III)."""

    def __init__(self, in_dim: int, out_size: int = HEIGHT_SIZE):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_size)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.fc(m)                                        # (B, h) logits


# =============================================================================
# Full model
# =============================================================================

class EchoScan(nn.Module):
    """Encoder → Multi-Aggregation → {floorplan, height} decoders.

    `forward` returns raw logits; probabilities/binary maps are produced by
    `predict`. Keep `floorplan_size`/`height_size` in sync with the labels
    written by `make_echoscan_dataset.py`.
    """

    def __init__(self, n_mics: int = N_MICS, rir_length: int = RIR_LENGTH,
                 floorplan_size: int = FLOORPLAN_SIZE,
                 height_size: int = HEIGHT_SIZE, ma_proj_dim: int = 256,
                 decoder_init_size: int = DECODER_INIT):
        super().__init__()
        self.n_mics = n_mics
        self.rir_length = rir_length
        self.floorplan_size = floorplan_size
        self.height_size = height_size

        self.encoder = EchoEncoder(n_mics=n_mics)
        self.ma = MultiAggregation(self.encoder.out_channels, proj_dim=ma_proj_dim)
        self.floorplan_decoder = FloorplanDecoder(
            self.ma.out_dim, out_size=floorplan_size, init_size=decoder_init_size)
        self.height_decoder = HeightDecoder(self.ma.out_dim, out_size=height_size)

    def forward(self, rir: torch.Tensor) -> dict:
        m = self.ma(self.encoder(rir))
        return {
            "floorplan_logits": self.floorplan_decoder(m),       # (B, 1, b, b)
            "height_logits": self.height_decoder(m),             # (B, h)
        }

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def predict(self, rir: torch.Tensor, threshold: float = 0.5) -> dict:
        """eval-mode inference → probability + binary maps. Always call after
        `from_checkpoint` (which freezes params); we re-assert eval() here."""
        self.eval()
        out = self.forward(rir)
        fp_prob = torch.sigmoid(out["floorplan_logits"]).squeeze(1)   # (B, b, b)
        h_prob = torch.sigmoid(out["height_logits"])                  # (B, h)
        return {
            "floorplan_prob": fp_prob,
            "height_prob": h_prob,
            "floorplan": (fp_prob >= threshold).to(torch.uint8),
            "height": (h_prob >= threshold).to(torch.uint8),
        }

    @staticmethod
    def reconstruct_3d(floorplan: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
        """Binary maps → 3-D occupancy Y^3D = Y^LW ⊗ y^H.

        floorplan (b, b), height (h,) → (b, b, h) bool. Memory-heavy at full
        resolution (1024²×512 ≈ 0.5 G voxels) — intended for cropped/down-
        sampled volumes or visualisation only.
        """
        return (floorplan.bool()[..., None] & height.bool()[None, None, :])

    # -- checkpoints ----------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu", **model_kwargs) -> "EchoScan":
        """Load weights from a `.pt` dump (raw state_dict or {'state_dict':...,
        'model_kwargs':...}). Aborts loudly if nothing matched, so a silent
        all-random model can never reach inference."""
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(blob, dict) and "state_dict" in blob:
            sd = blob["state_dict"]
            model_kwargs = {**blob.get("model_kwargs", {}), **model_kwargs}
        else:
            sd = blob
        sd = {k[len("echoscan."):] if k.startswith("echoscan.") else k: v
              for k, v in sd.items()}
        model = cls(**model_kwargs)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        applied = len(sd) - len(unexpected)
        if applied == 0:
            raise RuntimeError(
                f"from_checkpoint({path!r}): 0/{len(sd)} tensors matched — "
                f"checkpoint/architecture mismatch, refusing to run.")
        print(f"[EchoScan] loaded {applied}/{len(sd)} tensors from {path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model


# =============================================================================
# Losses (Eq. 2–3): MSE + α·Dice on the floorplan, β·PIT-MSE on the height
# =============================================================================

def floorplan_losses(logits: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """logits/target (B, 1, b, b). Returns (MSE on sigmoid probs, soft Dice)."""
    prob = torch.sigmoid(logits)
    mse = F.mse_loss(prob, target)
    p = prob.flatten(1)
    y = target.flatten(1)
    inter = (p * y).sum(dim=1)
    dice = 1.0 - (2.0 * inter) / (p.sum(dim=1) + y.sum(dim=1) + eps)
    return mse, dice.mean()


def height_loss_pit(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant height MSE (B, h): the vertical orientation is
    ambiguous, so score the prediction against the target and its flip and
    keep the lower per-sample loss."""
    prob = torch.sigmoid(logits)
    mse_a = ((prob - target) ** 2).mean(dim=1)
    mse_b = ((prob - torch.flip(target, dims=[1])) ** 2).mean(dim=1)
    return torch.minimum(mse_a, mse_b).mean()


def echoscan_loss(out: dict, floorplan: torch.Tensor, height: torch.Tensor,
                  alpha: float = 0.3, beta: float = 1.0) -> tuple[torch.Tensor, dict]:
    """L_total = MSE(LW) + α·Dice(LW) + β·PIT-MSE(H)  (α=0.3, β=1.0)."""
    mse_lw, dice_lw = floorplan_losses(out["floorplan_logits"], floorplan)
    mse_h = height_loss_pit(out["height_logits"], height)
    total = mse_lw + alpha * dice_lw + beta * mse_h
    return total, {
        "loss_floorplan_mse": mse_lw.detach(),
        "loss_floorplan_dice": dice_lw.detach(),
        "loss_height_mse": mse_h.detach(),
    }


# =============================================================================
# Metrics (evaluation)
# =============================================================================

def iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary IoU over all but the batch dim → (B,)."""
    p = pred.flatten(1).bool()
    y = target.flatten(1).bool()
    inter = (p & y).sum(dim=1).float()
    union = (p | y).sum(dim=1).float()
    return inter / (union + eps)
