"""BatVision — depth-map (space map) prediction from binaural echoes.

Re-implementation of the audio-only depth-prediction baseline of

    Jesper H. Brunetto, Sascha Hornauer, Stella X. Yu, Fabien Moutarde,
    "The Audio-Visual BatVision Dataset for Research on Sight and Sound",
    IROS 2023.  arXiv:2303.07257  —  https://arxiv.org/abs/2303.07257

and the earlier

    Jesper Haahr Christensen, Sascha Hornauer, Stella X. Yu,
    "BatVision: Learning to See 3D Spatial Layout with Two Ears",
    ICRA 2020.  arXiv:1912.07011  —  https://arxiv.org/abs/1912.07011

BatVision listens with two ears: a loudspeaker emits a short chirp and the two
microphones record the **binaural echoes**. A network turns those echoes into a
**depth map** of the field of view ahead — a dense *space map* that resolves
walls, hallways, door openings and roughly outlined furniture in azimuth,
elevation and distance. This is the audio→space-map counterpart to EchoScan:
real recordings (not simulated RIRs) and a forward-facing depth image (not a
top-down floorplan).

Pipeline (audio-only baseline, Sec. of the dataset paper):

    binaural waveform  X in R^{2 x T}      (2 ch, 44.1 kHz, ~72 ms (V1) / 0.45 s (V2))
      └─ magnitude spectrogram per ch       STFT n_fft=512, win=64, hop=16, |.|^1
            → S in R^{2 x F x Ts}            resized to 2 x 256 x 256
      └─ U-Net generator (pix2pix unet_256) 8 down / 8 up, skip connections
            → D in R^{1 x 256 x 256}         sigmoid → normalised depth in [0,1]
    metres            depth_m = D * max_depth (12 m for V1, 30 m for V2)

This module holds the **inference** model + losses/metrics. Training lives in
`train_batvision.py` (transformers `Trainer`); evaluation in
`eval_batvision.py`. The U-Net follows the authors' baseline, which reuses the
pix2pix `UnetGenerator` (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix);
official BatVision code: https://github.com/AmandineBtto/Batvision-Dataset .
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Dataset / preprocessing defaults (BatVision dataset paper). Downstream reads these.
# -----------------------------------------------------------------------------
SAMPLE_RATE = 44100          # Hz, both BatVision V1 and V2
N_CHANNELS = 2               # binaural (left, right)
IMAGE_SIZE = 256             # depth map + spectrogram resized to 256x256 (unet_256)

# STFT parameters from the original BatVision paper (used by the dataset repo).
N_FFT = 512
WIN_LENGTH = 64
HOP_LENGTH = 16
SPEC_POWER = 1.0             # magnitude spectrogram (no dB)

MAX_DEPTH_V1 = 12.0          # m — BatVision V1 (UC Berkeley office)
MAX_DEPTH_V2 = 30.0          # m — BatVision V2 (Mines Paris campus)


# =============================================================================
# Audio front end — binaural magnitude spectrogram (torch.stft based)
# =============================================================================

def binaural_spectrogram(waveform: torch.Tensor, n_fft: int = N_FFT,
                         win_length: int = WIN_LENGTH, hop_length: int = HOP_LENGTH,
                         power: float = SPEC_POWER) -> torch.Tensor:
    """(2, T) or (B, 2, T) waveform → magnitude spectrogram (..., 2, F, Ts).

    Equivalent to ``torchaudio.transforms.Spectrogram(n_fft, win_length,
    hop_length, power)`` (Hann window, ``center=True``, reflect padding) but
    implemented with ``torch.stft`` so torchaudio is not a hard dependency.
    """
    squeeze = waveform.dim() == 2
    if squeeze:
        waveform = waveform.unsqueeze(0)                  # (1, 2, T)
    b, c, t = waveform.shape
    window = torch.hann_window(win_length, device=waveform.device, dtype=waveform.dtype)
    spec = torch.stft(
        waveform.reshape(b * c, t), n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=window, center=True,
        pad_mode="reflect", normalized=False, return_complex=True,
    ).abs()                                               # (b*c, F, Ts)
    if power != 1.0:
        spec = spec.pow(power)
    spec = spec.reshape(b, c, spec.shape[-2], spec.shape[-1])
    return spec.squeeze(0) if squeeze else spec


def resize_2d(x: torch.Tensor, size: int = IMAGE_SIZE, mode: str = "bilinear") -> torch.Tensor:
    """Resize the last two dims of a (C, H, W) / (B, C, H, W) tensor to size×size."""
    squeeze = x.dim() == 3
    if squeeze:
        x = x.unsqueeze(0)
    kw = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    x = F.interpolate(x, size=(size, size), mode=mode, **kw)
    return x.squeeze(0) if squeeze else x


# =============================================================================
# pix2pix U-Net generator (the BatVision audio-only baseline backbone)
#   ported from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# =============================================================================

def _norm_layer(norm: str):
    if norm == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    raise ValueError(f"unknown norm: {norm}")


class UnetSkipConnectionBlock(nn.Module):
    """One U-Net level: downsample → submodule → upsample, with a skip concat.

        x ─────────────────── identity ───────────────────┐
        └─ downconv ── |submodule| ── upconv ──────────────┴─ cat
    """

    def __init__(self, outer_nc: int, inner_nc: int, input_nc: int | None = None,
                 submodule: "UnetSkipConnectionBlock | None" = None,
                 outermost: bool = False, innermost: bool = False,
                 norm_layer=nn.BatchNorm2d, use_dropout: bool = False,
                 final_sigmoid: bool = True):
        super().__init__()
        self.outermost = outermost
        use_bias = (norm_layer.func == nn.InstanceNorm2d
                    if isinstance(norm_layer, functools.partial)
                    else norm_layer == nn.InstanceNorm2d)
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            # sigmoid → normalised depth in [0,1]; ReLU → raw metres (depth_norm off)
            up = [uprelu, upconv, nn.Sigmoid() if final_sigmoid else nn.ReLU()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up + ([nn.Dropout(0.5)] if use_dropout else [])
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)           # skip connection


class UnetGenerator(nn.Module):
    """U-Net built recursively from the innermost level out (pix2pix)."""

    def __init__(self, input_nc: int, output_nc: int, num_downs: int, ngf: int = 64,
                 norm_layer=nn.BatchNorm2d, use_dropout: bool = False,
                 final_sigmoid: bool = True):
        super().__init__()
        block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                        norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):                    # innermost ngf*8 levels
            block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=block,
                                            norm_layer=norm_layer, use_dropout=use_dropout)
        block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=block,
                                             outermost=True, norm_layer=norm_layer,
                                             final_sigmoid=final_sigmoid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# =============================================================================
# Full model
# =============================================================================

_GEN_DOWNS = {"unet_256": 8, "unet_128": 7}
_GEN_SIZE = {"unet_256": 256, "unet_128": 128}


class BatVision(nn.Module):
    """Binaural spectrogram (B, 2, S, S) → normalised depth map (B, 1, S, S).

    `forward` returns the (already-activated) normalised depth in [0,1] when
    ``depth_norm`` is set (the default), matching the dataset's MinMax target;
    `predict` rescales it to metres via ``max_depth``.
    """

    def __init__(self, generator: str = "unet_256", ngf: int = 64,
                 in_channels: int = N_CHANNELS, norm: str = "batch",
                 depth_norm: bool = True, max_depth: float = MAX_DEPTH_V1):
        super().__init__()
        if generator not in _GEN_DOWNS:
            raise ValueError(f"generator must be one of {list(_GEN_DOWNS)}, got {generator}")
        self.generator = generator
        self.image_size = _GEN_SIZE[generator]
        self.depth_norm = depth_norm
        self.max_depth = float(max_depth)
        self.net = UnetGenerator(
            input_nc=in_channels, output_nc=1, num_downs=_GEN_DOWNS[generator],
            ngf=ngf, norm_layer=_norm_layer(norm), use_dropout=False,
            final_sigmoid=depth_norm,
        )
        self.net.apply(self._init_weights)               # pix2pix normal(0, 0.02) init

    @staticmethod
    def _init_weights(m: nn.Module, gain: float = 0.02):
        cls = m.__class__.__name__
        if hasattr(m, "weight") and (cls.find("Conv") != -1 or cls.find("Linear") != -1):
            nn.init.normal_(m.weight.data, 0.0, gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif cls.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.net(spectrogram)                     # (B, 1, S, S), [0,1] if depth_norm

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def predict(self, spectrogram: torch.Tensor) -> dict:
        """eval-mode inference → normalised + metric depth maps. Always call
        after `from_checkpoint` (which freezes params); we re-assert eval()."""
        self.eval()
        norm = self.forward(spectrogram)                 # (B, 1, S, S)
        depth_m = norm * self.max_depth if self.depth_norm else norm
        return {"depth_norm": norm, "depth_m": depth_m}

    # -- checkpoints ----------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu", **model_kwargs) -> "BatVision":
        """Load weights from a `.pt` dump (raw state_dict or {'state_dict':...,
        'model_kwargs':...}). Aborts loudly if nothing matched, so a silent
        all-random model can never reach inference."""
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(blob, dict) and "state_dict" in blob:
            sd = blob["state_dict"]
            model_kwargs = {**blob.get("model_kwargs", {}), **model_kwargs}
        else:
            sd = blob
        sd = {k[len("batvision."):] if k.startswith("batvision.") else k: v
              for k, v in sd.items()}
        model = cls(**model_kwargs)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        applied = len(sd) - len(unexpected)
        if applied == 0:
            raise RuntimeError(
                f"from_checkpoint({path!r}): 0/{len(sd)} tensors matched — "
                f"checkpoint/architecture mismatch, refusing to run.")
        print(f"[BatVision] loaded {applied}/{len(sd)} tensors from {path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model


# =============================================================================
# Loss — masked L1 on valid (non-zero) depth pixels (BatVision baseline)
# =============================================================================

def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 over pixels where the ground-truth depth is valid (> 0). Zero-depth
    pixels are sensor holes / clipped sky and are excluded, exactly as in the
    reference training loop."""
    mask = target != 0
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.l1_loss(pred[mask], target[mask])


# =============================================================================
# Metrics (evaluation) — depths in METRES. Port of the BatVision repo's
# compute_errors (originally from "Beyond Image to Depth").
# =============================================================================

def compute_errors(gt: torch.Tensor, pred: torch.Tensor) -> dict:
    """Standard monocular-depth error metrics over valid (gt>0) pixels.

    Returns abs_rel, rmse, delta1/2/3 (a1/a2/a3), log10, mae as Python floats.
    `gt`/`pred` are metric-depth tensors of identical shape.
    """
    gt = gt.flatten().float()
    pred = pred.flatten().float()
    mask = gt > 0
    gt, pred = gt[mask], pred[mask]
    if gt.numel() == 0:
        return {k: 0.0 for k in ("abs_rel", "rmse", "delta1", "delta2", "delta3", "log10", "mae")}
    pred = pred.clamp(min=1e-6)

    thresh = torch.maximum(gt / pred, pred / gt)
    out = {
        "delta1": (thresh < 1.25).float().mean().item(),
        "delta2": (thresh < 1.25 ** 2).float().mean().item(),
        "delta3": (thresh < 1.25 ** 3).float().mean().item(),
        "rmse": torch.sqrt(((gt - pred) ** 2).mean()).item(),
        "abs_rel": (torch.abs(gt - pred) / gt).mean().item(),
        "log10": torch.abs(torch.log10(gt) - torch.log10(pred)).mean().item(),
        "mae": torch.abs(gt - pred).mean().item(),
    }
    return {k: (0.0 if math.isnan(v) else v) for k, v in out.items()}
