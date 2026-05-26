"""PhaseCoder — microphone-geometry-agnostic spatial audio encoder.

Paper-faithful re-implementation of

    Artem Dementyev, Wazeer Zulfikar, Sinan Hersek, Pascal Getreuer, et al.,
    "PhaseCoder: Microphone Geometry-Agnostic Spatial Audio Understanding for
     Multimodal LLMs", Google DeepMind & Google AR, 2026.
    arXiv:2601.21124  —  https://arxiv.org/abs/2601.21124

PhaseCoder turns *raw multichannel audio + the 3-D coordinates of each
microphone* into a compact, microphone-independent **spatial audio token**.
Because the array geometry enters only through a positional encoding (not the
weights), one trained model serves arrays with **any number / layout of mics**
(the paper uses 3–8 mics, 7–18 cm aperture). The token feeds two uses:

    1. localization heads — joint azimuth / elevation / distance classification;
    2. a projector that injects the tokens into a multimodal LLM (Gemma 3n) for
       spatial reasoning and direction-targeted transcription.

Pipeline (Sec. 3 of the paper):

    waveform  X in R^{C x T}   +  mic coords  M in R^{C x 3}     C mics, 16 kHz
      └─ STFT per channel        Hann win=256, hop=128 → 129 bins, F frames
      └─ patch features          per (mic, frame): [|S| ‖ angle(S)] (258) → Linear → D=256
      └─ + positional encodings  sequential 1D ⊕ frame-level ⊕ MPE(mic spherical)
      └─ [CLS] + L=C·F patches → Transformer (5 blocks, 4 heads, D=256, FFN 256)
      └─ CLS → MLP(256→256→256)  = spatial audio token  z in R^{256}
      ├─ heads  azimuth / elevation / distance  (softmax classification)
      └─ projector  z → R^{2048}  (2-layer GELU MLP, for the LLM)

This module is the **inference** model + losses/metrics. Training lives in
`train_phasecoder.py` (transformers `Trainer`); evaluation in
`eval_phasecoder.py`. No official code release was found at the time of
writing, so the network follows the paper's text/equations; the microphone
positional encoding (MPE) reuses GI-DOAENet (Bohlender et al., IEEE 2025,
α=7.0, β=4.0). Under-specified details (projection LayerNorm, pre/post-norm)
are chosen conventionally and flagged in the README.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Paper defaults (Sec. 3 / 4). Everything downstream reads these.
# -----------------------------------------------------------------------------
SAMPLE_RATE = 16_000        # Hz
CLIP_SAMPLES = 4_096        # ~256 ms → F=33 STFT frames (paper: "250 ms, 33 frames")
N_FFT = 256                 # Hann window 256 samples
HOP = 128                   # 50 % overlap
N_FREQ = N_FFT // 2 + 1     # 129 bins (real STFT)

EMBED_DIM = 256             # D
N_BLOCKS = 5                # transformer blocks
N_HEADS = 4                 # self-attention heads
FFN_DIM = 256               # 1× expansion (paper)

MPE_ALPHA = 7.0             # GI-DOAENet positional-encoding constants
MPE_BETA = 4.0

# Classification heads. Last index of each head = "no speech / no source".
# Counts match the paper (azimuth 38, elevation 18, distance 13).
N_AZIMUTH = 38              # 37 angle bins over [0,360) + 1 no-source
N_ELEVATION = 18            # 17 angle bins over [-90,90] + 1 no-source
N_DISTANCE = 13             # 12 distance bins + 1 no-source
DIST_MIN, DIST_MAX = 0.5, 6.5   # m, spanned by the 12 distance bins

LLM_DIM = 2048              # Gemma 3n hidden size (projector output)


# =============================================================================
# Bin <-> value helpers (shared by dataset, training and eval)
# =============================================================================

def azimuth_bin_centers(n_classes: int = N_AZIMUTH) -> torch.Tensor:
    """Centres (degrees) of the (n_classes-1) azimuth bins over [0, 360)."""
    k = n_classes - 1
    return torch.arange(k, dtype=torch.float32) * (360.0 / k)


def elevation_bin_centers(n_classes: int = N_ELEVATION) -> torch.Tensor:
    """Centres (degrees) of the (n_classes-1) elevation bins over [-90, 90]."""
    k = n_classes - 1
    return torch.linspace(-90.0, 90.0, k)


def distance_bin_centers(n_classes: int = N_DISTANCE,
                         dmin: float = DIST_MIN, dmax: float = DIST_MAX) -> torch.Tensor:
    """Centres (metres) of the (n_classes-1) distance bins over [dmin, dmax]."""
    k = n_classes - 1
    return torch.linspace(dmin, dmax, k)


def value_to_bin(value: float, centers: torch.Tensor, *, circular: bool = False) -> int:
    """Nearest-centre bin index for a scalar value (circular for azimuth)."""
    v = torch.tensor(float(value))
    if circular:
        diff = torch.remainder(centers - v + 180.0, 360.0) - 180.0
        return int(torch.argmin(diff.abs()).item())
    return int(torch.argmin((centers - v).abs()).item())


# =============================================================================
# STFT patch front end
# =============================================================================

class PatchExtractor(nn.Module):
    """Multichannel waveform → patch embeddings (B, C*F, D).

    Per (mic, frame) patch = [magnitude(129) ‖ phase(129)] → Linear(258→D).
    A LayerNorm follows the projection (under-specified in the paper; a
    standard ViT choice). Returns embeddings and the frame count F.
    """

    def __init__(self, n_fft: int = N_FFT, hop: int = HOP, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.n_freq = n_fft // 2 + 1
        self.proj = nn.Linear(2 * self.n_freq, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, int]:
        """audio (B, C, T) → (patches (B, C*F, D), F). Patches are channel-major:
        index = c*F + f."""
        b, c, t = audio.shape
        spec = torch.stft(
            audio.reshape(b * c, t), n_fft=self.n_fft, hop_length=self.hop,
            win_length=self.n_fft, window=self.window.to(audio.dtype),
            center=True, pad_mode="reflect", return_complex=True,
        )                                                   # (b*c, n_freq, F)
        f = spec.shape[-1]
        mag = spec.abs()                                    # (b*c, n_freq, F)
        phase = torch.angle(spec)
        feat = torch.cat([mag, phase], dim=1)               # (b*c, 2*n_freq, F)
        feat = feat.permute(0, 2, 1).reshape(b, c * f, 2 * self.n_freq)  # channel-major
        return self.norm(self.proj(feat)), f


# =============================================================================
# Positional encodings
# =============================================================================

def sinusoidal_pe(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard 1-D sinusoidal positional encoding for arbitrary positions.

    positions (N,) → (N, dim). Even dims = sin, odd = cos (Transformer paper).
    """
    device = positions.device
    half = dim // 2
    div = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / max(half, 1)))
    ang = positions.float()[:, None] * div[None, :]         # (N, half)
    pe = torch.zeros(positions.shape[0], dim, device=device)
    pe[:, 0::2] = torch.sin(ang)[:, : pe[:, 0::2].shape[1]]
    pe[:, 1::2] = torch.cos(ang)[:, : pe[:, 1::2].shape[1]]
    return pe


def microphone_positional_encoding(coords: torch.Tensor, dim: int = EMBED_DIM,
                                    alpha: float = MPE_ALPHA, beta: float = MPE_BETA,
                                    eps: float = 1e-8) -> torch.Tensor:
    """Geometry-aware mic encoding (GI-DOAENet MPE).

    coords (B, C, 3) Cartesian metres → (B, C, dim). Mic positions are taken
    relative to the array centroid and converted to spherical (r, elevation θ,
    azimuth φ):

        φ = atan2(y - c_y, x - c_x)              (azimuth)
        θ = atan2(z - c_z, hypot(x-c_x, y-c_y))  (elevation)

    P_i = α·r_i·[cos(2πβv+θ), sin(2πβv+θ), cos(2πβv+φ), sin(2πβv+φ)]
    with v = (4/D)·[0..D/4-1].  (paper Eq. for MPE; α=7, β=4)
    """
    centroid = coords.mean(dim=1, keepdim=True)             # (B, 1, 3)
    rel = coords - centroid                                 # (B, C, 3)
    x, y, z = rel[..., 0], rel[..., 1], rel[..., 2]
    r = torch.linalg.norm(rel, dim=-1)                      # (B, C)
    phi = torch.atan2(y, x)                                 # azimuth
    theta = torch.atan2(z, torch.hypot(x, y) + eps)         # elevation

    quarter = dim // 4
    v = (4.0 / dim) * torch.arange(quarter, device=coords.device, dtype=torch.float32)
    ang = 2.0 * math.pi * beta * v                          # (D/4,)
    t = theta[..., None] + ang                              # (B, C, D/4)
    p = phi[..., None] + ang
    enc = torch.cat([torch.cos(t), torch.sin(t), torch.cos(p), torch.sin(p)], dim=-1)
    return alpha * r[..., None] * enc                       # (B, C, dim)


# =============================================================================
# Encoder
# =============================================================================

class PhaseCoder(nn.Module):
    """Spatial audio encoder. `forward` returns localization logits + the
    spatial token; `predict` decodes az/el/distance. Geometry-agnostic: accepts
    any mic count C and frame count F (all positional encodings are computed on
    the fly; only the [CLS] token and weights are learned)."""

    def __init__(self, embed_dim: int = EMBED_DIM, n_blocks: int = N_BLOCKS,
                 n_heads: int = N_HEADS, ffn_dim: int = FFN_DIM,
                 n_fft: int = N_FFT, hop: int = HOP, dropout: float = 0.0,
                 n_azimuth: int = N_AZIMUTH, n_elevation: int = N_ELEVATION,
                 n_distance: int = N_DISTANCE,
                 dist_min: float = DIST_MIN, dist_max: float = DIST_MAX,
                 mpe_alpha: float = MPE_ALPHA, mpe_beta: float = MPE_BETA):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_azimuth, self.n_elevation, self.n_distance = n_azimuth, n_elevation, n_distance
        self.dist_min, self.dist_max = dist_min, dist_max
        self.mpe_alpha, self.mpe_beta = mpe_alpha, mpe_beta

        self.patch = PatchExtractor(n_fft=n_fft, hop=hop, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation=nn.GELU(), batch_first=True,
            norm_first=True,                                # pre-norm (stable ViT default)
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks,
                                             norm=nn.LayerNorm(embed_dim),
                                             enable_nested_tensor=False)

        # CLS → spatial audio token (2-layer MLP, paper Sec. 3).
        self.token_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Localization heads (single linear each).
        self.head_azimuth = nn.Linear(embed_dim, n_azimuth)
        self.head_elevation = nn.Linear(embed_dim, n_elevation)
        self.head_distance = nn.Linear(embed_dim, n_distance)

    # -- core -----------------------------------------------------------------
    def encode(self, audio: torch.Tensor, mic_coords: torch.Tensor,
               channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        """audio (B, C, T), mic_coords (B, C, 3), channel_mask (B, C) bool with
        True = valid mic (None = all valid) → spatial token (B, D)."""
        b, c, _ = audio.shape
        patches, f = self.patch(audio)                      # (B, C*F, D)

        # positional encodings (all additive)
        pos = torch.arange(c * f, device=audio.device)
        seq_pe = sinusoidal_pe(pos, self.embed_dim)                         # (C*F, D)
        frame_pe = sinusoidal_pe(torch.arange(f, device=audio.device), self.embed_dim)
        frame_pe = frame_pe.repeat(c, 1)                                    # channel-major
        mpe = microphone_positional_encoding(mic_coords, self.embed_dim,
                                              self.mpe_alpha, self.mpe_beta)  # (B, C, D)
        mpe = mpe.repeat_interleave(f, dim=1)                               # (B, C*F, D)
        patches = patches + seq_pe[None] + frame_pe[None] + mpe

        cls = self.cls_token.expand(b, 1, -1)
        x = torch.cat([cls, patches], dim=1)                # (B, 1+C*F, D)

        # padding mask: ignore patches of invalid (padded) mics; keep CLS.
        key_padding = None
        if channel_mask is not None:
            patch_mask = (~channel_mask).repeat_interleave(f, dim=1)        # (B, C*F) True=pad
            cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=audio.device)
            key_padding = torch.cat([cls_mask, patch_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=key_padding)
        return self.token_mlp(x[:, 0])                      # CLS → (B, D)

    def forward(self, audio: torch.Tensor, mic_coords: torch.Tensor,
                channel_mask: torch.Tensor | None = None) -> dict:
        token = self.encode(audio, mic_coords, channel_mask)
        return {
            "spatial_token": token,
            "azimuth_logits": self.head_azimuth(token),
            "elevation_logits": self.head_elevation(token),
            "distance_logits": self.head_distance(token),
        }

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def predict(self, audio: torch.Tensor, mic_coords: torch.Tensor,
                channel_mask: torch.Tensor | None = None) -> dict:
        """eval-mode inference → predicted bins + decoded az/el/distance values.
        Always call after `from_checkpoint` (which freezes params)."""
        self.eval()
        out = self.forward(audio, mic_coords, channel_mask)
        az = out["azimuth_logits"].argmax(-1)
        el = out["elevation_logits"].argmax(-1)
        di = out["distance_logits"].argmax(-1)
        az_c = azimuth_bin_centers(self.n_azimuth).to(audio.device)
        el_c = elevation_bin_centers(self.n_elevation).to(audio.device)
        di_c = distance_bin_centers(self.n_distance, self.dist_min, self.dist_max).to(audio.device)
        no_az, no_el, no_di = self.n_azimuth - 1, self.n_elevation - 1, self.n_distance - 1
        return {
            "spatial_token": out["spatial_token"],
            "azimuth_bin": az, "elevation_bin": el, "distance_bin": di,
            "no_source": (az == no_az),
            "azimuth_deg": torch.where(az == no_az, torch.full_like(az, -1, dtype=torch.float32),
                                       az_c[az.clamp(max=no_az - 1)]),
            "elevation_deg": torch.where(el == no_el, torch.full_like(el, -1, dtype=torch.float32),
                                         el_c[el.clamp(max=no_el - 1)]),
            "distance_m": torch.where(di == no_di, torch.full_like(di, -1, dtype=torch.float32),
                                      di_c[di.clamp(max=no_di - 1)]),
        }

    # -- checkpoints ----------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu", **model_kwargs) -> "PhaseCoder":
        """Load weights from a `.pt` dump (raw state_dict or {'state_dict':...,
        'model_kwargs':...}). Aborts loudly if nothing matched, so a silent
        all-random model can never reach inference."""
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(blob, dict) and "state_dict" in blob:
            sd = blob["state_dict"]
            model_kwargs = {**blob.get("model_kwargs", {}), **model_kwargs}
        else:
            sd = blob
        sd = {k[len("phasecoder."):] if k.startswith("phasecoder.") else k: v
              for k, v in sd.items()}
        model = cls(**model_kwargs)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        applied = len(sd) - len(unexpected)
        if applied == 0:
            raise RuntimeError(
                f"from_checkpoint({path!r}): 0/{len(sd)} tensors matched — "
                f"checkpoint/architecture mismatch, refusing to run.")
        print(f"[PhaseCoder] loaded {applied}/{len(sd)} tensors from {path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model


# =============================================================================
# LLM projector — spatial token → Gemma 3n hidden size (Sec. 4)
# =============================================================================

class SpatialTokenProjector(nn.Module):
    """2-layer GELU MLP, 256 → 2048 → 2048. Maps PhaseCoder spatial tokens into
    the LLM embedding space; prepended between [BSA]/[ESA] markers before the
    mono audio tokens (the LLM/LoRA fine-tuning itself is out of scope here)."""

    def __init__(self, in_dim: int = EMBED_DIM, out_dim: int = LLM_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


# =============================================================================
# Loss (Sec. 3): weighted multitask cross-entropy
# =============================================================================

def phasecoder_loss(out: dict, azimuth: torch.Tensor, elevation: torch.Tensor,
                    distance: torch.Tensor, lambda_az: float = 1.0,
                    lambda_el: float = 1.0, lambda_di: float = 0.5,
                    ) -> tuple[torch.Tensor, dict]:
    """L = λ_az·CE(az) + λ_el·CE(el) + λ_di·CE(dist).  (λ = 1.0, 1.0, 0.5)"""
    ce_az = F.cross_entropy(out["azimuth_logits"], azimuth)
    ce_el = F.cross_entropy(out["elevation_logits"], elevation)
    ce_di = F.cross_entropy(out["distance_logits"], distance)
    total = lambda_az * ce_az + lambda_el * ce_el + lambda_di * ce_di
    return total, {
        "loss_azimuth": ce_az.detach(),
        "loss_elevation": ce_el.detach(),
        "loss_distance": ce_di.detach(),
    }


# =============================================================================
# Metrics (evaluation)
# =============================================================================

def angular_error_deg(pred_deg: torch.Tensor, true_deg: torch.Tensor,
                      circular: bool = True) -> torch.Tensor:
    """Per-sample absolute angular error (degrees); wraps at 360° if circular.
    Ignores samples where either value is the no-source sentinel (-1)."""
    valid = (pred_deg >= 0) & (true_deg >= 0)
    diff = pred_deg - true_deg
    if circular:
        diff = torch.remainder(diff + 180.0, 360.0) - 180.0
    err = diff.abs()
    return err[valid]
