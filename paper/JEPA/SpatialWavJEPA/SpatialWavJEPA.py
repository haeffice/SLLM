"""Standalone Spatial WavJEPA inference module (PyTorch 2.8).

Self-contained reimplementation of the deployable half of **WavJEPA-Nat**,
the multi-channel / spatial variant of WavJEPA (arXiv:2509.23238, ref impl:
github.com/labhamlet/wavjepa). Only the student-side encoder used at
inference lives here — the EMA teacher, decoder, mask token, and predictor
live in `SpatialWavJEPA_Trainer.py`.

Architecture (WavJEPA-Nat, binaural):
    Two *independent* Wav2Vec2-style 1D conv stacks (one per channel,
    320x stride) → shared LayerNorm → shared Linear(512->768)
        → concat the two per-channel token streams (channel-major:
          [ch0_t0..ch0_t199, ch1_t0..ch1_t199]) into one 2N-token sequence
        → + 2D sinusoidal positional embedding (channel x time)
        → 12-layer post-norm Transformer encoder (d=768, h=12, FF=3072, GELU)
    Input  : raw binaural waveform (B, 2, T) at 16 kHz
    Output : (B, 2*N, 768) @ 100 Hz per channel (N=200 for a 2.01 s clip)

The student-side state-dict keys are exactly:
    extract_audio_1.*  extract_audio_2.*  feature_norms.*
    post_extraction_mapper.*  encoder.*  pos_encoding_encoder
A checkpoint saved by `SpatialWavJEPATrainer` is loaded here with the
training-only weights (teacher / decoder / mappers / mask_token /
pos_encoding_decoder) dropped.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the proven WavJEPA building blocks. `paper/JEPA/WavJEPA` is a sibling
# directory and its modules are flat (imported as top-level `WavJEPA`), so
# add it to sys.path before importing.
_WAVJEPA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "WavJEPA")
if _WAVJEPA_DIR not in sys.path:
    sys.path.insert(0, _WAVJEPA_DIR)

from WavJEPA import (  # noqa: E402
    DEFAULT_CONV_LAYERS_SPEC,
    SAMPLE_RATE,
    ConvFeatureExtractor,
    WavJEPA,
    compute_total_patches,
    make_sincos_pos_embed_1d,
)


logger = logging.getLogger("SpatialWavJEPA")


# =============================================================================
# Constants — binaural 2.01 s window (matches WavJEPA-base time resolution)
# =============================================================================

PROCESS_AUDIO_SECONDS = 2.01                                  # one clip
TARGET_LENGTH = int(SAMPLE_RATE * PROCESS_AUDIO_SECONDS)       # 32_160 samples
N_CHANNELS = 2                                                 # binaural
PATCHES_PER_CHANNEL = compute_total_patches(TARGET_LENGTH)     # 200
TOTAL_PATCHES = N_CHANNELS * PATCHES_PER_CHANNEL               # 400


# =============================================================================
# 2D sinusoidal positional embedding (channel x time, channel-major flatten)
# =============================================================================

def make_sincos_pos_embed_2d(
    embed_dim: int,
    n_channels: int,
    n_time: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Factorised 2D sinusoidal positional embedding for (channel, time).

    Each token (c, t) gets `concat([pe_channel[c], pe_time[t]])`. The
    feature budget is split in half: the low half encodes the channel
    index, the high half encodes the time index (MAE / AudioMAE 2D-sincos
    convention, built on the 1D primitive `make_sincos_pos_embed_1d`).

    Flatten order is **channel-major** — token index = c * n_time + t —
    so it lines up with the `[w1; w2]` concat in `SpatialWavJEPA.forward`
    and with `_SpanMasker`'s channel-flatten layout
    (`[ch0_t0..ch0_t(N-1), ch1_t0..ch1_t(N-1)]`).

    Returns (1, n_channels * n_time, embed_dim).
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    d_ch = embed_dim // 2
    d_t = embed_dim - d_ch
    if d_ch % 2:                       # make_sincos_pos_embed_1d needs even dims
        d_ch -= 1
        d_t = embed_dim - d_ch

    pe_ch = make_sincos_pos_embed_1d(d_ch, n_channels, dtype=dtype)[0]   # (C, d_ch)
    pe_t = make_sincos_pos_embed_1d(d_t, n_time, dtype=dtype)[0]         # (T, d_t)

    # Broadcast to the (C, T) grid and concat along the feature axis.
    ch = pe_ch.unsqueeze(1).expand(n_channels, n_time, d_ch)            # (C, T, d_ch)
    tm = pe_t.unsqueeze(0).expand(n_channels, n_time, d_t)              # (C, T, d_t)
    grid = torch.cat([ch, tm], dim=-1)                                  # (C, T, D)
    flat = grid.reshape(n_channels * n_time, embed_dim)                 # channel-major
    return flat.unsqueeze(0).contiguous()                               # (1, C*T, D)


# =============================================================================
# Config + Inference model
# =============================================================================

@dataclass
class SpatialWavJEPAConfig:
    # waveform / window
    sample_rate: int = SAMPLE_RATE
    process_audio_seconds: float = PROCESS_AUDIO_SECONDS
    n_channels: int = N_CHANNELS
    patches_per_channel: int = PATCHES_PER_CHANNEL
    total_patches: int = TOTAL_PATCHES

    # conv front-end (one independent stack per channel)
    conv_layers_spec: list = field(default_factory=lambda: list(DEFAULT_CONV_LAYERS_SPEC))
    conv_dropout: float = 0.0
    conv_mode: str = "default"
    conv_bias: bool = False
    conv_depthwise: bool = False

    # transformer encoder (shared across the concatenated token stream)
    encoder_d_model: int = 768
    encoder_nhead: int = 12
    encoder_dim_feedforward: int = 3072    # mlp_ratio 4
    encoder_num_layers: int = 12
    encoder_dropout: float = 0.0
    encoder_layer_norm_eps: float = 1e-6
    encoder_norm_first: bool = False
    encoder_bias: bool = True
    encoder_enable_nested_tensor: bool = False
    encoder_mask_check: bool = True

    @property
    def target_length(self) -> int:
        return int(self.sample_rate * self.process_audio_seconds)


class SpatialWavJEPA(nn.Module):
    """Inference-only Spatial WavJEPA (WavJEPA-Nat student encoder).

    `self.encoder` is the deployed Transformer; the teacher / predictor used
    during JEPA training are NOT present here (see `SpatialWavJEPA_Trainer.py`).
    A single fixed `process_audio_seconds` window is encoded in one pass:
    shorter input is silence-padded, longer input is truncated (chunk longer
    recordings caller-side and stack the outputs along time if needed).
    """

    def __init__(self, config: Optional[SpatialWavJEPAConfig] = None):
        super().__init__()
        cfg = config or SpatialWavJEPAConfig()
        self.config = cfg

        if cfg.n_channels != 2:
            raise ValueError(
                f"SpatialWavJEPA (WavJEPA-Nat) is binaural; n_channels must be 2, "
                f"got {cfg.n_channels}."
            )

        # Sanity: per-channel conv output must match patches_per_channel and
        # the 2D pos-embed length must equal total_patches.
        expected = compute_total_patches(cfg.target_length, cfg.conv_layers_spec)
        if cfg.patches_per_channel != expected:
            raise ValueError(
                f"config.patches_per_channel={cfg.patches_per_channel} but conv "
                f"stack outputs {expected} tokens for target_length="
                f"{cfg.target_length}. Keep process_audio_seconds=2.01 "
                f"(32160 samples -> 200 patches)."
            )
        if cfg.total_patches != cfg.n_channels * cfg.patches_per_channel:
            raise ValueError(
                f"config.total_patches={cfg.total_patches} must equal "
                f"n_channels*patches_per_channel="
                f"{cfg.n_channels * cfg.patches_per_channel}."
            )

        # Two INDEPENDENT per-channel conv stacks (WavJEPA-Nat). Shared
        # LayerNorm + post-extraction mapper keep the per-token feature space
        # identical across channels before the shared Transformer.
        self.extract_audio_1 = ConvFeatureExtractor(
            conv_layers_spec=cfg.conv_layers_spec, in_channels=1,
            dropout=cfg.conv_dropout, mode=cfg.conv_mode,
            conv_bias=cfg.conv_bias, depthwise=cfg.conv_depthwise,
        )
        self.extract_audio_2 = ConvFeatureExtractor(
            conv_layers_spec=cfg.conv_layers_spec, in_channels=1,
            dropout=cfg.conv_dropout, mode=cfg.conv_mode,
            conv_bias=cfg.conv_bias, depthwise=cfg.conv_depthwise,
        )
        emb = self.extract_audio_1.embedding_dim
        self.feature_norms = nn.LayerNorm(emb)
        self.post_extraction_mapper: Optional[nn.Module] = (
            nn.Linear(emb, cfg.encoder_d_model) if emb != cfg.encoder_d_model else None
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.encoder_d_model,
            nhead=cfg.encoder_nhead,
            dim_feedforward=cfg.encoder_dim_feedforward,
            dropout=cfg.encoder_dropout,
            activation=nn.GELU(),
            layer_norm_eps=cfg.encoder_layer_norm_eps,
            batch_first=True,
            norm_first=cfg.encoder_norm_first,
            bias=cfg.encoder_bias,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.encoder_num_layers,
            norm=nn.LayerNorm(cfg.encoder_d_model),
            enable_nested_tensor=cfg.encoder_enable_nested_tensor,
            mask_check=cfg.encoder_mask_check,
        )

        self.register_buffer(
            "pos_encoding_encoder",
            make_sincos_pos_embed_2d(
                cfg.encoder_d_model, cfg.n_channels, cfg.patches_per_channel
            ),
            persistent=True,
        )

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize(audio: torch.Tensor) -> torch.Tensor:
        """Joint (channel, time) zero-mean / unit-std per clip.

        Channels are normalised *together* (not independently) so the
        inter-channel level cue (ILD) survives — critical for spatial.
        Matches `WavJEPA._normalize_segment`.
        """
        mean = audio.mean(dim=(-2, -1), keepdim=True)
        std = audio.std(dim=(-2, -1), keepdim=True)
        return (audio - mean) / (std + 1e-5)

    def _embed_channels(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, 2, T) -> (B, 2*N, enc_d), channel-major, pos-added."""
        w1 = self.extract_audio_1(audio[:, 0:1, :])          # (B, N, 512)
        w2 = self.extract_audio_2(audio[:, 1:2, :])          # (B, N, 512)
        w1 = self.feature_norms(w1)
        w2 = self.feature_norms(w2)
        if self.post_extraction_mapper is not None:
            w1 = self.post_extraction_mapper(w1)             # (B, N, enc_d)
            w2 = self.post_extraction_mapper(w2)
        w = torch.cat([w1, w2], dim=1)                       # (B, 2N, enc_d)
        return w + self.pos_encoding_encoder

    @torch.inference_mode()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode a binaural waveform into per-frame embeddings.

        Args:
            audio: (B, 2, T) float waveform at `config.sample_rate`.
                   T < target_length is silence-padded, T > is truncated.

        Returns:
            (B, 2*patches_per_channel, encoder_d_model) — channel-major:
            the first N rows are channel 0, the next N are channel 1.
        """
        if audio.ndim != 3 or audio.shape[1] != self.config.n_channels:
            raise ValueError(
                f"audio must be (B, {self.config.n_channels}, T); got "
                f"{tuple(audio.shape)}"
            )
        seg_len = self.config.target_length
        T_in = audio.shape[-1]
        if T_in < seg_len:
            audio = F.pad(audio, (0, seg_len - T_in))
        elif T_in > seg_len:
            audio = audio[..., :seg_len]

        local = self._embed_channels(self._normalize(audio))
        return self.encoder(local)

    @torch.inference_mode()
    def get_audio_representation(
        self, audio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same as `forward`, plus per-frame timestamps (ms).

        Timestamps are per *channel* position (0..N-1 then repeated for the
        second channel), since the token stream is channel-major.
        """
        emb = self.forward(audio)
        cfg = self.config
        sec = cfg.target_length / cfg.sample_rate
        step_ms = sec / cfg.patches_per_channel * 1000.0
        per_ch = torch.arange(cfg.patches_per_channel, device=emb.device,
                              dtype=torch.float32) * step_ms
        ts = per_ch.repeat(cfg.n_channels)                  # (2N,)
        return emb, ts.unsqueeze(0).expand(emb.shape[0], -1).contiguous()

    # -------------------------------------------------------------------------
    # Checkpoint loading
    # -------------------------------------------------------------------------

    # Student-side keys only. A training checkpoint also carries
    # `teacher_encoder.*`, `decoder.*`, `*_mapper.*`, `mask_token`,
    # `pos_encoding_decoder` — all dropped here.
    _INFERENCE_KEY_PREFIXES = (
        "extract_audio_1.",
        "extract_audio_2.",
        "feature_norms.",
        "post_extraction_mapper.",
        "encoder.",
        "pos_encoding_encoder",
    )

    @classmethod
    def _filter_inference_keys(
        cls, sd: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Strip wrapper prefixes (reuse WavJEPA's) and keep student keys."""
        out: dict[str, torch.Tensor] = {}
        for raw_k, v in sd.items():
            k = raw_k
            changed = True
            while changed:
                changed = False
                for p in WavJEPA._STRIP_PREFIXES:
                    if k.startswith(p):
                        k = k[len(p):]
                        changed = True
            if any(k == p or k.startswith(p) for p in cls._INFERENCE_KEY_PREFIXES):
                out[k] = v
        return out

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        config: Optional[SpatialWavJEPAConfig] = None,
        map_location: str = "cpu",
        device: Optional[torch.device] = None,
        strict: bool = False,
    ) -> "SpatialWavJEPA":
        """Build a `SpatialWavJEPA` and load student weights from a `.pt`.

        Accepts raw or wrapped state dicts (`state_dict`,
        `model_state_dict`, `model`), strips `module.` / `_orig_mod.` /
        `model.` prefixes (via `WavJEPA`), drops training-only keys. A
        `pos_encoding_encoder` whose shape doesn't match this config is
        dropped (the model keeps its own sincos buffer). Loading failure
        with `strict=True`, or any missing/unexpected key, is logged; the
        load is aborted (raise) when nothing usable was applied.
        """
        model = cls(config)
        try:
            blob = torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            blob = torch.load(path, map_location=map_location, weights_only=False)

        sd = WavJEPA._unwrap_state_dict(blob)
        filtered = cls._filter_inference_keys(sd)

        pe = filtered.get("pos_encoding_encoder")
        if pe is not None and pe.shape != model.pos_encoding_encoder.shape:
            logger.warning(
                "pos_encoding_encoder shape %s != model %s — keeping model's own.",
                tuple(pe.shape), tuple(model.pos_encoding_encoder.shape),
            )
            filtered.pop("pos_encoding_encoder")

        if not filtered:
            raise RuntimeError(
                f"No student-side keys found in checkpoint {path!r}; "
                f"refusing to run an uninitialised encoder."
            )

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        total = len(filtered)
        applied = total - len(unexpected)
        if not missing and not unexpected:
            logger.info("SpatialWavJEPA loaded successfully (%d/%d keys)", applied, total)
        else:
            logger.warning(
                "SpatialWavJEPA partial load: %d/%d keys (missing=%d, unexpected=%d)",
                applied, total, len(missing), len(unexpected),
            )
            if strict and (missing or unexpected):
                raise RuntimeError(
                    f"strict=True but missing={missing} unexpected={unexpected}"
                )
        if applied == 0:
            raise RuntimeError(
                f"0/{total} checkpoint keys applied from {path!r} — aborting."
            )

        if device is not None:
            model = model.to(device)
        model.eval()
        model.requires_grad_(False)        # inference-only: freeze all params
        return model
