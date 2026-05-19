"""Standalone WavJEPA inference module (PyTorch 2.8).

Self-contained reimplementation of the deployable half of WavJEPA
(arXiv:2509.23238, ref impl: github.com/labhamlet/wavjepa, ckpt:
huggingface.co/labhamlet/wavjepa-base). Includes only what is needed to
encode a raw waveform into per-frame embeddings — the EMA teacher,
decoder, mask token, and predictor live in `WavJEPA_Trainer.py`.

Architecture (base config, matches the released `wavjepa-base`):
    Wav2Vec2-style 1D conv stack (7 layers, 320× stride)
        → LayerNorm → Linear(512→768)
        → 12-layer post-norm Transformer encoder (d=768, h=12, FF=3072, GELU)
    Input  : raw waveform (B, 1, T) at 16 kHz
    Output : (B, T_out, 768) @ 100 Hz (10 ms stride)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


logger = logging.getLogger("WavJEPA")


# =============================================================================
# Constants — match the released `wavjepa-base` checkpoint
# =============================================================================

SAMPLE_RATE = 16_000
PROCESS_AUDIO_SECONDS = 2.01            # one segment fed to the encoder
TARGET_LENGTH = int(SAMPLE_RATE * PROCESS_AUDIO_SECONDS)  # 32_160
TOTAL_PATCHES = 200                     # output time steps per segment (~100 Hz)
DEFAULT_CONV_LAYERS_SPEC = (
    [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)]
)


# =============================================================================
# Conv feature extractor (Wav2Vec2 style)
# =============================================================================

class _ChannelLastLayerNorm(nn.Module):
    """LayerNorm over channels for (B, C, T) tensors (rearrange-free)."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(-1, -2)).transpose(-1, -2)


def _make_conv_block(n_in: int, n_out: int, k: int, stride: int,
                     *, is_layer_norm: bool, is_group_norm: bool,
                     conv_bias: bool, depthwise: bool,
                     dropout: float) -> nn.Sequential:
    if depthwise:
        assert n_out % n_in == 0, f"depthwise needs n_out%n_in==0, got {n_in}, {n_out}"
        conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias, groups=n_in)
    else:
        conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
    nn.init.kaiming_normal_(conv.weight)

    assert not (is_layer_norm and is_group_norm), "layer/group norm exclusive"
    if is_layer_norm:
        return nn.Sequential(
            conv,
            nn.Dropout(p=dropout),
            _ChannelLastLayerNorm(n_out),
            nn.GELU(),
        )
    if is_group_norm:
        # GroupNorm(num_groups=n_out, num_channels=n_out) == per-channel LN.
        return nn.Sequential(
            conv,
            nn.Dropout(p=dropout),
            nn.GroupNorm(n_out, n_out, affine=True),
            nn.GELU(),
        )
    return nn.Sequential(conv, nn.Dropout(p=dropout), nn.GELU())


def compute_total_patches(
    target_length: int,
    conv_layers_spec: Optional[list[tuple[int, int, int]]] = None,
) -> int:
    """Analytical conv-stack output length for `(B, C, target_length)` input.

    Closed-form version of `ConvFeatureExtractor.total_patches` — doesn't
    instantiate any modules, so it's safe to call from a config builder
    before any `nn.Module` exists.

    Used by `WavJEPA_Long.make_long_config` and
    `WavJEPA_Trainer.make_long_trainer_config` to size `total_patches`
    automatically for non-default `target_length`.
    """
    spec = list(conv_layers_spec or DEFAULT_CONV_LAYERS_SPEC)
    n = target_length
    for _, k, s in spec:
        n = (n - k) // s + 1
    return n


class ConvFeatureExtractor(nn.Module):
    """1D conv stack matching wav2vec2 / wavjepa base.

    Args mirror `wavjepa.extractors.audio_extractor.ConvFeatureExtractor`.
    Input  : (B, in_channels, T)
    Output : (B, T_out, embedding_dim)   — last conv channel count.
    """

    def __init__(
        self,
        conv_layers_spec: Optional[list[tuple[int, int, int]]] = None,
        in_channels: int = 1,
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        depthwise: bool = False,
    ):
        super().__init__()
        assert mode in {"default", "layer_norm"}
        self.in_channels = in_channels
        self.conv_layers_spec = list(conv_layers_spec or DEFAULT_CONV_LAYERS_SPEC)

        layers: list[nn.Sequential] = []
        prev = in_channels
        for i, (dim, k, stride) in enumerate(self.conv_layers_spec):
            layers.append(_make_conv_block(
                prev, dim, k, stride,
                is_layer_norm=(mode == "layer_norm"),
                is_group_norm=(mode == "default" and i == 0),
                conv_bias=conv_bias,
                depthwise=depthwise,
                dropout=dropout,
            ))
            prev = dim
        self.cnn = nn.Sequential(*layers)
        self.embedding_dim = self.conv_layers_spec[-1][0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                                  # (B, C, T_out)
        return x.transpose(1, 2).contiguous()            # (B, T_out, C)

    @torch.no_grad()
    def total_patches(self, time: int) -> int:
        device = next(self.parameters()).device
        dummy = torch.zeros(1, self.in_channels, time, device=device)
        return self.cnn(dummy).shape[-1]


# =============================================================================
# Sinusoidal positional embedding (computed in torch, no numpy dep)
# =============================================================================

def make_sincos_pos_embed_1d(embed_dim: int, length: int,
                             dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """1D sinusoidal positional embedding (AudioMAE / MoCo-v3 convention).

    Returns (1, length, embed_dim).
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    half = embed_dim // 2
    # Use float64 for the angle frequencies for parity with the numpy
    # reference impl; cast down at the end.
    omega = torch.arange(half, dtype=torch.float64) / float(half)
    omega = 1.0 / (10_000.0 ** omega)
    pos = torch.arange(length, dtype=torch.float64)
    out = torch.outer(pos, omega)                              # (L, D/2)
    emb = torch.cat([out.sin(), out.cos()], dim=1)             # (L, D)
    return emb.to(dtype=dtype).unsqueeze(0)                    # (1, L, D)


# =============================================================================
# Config + Inference model
# =============================================================================

@dataclass
class WavJEPAConfig:
    # encoder / waveform
    sample_rate: int = SAMPLE_RATE
    process_audio_seconds: float = PROCESS_AUDIO_SECONDS
    total_patches: int = TOTAL_PATCHES
    in_channels: int = 1

    # conv front-end
    conv_layers_spec: list = field(default_factory=lambda: list(DEFAULT_CONV_LAYERS_SPEC))
    conv_in_channels: int = 1
    conv_dropout: float = 0.0
    conv_mode: str = "default"
    conv_bias: bool = False
    conv_depthwise: bool = False

    # transformer encoder
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


class WavJEPA(nn.Module):
    """Inference-only WavJEPA.

    Holds the student-side waveform encoder used at inference. `self.encoder`
    is the deployed Transformer; the teacher / predictor used during JEPA
    training are NOT present here (see `WavJEPA_Trainer.py`).
    """

    def __init__(self, config: Optional[WavJEPAConfig] = None):
        super().__init__()
        cfg = config or WavJEPAConfig()
        self.config = cfg

        self.extract_audio = ConvFeatureExtractor(
            conv_layers_spec=cfg.conv_layers_spec,
            in_channels=cfg.conv_in_channels,
            dropout=cfg.conv_dropout,
            mode=cfg.conv_mode,
            conv_bias=cfg.conv_bias,
            depthwise=cfg.conv_depthwise,
        )
        self.feature_norms = nn.LayerNorm(self.extract_audio.embedding_dim)
        self.post_extraction_mapper: Optional[nn.Module] = (
            nn.Linear(self.extract_audio.embedding_dim, cfg.encoder_d_model)
            if self.extract_audio.embedding_dim != cfg.encoder_d_model else None
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

        # Fixed sinusoidal positional embedding. Registered as a buffer (not a
        # Parameter) — it's never trained and torch 2.x's idiomatic place for
        # static tensors is `register_buffer`. State-dict key is unchanged so
        # checkpoints saved with `nn.Parameter(requires_grad=False)` (the
        # upstream layout) still load.
        self.register_buffer(
            "pos_encoding_encoder",
            make_sincos_pos_embed_1d(cfg.encoder_d_model, cfg.total_patches),
            persistent=True,
        )

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_segment(audio: torch.Tensor) -> torch.Tensor:
        """Zero-mean / unit-std normalisation per (channel, time) of each clip."""
        mean = audio.mean(dim=(-2, -1), keepdim=True)
        std = audio.std(dim=(-2, -1), keepdim=True)
        return (audio - mean) / (std + 1e-5)

    def _encode_segment(self, audio: torch.Tensor,
                        padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        local = self.extract_audio(audio)               # (B, T_out, 512)
        local = self.feature_norms(local)
        if self.post_extraction_mapper is not None:
            local = self.post_extraction_mapper(local)  # (B, T_out, 768)
        local = local + self.pos_encoding_encoder
        return self.encoder(local, src_key_padding_mask=padding_mask)

    @torch.inference_mode()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode a raw waveform into per-frame embeddings.

        Args:
            audio: (B, in_channels, T) float waveform at `config.sample_rate`.

        Returns:
            (B, T_out, encoder_d_model) where T_out ≈ ceil(T / 160)
            (∼100 Hz for 16 kHz input).
        """
        if audio.ndim != 3:
            raise ValueError(
                f"audio must be 3D (B, C, T); got shape {tuple(audio.shape)}"
            )
        cfg = self.config
        seg_len = cfg.target_length
        B, _, T_in = audio.shape

        pad_frames = (seg_len - (T_in % seg_len)) % seg_len
        if pad_frames > 0:
            audio = torch.nn.functional.pad(audio, (0, pad_frames))

        n_segments = audio.shape[-1] // seg_len
        output_steps = cfg.total_patches // cfg.in_channels
        total_out = output_steps * n_segments

        # Per-segment trailing padding mask: only the *last* segment may carry
        # padding tokens (everything else is real signal). We anchor the
        # output rate to integer seconds (200 patches / 2 s = 100 Hz) to
        # match the upstream reference impl, regardless of
        # `process_audio_seconds` carrying a fractional 0.01 s tail.
        proc_seconds_int = max(1, int(cfg.process_audio_seconds))
        output_sr = output_steps // proc_seconds_int           # 100 Hz @ base
        pad_seconds = pad_frames / cfg.sample_rate
        pad_steps = int(pad_seconds * output_sr)
        cut_off = total_out - pad_steps

        outs: list[torch.Tensor] = []
        for i in range(n_segments):
            seg = audio[..., i * seg_len:(i + 1) * seg_len]
            is_last = (i == n_segments - 1)
            seg_pad = pad_steps if (is_last and pad_steps > 0) else 0
            if seg_pad > 0:
                pmask = torch.zeros(B, output_steps, dtype=torch.bool,
                                    device=audio.device)
                pmask[:, output_steps - seg_pad:] = True
            else:
                pmask = None
            outs.append(self._encode_segment(self._normalize_segment(seg), pmask))

        x = torch.cat(outs, dim=1)
        return x[:, :cut_off, :]

    @torch.inference_mode()
    def get_audio_representation(
        self, audio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same as `forward`, plus per-frame timestamps (in ms).

        Returns:
            (embeddings, timestamps)
                embeddings : (B, T_out, encoder_d_model)
                timestamps : (B, T_out) — frame centers in milliseconds.
        """
        embeddings = self.forward(audio)
        B = audio.shape[0]
        sec = audio.shape[-1] / self.config.sample_rate
        step_ms = sec / embeddings.shape[1] * 1000.0
        ts = torch.arange(embeddings.shape[1], device=embeddings.device,
                          dtype=torch.float32) * step_ms
        return embeddings, ts.unsqueeze(0).expand(B, -1).contiguous()

    # -------------------------------------------------------------------------
    # Checkpoint loading
    # -------------------------------------------------------------------------

    # Keys that belong to the inference model. A training checkpoint also
    # contains `teacher_encoder.*`, `decoder.*`, `*_mapper.*`, `mask_token`,
    # and `pos_encoding_decoder` — all of which we drop here.
    _INFERENCE_KEY_PREFIXES = (
        "extract_audio.",
        "feature_norms.",
        "post_extraction_mapper.",
        "encoder.",
        "pos_encoding_encoder",
    )

    # Wrapper prefixes peeled off in order. Many `.pt` files chain multiple
    # of these (Lightning `state_dict`, DDP `module.`, `torch.compile`
    # `_orig_mod.`, HF `WavJEPAModel.model.`, the upstream JEPA's `model.`).
    _STRIP_PREFIXES = ("module.", "_orig_mod.", "model.")

    @classmethod
    def _unwrap_state_dict(cls, obj) -> dict[str, torch.Tensor]:
        """Pull a tensor dict out of common save formats."""
        if isinstance(obj, dict):
            for key in ("state_dict", "model_state_dict", "model", "weights"):
                if key in obj and isinstance(obj[key], dict):
                    return cls._unwrap_state_dict(obj[key])
            if all(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj
        raise ValueError(
            f"Could not find a tensor state-dict in checkpoint of type {type(obj)}"
        )

    @classmethod
    def _filter_inference_keys(
        cls, sd: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Strip wrapper prefixes and drop training-only weights."""
        out: dict[str, torch.Tensor] = {}
        for raw_k, v in sd.items():
            k = raw_k
            changed = True
            while changed:
                changed = False
                for p in cls._STRIP_PREFIXES:
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
        config: Optional[WavJEPAConfig] = None,
        map_location: str = "cpu",
        device: Optional[torch.device] = None,
        strict: bool = False,
    ) -> "WavJEPA":
        """Build a WavJEPA and load weights from a `.pt` checkpoint.

        Accepted checkpoint shapes (auto-detected):
            * raw `{key: tensor}` state dict
            * `{"state_dict": ...}`            (PyTorch Lightning)
            * `{"model_state_dict": ...}`     (common training scripts)
            * `{"model": ...}`                 (HF / SLAM-LLM style)

        Wrapper prefixes (`module.`, `_orig_mod.`, `model.`) are peeled off
        automatically. Training-only weights (teacher, decoder, predictor,
        mask token, decoder pos embed) are silently dropped.

        On torch 2.8 `torch.load` defaults to `weights_only=True`. We try
        that first (it's strictly safer) and fall back to `weights_only=False`
        only if the checkpoint contains non-tensor pickle metadata.
        """
        model = cls(config)
        try:
            blob = torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            blob = torch.load(path, map_location=map_location, weights_only=False)

        sd = cls._unwrap_state_dict(blob)
        filtered = cls._filter_inference_keys(sd)
        missing, unexpected = model.load_state_dict(filtered, strict=False)

        total = len(filtered)
        applied = total - len(unexpected)
        if not missing and not unexpected:
            logger.info("WavJEPA loaded successfully (%d/%d keys)", applied, total)
        else:
            logger.warning(
                "WavJEPA partial load: %d/%d keys (missing=%d, unexpected=%d)",
                applied, total, len(missing), len(unexpected),
            )
            if strict and (missing or unexpected):
                raise RuntimeError(
                    f"strict=True but missing={missing} unexpected={unexpected}"
                )

        if device is not None:
            model = model.to(device)
        model.eval()
        return model
