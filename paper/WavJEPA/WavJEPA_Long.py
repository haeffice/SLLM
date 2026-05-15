"""Long-context WavJEPA inference module (PyTorch 2.8).

Companion to `WavJEPA.py`. The original `WavJEPA` was pre-trained on 2.01-s
clips and at inference splits longer audio into independent 2-s segments
(no cross-segment attention). `WavJEPALong` is for a **single-pass**
finetune at a longer fixed window (10 s by default): the encoder sees the
whole window as one Transformer sequence, the positional embedding spans
the full window, and cross-time attention works over the entire input.

State-dict layout is intentionally identical to `WavJEPA` (same submodule
names: `extract_audio`, `feature_norms`, `post_extraction_mapper`,
`encoder`, `pos_encoding_encoder`), so the **same trainer**
(`WavJEPATrainer` from `WavJEPA_Trainer.py`) can be used — pass
`process_audio_seconds=10.0` (plus scaled mask spans) to its kwargs-only
`__init__` and the trainer's pos-embedding length will match this
encoder. Training scripts therefore never need to import this module.

Behaviour for variable-length input:
    audio length T_in       action
    ─────────────────────  ──────────────────────────────────────────────
    T_in <  target_length  silence-pad to target_length; per-sample
                           padding mask built from `audio_lengths`
                           (falls back to T_in if not provided)
    T_in == target_length  single encoder pass (the fast path)
    T_in >  target_length  truncated to target_length (a `lengths` arg
                           is *not* a substitute for proper windowing)

Caveat — this module **assumes you will finetune end-to-end** at the new
window length. Loading the released `wavjepa-base` (200-patch) checkpoint
into this 999-patch model triggers a pos-embed shape mismatch which we
silently skip; the new sincos pos embed covers positions [0..998], i.e.
extrapolation from pre-training. Without finetuning, those positions are
out-of-distribution for the encoder.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from WavJEPA import (
    DEFAULT_CONV_LAYERS_SPEC,
    SAMPLE_RATE,
    ConvFeatureExtractor,
    WavJEPA,
    WavJEPAConfig,
    compute_total_patches,
    make_sincos_pos_embed_1d,
)


logger = logging.getLogger("WavJEPA_Long")


# =============================================================================
# Defaults for the long window
# =============================================================================

LONG_PROCESS_AUDIO_SECONDS = 10.0
LONG_TARGET_LENGTH = int(SAMPLE_RATE * LONG_PROCESS_AUDIO_SECONDS)  # 160_000


# =============================================================================
# Helpers
# =============================================================================

def make_long_config(
    seconds: float = LONG_PROCESS_AUDIO_SECONDS,
    sample_rate: int = SAMPLE_RATE,
    conv_layers_spec: Optional[list[tuple[int, int, int]]] = None,
    in_channels: int = 1,
    **overrides,
) -> WavJEPAConfig:
    """Build a `WavJEPAConfig` for the long single-pass window.

    `total_patches` is derived from the conv stack so the user never has
    to hand-compute it. Any other `WavJEPAConfig` field can be overridden
    via kwargs.
    """
    target_length = int(sample_rate * seconds)
    spec = list(conv_layers_spec or DEFAULT_CONV_LAYERS_SPEC)
    total_patches = compute_total_patches(target_length, spec)
    return WavJEPAConfig(
        sample_rate=sample_rate,
        process_audio_seconds=seconds,
        total_patches=total_patches,
        in_channels=in_channels,
        conv_layers_spec=spec,
        conv_in_channels=in_channels,
        **overrides,
    )


# =============================================================================
# Single-pass long-window encoder
# =============================================================================

class WavJEPALong(nn.Module):
    """Single-pass WavJEPA for a fixed long window (default 10 s).

    Same submodule layout as `WavJEPA` — checkpoints saved by
    `WavJEPATrainer` configured with `make_long_trainer_config(...)`
    load cleanly via `WavJEPALong.from_checkpoint(path)`.
    """

    def __init__(
        self,
        config: Optional[WavJEPAConfig] = None,
        seconds: Optional[float] = None,
    ):
        super().__init__()
        if config is None:
            config = make_long_config(seconds=seconds or LONG_PROCESS_AUDIO_SECONDS)
        elif seconds is not None and config.process_audio_seconds != seconds:
            raise ValueError(
                f"`seconds={seconds}` conflicts with "
                f"`config.process_audio_seconds={config.process_audio_seconds}`"
            )
        self.config = config
        cfg = config

        # Sanity: pos embed length must equal conv output length.
        expected = compute_total_patches(cfg.target_length, cfg.conv_layers_spec)
        if cfg.total_patches != expected:
            raise ValueError(
                f"config.total_patches={cfg.total_patches} but conv stack outputs "
                f"{expected} tokens for target_length={cfg.target_length}. "
                f"Use `make_long_config(...)` to derive total_patches automatically."
            )

        # Same student path as WavJEPA — keys must match for ckpt compatibility.
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

        self.register_buffer(
            "pos_encoding_encoder",
            make_sincos_pos_embed_1d(cfg.encoder_d_model, cfg.total_patches),
            persistent=True,
        )

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize(audio: torch.Tensor, audio_lengths: Optional[torch.Tensor]) -> torch.Tensor:
        """Per-clip zero-mean / unit-std normalisation.

        When `audio_lengths` is given, mean/std are computed over the real
        samples only (so silence-pad doesn't skew the statistics). Otherwise
        they're computed over the full tensor — matching `WavJEPA._normalize_segment`.
        """
        if audio_lengths is None:
            mean = audio.mean(dim=(-2, -1), keepdim=True)
            std = audio.std(dim=(-2, -1), keepdim=True)
            return (audio - mean) / (std + 1e-5)

        _, C, T = audio.shape
        idx = torch.arange(T, device=audio.device).unsqueeze(0)             # (1, T)
        valid = idx < audio_lengths.to(audio.device).unsqueeze(1)            # (B, T)
        valid_f = valid.to(audio.dtype).unsqueeze(1)                         # (B, 1, T)

        denom = valid_f.sum(dim=(-2, -1), keepdim=True) * C
        denom = denom.clamp_min(1)
        mean = (audio * valid_f).sum(dim=(-2, -1), keepdim=True) / denom
        var = (((audio - mean) ** 2) * valid_f).sum(dim=(-2, -1), keepdim=True) / denom
        std = var.clamp_min(0).sqrt()

        # Zero the pad region so it stays "silence" after normalisation (no
        # spurious -mean/std DC offset leaking into the conv stack).
        normalised = (audio - mean) / (std + 1e-5)
        return normalised * valid_f

    def _build_padding_mask(
        self, audio_lengths: torch.Tensor, batch_size: int, device: torch.device,
    ) -> torch.Tensor:
        """Convert per-sample real-sample counts into a (B, total_patches) bool mask.

        True = padded position the encoder must ignore.
        """
        cfg = self.config
        samples_per_token = cfg.target_length / cfg.total_patches
        valid_tokens = (audio_lengths.to(device=device, dtype=torch.float32)
                        / samples_per_token).floor().to(torch.long)
        valid_tokens = valid_tokens.clamp(min=1, max=cfg.total_patches)
        positions = torch.arange(cfg.total_patches, device=device).unsqueeze(0)
        return positions.expand(batch_size, -1) >= valid_tokens.unsqueeze(1)

    @torch.inference_mode()
    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a fixed-window waveform in one Transformer pass.

        Args:
            audio          : (B, C, T) float waveform at `config.sample_rate`.
                             T < target_length is silence-padded; T > target_length
                             is truncated.
            audio_lengths  : optional (B,) int tensor of real sample counts per
                             clip. If given, used both to mask attention over
                             the silence tail AND to compute normalisation
                             statistics over the real signal only. If omitted,
                             a uniform-batch heuristic is used (one length for
                             the whole batch from T_in vs target_length).

        Returns:
            (B, total_patches, encoder_d_model). Always full length — silence-
            region embeddings are present but the encoder did not attend to/
            from them when `audio_lengths` was provided.
        """
        if audio.ndim != 3:
            raise ValueError(
                f"audio must be 3D (B, C, T); got shape {tuple(audio.shape)}"
            )
        cfg = self.config
        seg_len = cfg.target_length
        B, _, T_in = audio.shape

        # Normalise length to target_length.
        if T_in < seg_len:
            audio = F.pad(audio, (0, seg_len - T_in))
            if audio_lengths is None:
                audio_lengths = torch.full(
                    (B,), T_in, dtype=torch.long, device=audio.device,
                )
        elif T_in > seg_len:
            audio = audio[..., :seg_len]
            if audio_lengths is not None:
                audio_lengths = audio_lengths.clamp(max=seg_len)
        # T_in == seg_len: nothing to do; audio_lengths stays as caller passed.

        normalised = self._normalize(audio, audio_lengths)

        local = self.extract_audio(normalised)              # (B, S, 512)
        local = self.feature_norms(local)
        if self.post_extraction_mapper is not None:
            local = self.post_extraction_mapper(local)      # (B, S, enc_d)
        local = local + self.pos_encoding_encoder

        pmask = (
            self._build_padding_mask(audio_lengths, B, audio.device)
            if audio_lengths is not None else None
        )
        return self.encoder(local, src_key_padding_mask=pmask)

    @torch.inference_mode()
    def get_audio_representation(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same as `forward`, plus per-frame timestamps (in ms)."""
        embeddings = self.forward(audio, audio_lengths)
        step_ms = (self.config.target_length / self.config.sample_rate
                   / embeddings.shape[1] * 1000.0)
        ts = torch.arange(embeddings.shape[1], device=embeddings.device,
                          dtype=torch.float32) * step_ms
        return embeddings, ts.unsqueeze(0).expand(embeddings.shape[0], -1).contiguous()

    # -------------------------------------------------------------------------
    # Checkpoint loading
    # -------------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        config: Optional[WavJEPAConfig] = None,
        seconds: Optional[float] = None,
        map_location: str = "cpu",
        device: Optional[torch.device] = None,
        strict: bool = False,
    ) -> "WavJEPALong":
        """Build a `WavJEPALong` and load weights from a `.pt` checkpoint.

        Same loader contract as `WavJEPA.from_checkpoint`: accepts raw or
        wrapped state dicts (`state_dict`, `model_state_dict`, `model`),
        strips `module.` / `_orig_mod.` / `model.` prefixes, drops training-
        only keys (`teacher_encoder.*`, `decoder.*`, `*_mapper.*`,
        `mask_token`, `pos_encoding_decoder`).

        If the loaded `pos_encoding_encoder` doesn't match this model's
        configured `total_patches`, it is *silently dropped* — the long
        model uses its own sincos buffer for the new length. This is the
        usual case when loading a 2-s pre-trained `wavjepa-base` ckpt;
        finetune end-to-end at the long window before deploying.
        """
        model = cls(config=config, seconds=seconds)

        try:
            blob = torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            blob = torch.load(path, map_location=map_location, weights_only=False)

        sd = WavJEPA._unwrap_state_dict(blob)
        filtered = WavJEPA._filter_inference_keys(sd)

        # Drop pos_encoding_encoder if shape mismatches — the long model has
        # its own correctly-sized sincos buffer.
        pe = filtered.get("pos_encoding_encoder")
        if pe is not None and pe.shape != model.pos_encoding_encoder.shape:
            logger.info(
                "pos_encoding_encoder shape %s != model %s — keeping model's own sincos.",
                tuple(pe.shape), tuple(model.pos_encoding_encoder.shape),
            )
            filtered.pop("pos_encoding_encoder")

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        total = len(filtered)
        applied = total - len(unexpected)
        if not missing and not unexpected:
            logger.info("WavJEPALong loaded successfully (%d/%d keys)", applied, total)
        else:
            logger.warning(
                "WavJEPALong partial load: %d/%d keys (missing=%d, unexpected=%d)",
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
