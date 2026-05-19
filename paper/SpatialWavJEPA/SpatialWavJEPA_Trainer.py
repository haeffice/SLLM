"""Spatial WavJEPA (WavJEPA-Nat) JEPA trainer (PyTorch 2.8).

`SpatialWavJEPATrainer` subclasses `WavJEPATrainer` (from
`paper/WavJEPA/WavJEPA_Trainer.py`) and changes only the *student waveform
path* to the binaural WavJEPA-Nat design:

    * two INDEPENDENT per-channel conv stacks (`extract_audio_1/2`) instead
      of one `extract_audio`,
    * a 2D (channel x time) sinusoidal positional embedding instead of the
      1D one,
    * a 2N-token sequence (channel-major) fed to the shared Transformer,
    * context / target block indices SHARED across both channels (the
      parent `_SpanMasker`'s `in_channels=2` channel-flatten already does
      exactly this).

Everything else — the EMA teacher, decoder/predictor, data2vec-2 target
construction, masked MSE loss, EMA schedule/step — is inherited unchanged
(all of it is agnostic to the sequence length S, which is now 400).

Masking and optimisation defaults follow the WavJEPA paper
(arXiv:2509.23238): context block p=0.065 / len 10, target block p=0.025 /
len 10, K=8 targets, >=10% context.

The student-side state-dict keys match `SpatialWavJEPA` exactly, so a
checkpoint saved here loads with `SpatialWavJEPA.from_checkpoint(...)`.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

_WAVJEPA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "WavJEPA")
if _WAVJEPA_DIR not in sys.path:
    sys.path.insert(0, _WAVJEPA_DIR)

from WavJEPA import ConvFeatureExtractor  # noqa: E402
from WavJEPA_Trainer import WavJEPATrainer  # noqa: E402

from SpatialWavJEPA import (  # noqa: E402
    N_CHANNELS,
    PATCHES_PER_CHANNEL,
    PROCESS_AUDIO_SECONDS,
    TOTAL_PATCHES,
    make_sincos_pos_embed_2d,
)


logger = logging.getLogger("SpatialWavJEPA_Trainer")


# WavJEPA-paper masking / EMA defaults (override the parent trainer defaults).
_PAPER_DEFAULTS = dict(
    process_audio_seconds=PROCESS_AUDIO_SECONDS,   # 2.01 s -> 200 patches/ch
    total_patches=TOTAL_PATCHES,                    # 400 = 2 channels x 200
    in_channels=N_CHANNELS,                         # 2 (binaural)
    target_masks_per_context=8,                     # K = 8 targets
    target_prob=0.025,                              # p_target
    target_length=10,                               # M_target (patches)
    context_mask_prob=0.065,                        # p_context
    context_mask_length=10,                         # M_context (patches)
    ratio_cutoff=0.10,                              # >=10% of indices are context
    masker_kind="time-inverse",
    ema_decay=0.999,
    ema_end_decay=0.99999,
    ema_anneal_end_step=100_000,
    average_top_k_layers=12,
)


class SpatialWavJEPATrainer(WavJEPATrainer):
    """JEPA trainer for binaural WavJEPA-Nat.

    Construction reuses the parent (encoder / decoder / teacher / mappers /
    mask_token / masker / EMA), then swaps in the dual conv front-end and
    the 2D positional embeddings. After training, `self.state_dict()`'s
    student-side keys (`extract_audio_1.*`, `extract_audio_2.*`,
    `feature_norms.*`, `post_extraction_mapper.*`, `encoder.*`,
    `pos_encoding_encoder`) load via `SpatialWavJEPA.from_checkpoint(...)`.
    """

    def __init__(self, **kwargs):
        # WavJEPA-paper defaults win unless the caller explicitly overrides.
        merged = dict(_PAPER_DEFAULTS)
        merged.update(kwargs)
        if merged.get("in_channels") != 2 or merged.get("total_patches") != TOTAL_PATCHES:
            raise ValueError(
                "SpatialWavJEPATrainer is binaural: in_channels must be 2 and "
                f"total_patches must be {TOTAL_PATCHES}."
            )

        # Parent builds a single `extract_audio` (conv_in_channels=2), the
        # shared LayerNorm + 512->768 mapper, encoder, decoder, 1D pos-embeds,
        # masker, then `_init_weights` + `_init_teacher`. We keep everything
        # except `extract_audio` and the two pos-embed buffers.
        super().__init__(**merged)
        cfg = self.config

        # --- dual independent per-channel conv stacks (WavJEPA-Nat) ---
        del self.extract_audio
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
        self.extract_audio_1.apply(self._init_weights)
        self.extract_audio_2.apply(self._init_weights)

        # --- 2D (channel x time) positional embeddings (same shapes/keys as
        #     the parent's 1D buffers, so checkpoint parity holds) ---
        pe_enc = make_sincos_pos_embed_2d(
            cfg.encoder_d_model, N_CHANNELS, PATCHES_PER_CHANNEL
        )
        pe_dec = make_sincos_pos_embed_2d(
            cfg.decoder_d_model, N_CHANNELS, PATCHES_PER_CHANNEL
        )
        assert pe_enc.shape == self.pos_encoding_encoder.shape, (
            pe_enc.shape, self.pos_encoding_encoder.shape)
        assert pe_dec.shape == self.pos_encoding_decoder.shape, (
            pe_dec.shape, self.pos_encoding_decoder.shape)
        self.pos_encoding_encoder.copy_(pe_enc)
        self.pos_encoding_decoder.copy_(pe_dec)

    # -------------------------------------------------------------------------
    # Masks — force binaural shared cross-channel indices
    # -------------------------------------------------------------------------

    def generate_masks(
        self, batch_size: int, device: Optional[torch.device] = None,
        in_channels: int = N_CHANNELS,
    ):
        """Sample one batch of masks, shared across both channels.

        We sample at `PATCHES_PER_CHANNEL` (=200) with the parent masker in
        single-channel mode, then tile channel-major to `TOTAL_PATCHES`
        (=400): `[ch0_t0..ch0_t199, ch1_t0..ch1_t199]` with ch0 == ch1.

        This is done here (not via the parent's `in_channels>1` branch)
        because that branch flattens `ctx` channel-major but `target` /
        `ctx_and_target` time-interleaved — inconsistent with our
        channel-major `[w1; w2]` concat and 2D pos-embed. Tiling all three
        identically guarantees the token order matches everywhere.
        """
        ctx, tgt, c_or_t = self.masker(batch_size, PATCHES_PER_CHANNEL, 1)
        ctx = ctx.repeat(1, N_CHANNELS)                  # (B, 400)
        tgt = tgt.repeat(1, 1, N_CHANNELS)               # (B, K, 400)
        c_or_t = c_or_t.repeat(1, 1, N_CHANNELS)         # (B, K, 400)
        if device is not None:
            ctx, tgt, c_or_t = ctx.to(device), tgt.to(device), c_or_t.to(device)
        return ctx, tgt, c_or_t

    # -------------------------------------------------------------------------
    # Student waveform path: dual extract + channel-major concat
    # -------------------------------------------------------------------------

    def _spatial_local_features(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, 2, T) -> (B, 2*N, enc_d), channel-major, pos-added."""
        if audio.ndim != 3 or audio.shape[1] != N_CHANNELS:
            raise ValueError(
                f"audio must be (B, {N_CHANNELS}, T); got {tuple(audio.shape)}"
            )
        w1 = self.extract_audio_1(audio[:, 0:1, :])          # (B, N, 512)
        w2 = self.extract_audio_2(audio[:, 1:2, :])          # (B, N, 512)
        w1 = self.feature_norms(w1)
        w2 = self.feature_norms(w2)
        if self.post_extraction_mapper is not None:
            w1 = self.post_extraction_mapper(w1)             # (B, N, enc_d)
            w2 = self.post_extraction_mapper(w2)
        w = torch.cat([w1, w2], dim=1)                       # (B, 2N, enc_d)
        return w + self.pos_encoding_encoder

    def forward(
        self,
        audio: torch.Tensor,
        ctx_masks: torch.Tensor,
        target_indices: torch.Tensor,
        ctx_and_target_masks: torch.Tensor,
    ) -> dict:
        """One JEPA training step (binaural).

        Args mirror `WavJEPATrainer.forward`; `audio` is (B, 2, target_length)
        already normalised by the dataloader, and the mask tensors have a
        sequence length of `total_patches` (= 400).
        """
        # Student waveform path — the only change vs the parent.
        local_features = self._spatial_local_features(audio)            # (B, 2N, enc_d)

        S = self.config.total_patches
        assert local_features.shape[1] == S == ctx_masks.shape[-1], (
            local_features.shape, S, ctx_masks.shape,
        )

        # --- everything below is identical to the parent forward ---
        contextual = self._encoder_forward(local_features, src_key_padding_mask=ctx_masks)
        contextual = contextual[~ctx_masks]                             # (n_vis, enc_d)
        contextual = self.encoder_to_decoder_mapper(contextual)         # (n_vis, dec_d)

        preds = self._decoder_forward(
            contextual, ctx_masks,
            nr_targets=target_indices.shape[1],
            src_key_padding_mask=ctx_and_target_masks,
        )

        with torch.no_grad():
            targets = self._forward_teacher(local_features.detach())

        loss = self._masked_loss(preds, targets, target_indices)

        return {
            "loss": loss,
            "local_features": local_features,
            "contextual_features": contextual,
            "preds": preds,
            "targets": targets,
        }
