"""Standalone WavJEPA trainer module (PyTorch 2.8).

`WavJEPATrainer` is the JEPA training wrapper. It owns

    * the *student* path that the inference model (`WavJEPA`) will reuse —
      `self.extract_audio`, `self.feature_norms`, `self.post_extraction_mapper`,
      `self.encoder` (the deployable encoder), `self.pos_encoding_encoder`.
    * a *teacher* path — `self.teacher_encoder`, a frozen EMA-updated copy
      of the student encoder that produces the prediction targets.
    * a *predictor* — `self.decoder` (a Transformer fed mask tokens at the
      target positions plus context tokens from the student), with
      encoder↔decoder dim mappers and a learned `mask_token`.

The class is self-contained: EMA scheduling, EMA step, target normalisation
(top-K layer-averaged + per-layer instance norm — data2vec 2.0 style),
masked MSE loss, and a span masker all live inside the module. The training
loop only needs to call `forward(audio, ctx_masks, target_indices,
ctx_and_target_masks)` and `ema_step(global_step)` after each optimizer step.

References:
    * I-JEPA (Assran et al. 2023, http://arxiv.org/abs/2301.08243)
    * data2vec 2.0 (Baevski et al. 2022, http://arxiv.org/abs/2212.07525)
    * WavJEPA (Yousefi et al. 2025, http://arxiv.org/abs/2509.23238)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from WavJEPA import (
    DEFAULT_CONV_LAYERS_SPEC,
    PROCESS_AUDIO_SECONDS,
    SAMPLE_RATE,
    ConvFeatureExtractor,
    WavJEPAConfig,
    compute_total_patches,
    make_sincos_pos_embed_1d,
)


logger = logging.getLogger("WavJEPA_Trainer")


# =============================================================================
# Trainer config
# =============================================================================

@dataclass
class WavJEPATrainerConfig(WavJEPAConfig):
    # decoder (predictor)
    decoder_d_model: int = 384
    decoder_nhead: int = 12
    decoder_dim_feedforward: int = 1536    # mlp_ratio 4
    decoder_num_layers: int = 12
    decoder_dropout: float = 0.0
    decoder_layer_norm_eps: float = 1e-6
    decoder_norm_first: bool = False
    decoder_bias: bool = True
    decoder_enable_nested_tensor: bool = False
    decoder_mask_check: bool = True

    # EMA / target schedule
    ema_decay: float = 0.999
    ema_end_decay: float = 0.99999
    ema_anneal_end_step: int = 100_000
    average_top_k_layers: int = 12          # K-layer averaged target

    # masking
    target_masks_per_context: int = 4
    target_prob: float = 0.20
    target_length: int = 20
    context_mask_prob: float = 0.30
    context_mask_length: int = 10
    ratio_cutoff: float = 0.05
    masker_kind: str = "time-inverse"       # "time-inverse" | "speech"
    speech_min_context_len: int = 5


# =============================================================================
# Span mask generator
# (port of wavjepa.audio_masking.compute_mask_indices, condensed)
# =============================================================================

def _compute_mask_indices(
    bsz: int, sz: int, mask_prob: float, mask_length: int, *,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Sample boolean span masks of fixed `mask_length`.

    Returns `(bsz, sz)` bool tensor (True = masked). All rows are forced
    to have the same number of True positions (`require_same_masks=True`
    behaviour from the upstream impl).
    """
    mask = np.zeros((bsz, sz), dtype=bool)
    mask_idcs: list[np.ndarray] = []
    for _ in range(bsz):
        num_mask = int(mask_prob * sz / float(mask_length) + rng.random())
        num_mask = max(1, num_mask)
        starts = rng.choice(max(1, sz - mask_length + 1), num_mask, replace=False)
        idcs = np.unique(np.concatenate([starts + o for o in range(mask_length)]))
        idcs = idcs[idcs < sz]
        mask_idcs.append(idcs)
    # equalise count
    target_len = min(len(m) for m in mask_idcs)
    for i, idc in enumerate(mask_idcs):
        if len(idc) > target_len:
            idc = rng.choice(idc, target_len, replace=False)
        mask[i, idc] = True
    return torch.from_numpy(mask)


# =============================================================================
# Span masker — produces (ctx, targets, ctx∪targets) from patch count
# =============================================================================

class _SpanMasker(nn.Module):
    """Time-inverse-block masker (default) or speech-style target-only masker.

    Output (matches the layout that `WavJEPATrainer.forward` expects):
        ctx_masks               : (B, n_times)
            True = position is *not* visible to the student encoder.
        target_indices          : (B, n_targets_per_context, n_times)
            True = position the predictor must reconstruct.
        ctx_and_target_masks    : (B, n_targets_per_context, n_times)
            True = position is neither a visible context nor a target —
            i.e. the position the decoder must attend-mask out for this
            target slot. (xor of ctx_masks and target_indices.)
    """

    def __init__(self, cfg: WavJEPATrainerConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _filter_small_clusters(mask: torch.Tensor, min_len: int) -> torch.Tensor:
        """Drop True-runs shorter than `min_len` (used by speech masker)."""
        values, counts = torch.unique_consecutive(mask, return_counts=True)
        small = values & (counts < min_len)
        values = values.clone()
        values[small] = False
        return torch.repeat_interleave(values, counts)

    def forward(self, batch_size: int, n_times: int,
                in_channels: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        n_times_per_ch = n_times // max(1, in_channels)
        T = cfg.target_masks_per_context
        rng = np.random.default_rng()

        target_positions = torch.zeros(batch_size, T, n_times_per_ch, dtype=torch.bool)
        context_positions = torch.zeros(batch_size, n_times_per_ch, dtype=torch.bool)

        for b in range(batch_size):
            while True:
                targets_b = torch.zeros(T, n_times_per_ch, dtype=torch.bool)
                for ti in range(T):
                    targets_b[ti] = _compute_mask_indices(
                        1, n_times_per_ch, cfg.target_prob, cfg.target_length, rng=rng
                    )[0]
                any_target = torch.any(targets_b, dim=0)

                if cfg.masker_kind == "speech":
                    ctx_b = ~any_target
                    ctx_b = self._filter_small_clusters(ctx_b, cfg.speech_min_context_len)
                else:  # time-inverse
                    ctx_b = ~_compute_mask_indices(
                        1, n_times_per_ch, cfg.context_mask_prob, cfg.context_mask_length, rng=rng
                    )[0]
                    ctx_b = ctx_b & ~any_target

                if (ctx_b.sum().item() / max(1, n_times_per_ch)) >= cfg.ratio_cutoff:
                    target_positions[b] = targets_b
                    context_positions[b] = ctx_b
                    break

        ctx_masks = ~context_positions                          # (B, S)
        ctx_and_target = torch.logical_xor(
            ctx_masks.unsqueeze(1), target_positions
        )                                                       # (B, T, S)

        if in_channels > 1:
            # Channel-flattening extractor: repeat the mask across channels.
            ctx_masks = ctx_masks.unsqueeze(1).expand(-1, in_channels, -1)
            ctx_masks = ctx_masks.reshape(batch_size, in_channels * n_times_per_ch)
            target_positions = target_positions.unsqueeze(1).expand(-1, in_channels, -1, -1)
            target_positions = target_positions.permute(0, 2, 3, 1).reshape(
                batch_size, T, in_channels * n_times_per_ch
            )
            ctx_and_target = ctx_and_target.unsqueeze(1).expand(-1, in_channels, -1, -1)
            ctx_and_target = ctx_and_target.permute(0, 2, 3, 1).reshape(
                batch_size, T, in_channels * n_times_per_ch
            )

        return ctx_masks, target_positions, ctx_and_target


# =============================================================================
# Trainer model
# =============================================================================

def _build_transformer(d_model: int, nhead: int, ffn: int, num_layers: int,
                       dropout: float, layer_norm_eps: float,
                       norm_first: bool, bias: bool,
                       enable_nested_tensor: bool, mask_check: bool) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=ffn,
        dropout=dropout, activation=nn.GELU(),
        layer_norm_eps=layer_norm_eps,
        batch_first=True, norm_first=norm_first, bias=bias,
    )
    return nn.TransformerEncoder(
        layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
        enable_nested_tensor=enable_nested_tensor, mask_check=mask_check,
    )


class WavJEPATrainer(nn.Module):
    """JEPA trainer: student encoder + EMA teacher + decoder-predictor.

    `self.encoder` is the deployable Transformer (student). After training,
    save `self.state_dict()` to a `.pt` file and load it at inference time
    with `WavJEPA.from_checkpoint(...)` (or `WavJEPALong.from_checkpoint(...)`
    for a long-window finetune), which keeps only the student-side weights
    (`extract_audio.*`, `feature_norms.*`, `post_extraction_mapper.*`,
    `encoder.*`, `pos_encoding_encoder`).

    All config fields are keyword arguments of `__init__` (no separate
    `WavJEPATrainerConfig` to build) — long-window finetune is just
    `WavJEPATrainer(process_audio_seconds=10.0, target_length=100,
    context_mask_length=50)`. `total_patches` is auto-derived from the
    conv stack + `process_audio_seconds` when not explicitly given.

    NOTE — `target_length` here is a **mask span length in patches** (a
    legacy field-name collision with `process_audio_seconds * sample_rate`
    in samples; the latter is exposed as `self.config.target_length`
    *property* on the parent `WavJEPAConfig`). Scale `target_length` and
    `context_mask_length` up when `total_patches` grows so coverage
    fraction stays constant.
    """

    def __init__(
        self,
        *,
        # ---- Audio window ----
        sample_rate: int = SAMPLE_RATE,
        process_audio_seconds: float = PROCESS_AUDIO_SECONDS,
        total_patches: Optional[int] = None,                  # auto-derived if None
        in_channels: int = 1,
        # ---- Conv front-end ----
        conv_layers_spec: Optional[list[tuple[int, int, int]]] = None,
        conv_dropout: float = 0.0,
        conv_mode: str = "default",
        conv_bias: bool = False,
        conv_depthwise: bool = False,
        # ---- Encoder (student / inference encoder) ----
        encoder_d_model: int = 768,
        encoder_nhead: int = 12,
        encoder_dim_feedforward: int = 3072,
        encoder_num_layers: int = 12,
        encoder_dropout: float = 0.0,
        encoder_layer_norm_eps: float = 1e-6,
        encoder_norm_first: bool = False,
        encoder_bias: bool = True,
        encoder_enable_nested_tensor: bool = False,
        encoder_mask_check: bool = True,
        # ---- Decoder / predictor ----
        decoder_d_model: int = 384,
        decoder_nhead: int = 12,
        decoder_dim_feedforward: int = 1536,
        decoder_num_layers: int = 12,
        decoder_dropout: float = 0.0,
        decoder_layer_norm_eps: float = 1e-6,
        decoder_norm_first: bool = False,
        decoder_bias: bool = True,
        decoder_enable_nested_tensor: bool = False,
        decoder_mask_check: bool = True,
        # ---- EMA / target ----
        ema_decay: float = 0.999,
        ema_end_decay: float = 0.99999,
        ema_anneal_end_step: int = 100_000,
        average_top_k_layers: int = 12,
        # ---- Masking (lengths in PATCH units, not samples) ----
        target_masks_per_context: int = 4,
        target_prob: float = 0.20,
        target_length: int = 20,                              # mask span in patches
        context_mask_prob: float = 0.30,
        context_mask_length: int = 10,                        # context mask span in patches
        ratio_cutoff: float = 0.05,
        masker_kind: str = "time-inverse",                    # "time-inverse" | "speech"
        speech_min_context_len: int = 5,
        # ---- Other ----
        loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()

        spec = list(conv_layers_spec or DEFAULT_CONV_LAYERS_SPEC)
        if total_patches is None:
            total_patches = compute_total_patches(
                int(sample_rate * process_audio_seconds), spec,
            )

        # Snapshot every field on `self.config` so trainers can be inspected
        # / serialised the same way regardless of how they were built.
        self.config = WavJEPATrainerConfig(
            sample_rate=sample_rate,
            process_audio_seconds=process_audio_seconds,
            total_patches=total_patches,
            in_channels=in_channels,
            conv_layers_spec=spec,
            conv_in_channels=in_channels,
            conv_dropout=conv_dropout,
            conv_mode=conv_mode,
            conv_bias=conv_bias,
            conv_depthwise=conv_depthwise,
            encoder_d_model=encoder_d_model,
            encoder_nhead=encoder_nhead,
            encoder_dim_feedforward=encoder_dim_feedforward,
            encoder_num_layers=encoder_num_layers,
            encoder_dropout=encoder_dropout,
            encoder_layer_norm_eps=encoder_layer_norm_eps,
            encoder_norm_first=encoder_norm_first,
            encoder_bias=encoder_bias,
            encoder_enable_nested_tensor=encoder_enable_nested_tensor,
            encoder_mask_check=encoder_mask_check,
            decoder_d_model=decoder_d_model,
            decoder_nhead=decoder_nhead,
            decoder_dim_feedforward=decoder_dim_feedforward,
            decoder_num_layers=decoder_num_layers,
            decoder_dropout=decoder_dropout,
            decoder_layer_norm_eps=decoder_layer_norm_eps,
            decoder_norm_first=decoder_norm_first,
            decoder_bias=decoder_bias,
            decoder_enable_nested_tensor=decoder_enable_nested_tensor,
            decoder_mask_check=decoder_mask_check,
            ema_decay=ema_decay,
            ema_end_decay=ema_end_decay,
            ema_anneal_end_step=ema_anneal_end_step,
            average_top_k_layers=average_top_k_layers,
            target_masks_per_context=target_masks_per_context,
            target_prob=target_prob,
            target_length=target_length,
            context_mask_prob=context_mask_prob,
            context_mask_length=context_mask_length,
            ratio_cutoff=ratio_cutoff,
            masker_kind=masker_kind,
            speech_min_context_len=speech_min_context_len,
        )
        cfg = self.config

        # ---- student waveform path (shared with WavJEPA) ----
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

        self.encoder = _build_transformer(
            cfg.encoder_d_model, cfg.encoder_nhead, cfg.encoder_dim_feedforward,
            cfg.encoder_num_layers, cfg.encoder_dropout, cfg.encoder_layer_norm_eps,
            cfg.encoder_norm_first, cfg.encoder_bias,
            cfg.encoder_enable_nested_tensor, cfg.encoder_mask_check,
        )

        # ---- decoder / predictor ----
        self.decoder = _build_transformer(
            cfg.decoder_d_model, cfg.decoder_nhead, cfg.decoder_dim_feedforward,
            cfg.decoder_num_layers, cfg.decoder_dropout, cfg.decoder_layer_norm_eps,
            cfg.decoder_norm_first, cfg.decoder_bias,
            cfg.decoder_enable_nested_tensor, cfg.decoder_mask_check,
        )
        self.encoder_to_decoder_mapper = nn.Linear(cfg.encoder_d_model, cfg.decoder_d_model)
        self.decoder_to_encoder_mapper = nn.Linear(cfg.decoder_d_model, cfg.encoder_d_model, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # ---- positional embeddings (fixed sinusoidal, non-trainable) ----
        self.register_buffer(
            "pos_encoding_encoder",
            make_sincos_pos_embed_1d(cfg.encoder_d_model, cfg.total_patches),
            persistent=True,
        )
        self.register_buffer(
            "pos_encoding_decoder",
            make_sincos_pos_embed_1d(cfg.decoder_d_model, cfg.total_patches),
            persistent=True,
        )

        self.loss_fn = loss_fn or nn.MSELoss(reduction="none")
        self.masker = _SpanMasker(cfg)

        self.apply(self._init_weights)
        self._init_teacher()

    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_teacher(self):
        # Teacher mirrors the student's encoder exactly at step 0; from then
        # on it's updated by EMA only. It must be an `nn.Module` attribute so
        # that `.to(device)` moves it and `state_dict()` checkpoints it.
        self.teacher_encoder: nn.Module = copy.deepcopy(self.encoder)
        self.teacher_encoder.requires_grad_(False)

    # -------------------------------------------------------------------------
    # Masks
    # -------------------------------------------------------------------------

    def generate_masks(
        self, batch_size: int, device: Optional[torch.device] = None,
        in_channels: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience helper — sample masks for one training batch.

        Useful if your dataloader doesn't produce masks. Returns the three
        tensors that `forward` consumes, moved to `device` if given.
        """
        ctx, tgt, c_or_t = self.masker(batch_size, self.config.total_patches, in_channels)
        if device is not None:
            ctx, tgt, c_or_t = ctx.to(device), tgt.to(device), c_or_t.to(device)
        return ctx, tgt, c_or_t

    # -------------------------------------------------------------------------
    # EMA
    # -------------------------------------------------------------------------

    def _ema_decay(self, global_step: int) -> float:
        cfg = self.config
        if global_step >= cfg.ema_anneal_end_step:
            return cfg.ema_end_decay
        r = cfg.ema_end_decay - cfg.ema_decay
        pct_remaining = 1.0 - (global_step / cfg.ema_anneal_end_step)
        return cfg.ema_end_decay - r * pct_remaining

    @torch.no_grad()
    def ema_step(self, global_step: int) -> float:
        """Polyak update of `teacher_encoder` from `encoder`. Call once per
        optimizer step (after `optimizer.step()`).

        Returns the EMA decay rate that was applied (useful for logging).
        """
        r = self._ema_decay(global_step)
        for student_p, teacher_p in zip(
            self.encoder.parameters(), self.teacher_encoder.parameters()
        ):
            teacher_p.data.mul_(r).add_(student_p.detach().data, alpha=(1.0 - r))
        # Sync buffers as a precaution (no-op for stateless encoders, but
        # future LayerNorm-with-tracking-stats / batchnorm variants need it).
        for student_b, teacher_b in zip(
            self.encoder.buffers(), self.teacher_encoder.buffers()
        ):
            teacher_b.data.copy_(student_b.data)
        return r

    # -------------------------------------------------------------------------
    # Targets (data2vec 2.0 style: top-K layers, instance-normed, averaged)
    # -------------------------------------------------------------------------

    @staticmethod
    def _make_targets(layer_outputs: list[torch.Tensor]) -> torch.Tensor:
        """Average top-K teacher layer outputs after instance norm.

        layer_outputs: K tensors of shape (B, S, D).
        Returns      : (B, S, D).
        """
        stacked = torch.stack(layer_outputs)                       # (K, B, S, D)
        transposed = stacked.transpose(2, 3)                        # (K, B, D, S)
        normalised = F.instance_norm(transposed)                    # over (D, S)
        normalised = normalised.transpose(2, 3)                     # (K, B, S, D)
        return normalised.mean(dim=0)                               # (B, S, D)

    @torch.no_grad()
    def _forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        K = self.config.average_top_k_layers
        layer_outputs: list[torch.Tensor] = []
        layers = self.teacher_encoder.layers
        n = len(layers)
        for i, blk in enumerate(layers):
            x = blk(x)
            if (n - i) <= K:
                layer_outputs.append(x)
        if K > 1:
            return self._make_targets(layer_outputs)
        return layer_outputs[-1]

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    def _masked_loss(self, pred: torch.Tensor, target: torch.Tensor,
                     target_indices: torch.Tensor) -> torch.Tensor:
        """
        pred           : (B*N, S, D) — decoder outputs, one slot per target group.
        target         : (B,   S, D) — teacher representation (same for all N).
        target_indices : (B,   N, S) — True at positions the predictor must hit.
        """
        B, N, _ = target_indices.shape
        D = pred.shape[-1]

        pred = pred.view(B, N, -1, D)
        target_b = target.unsqueeze(1).expand(-1, N, -1, -1)            # (B, N, S, D)
        per_elem = self.loss_fn(pred, target_b)                          # (B, N, S, D)
        per_step = per_elem.mean(dim=-1)                                 # (B, N, S)
        masked = per_step * target_indices.to(per_step.dtype)
        denom = target_indices.sum().clamp_min(1).to(per_step.dtype)
        return masked.sum() / denom

    # -------------------------------------------------------------------------
    # Encoder / decoder forward
    # -------------------------------------------------------------------------

    def _encoder_forward(self, x: torch.Tensor,
                         src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def _decoder_forward(self, contextual_features: torch.Tensor,
                         ctx_mask: torch.Tensor, nr_targets: int,
                         src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """Mask-token decoder, then map back to encoder dim.

        contextual_features : (n_visible_total, decoder_d) — flattened visible
                              tokens from the student encoder (after the
                              encoder→decoder mapper).
        ctx_mask            : (B, S)        True = masked out of context.
        nr_targets          : N             one decoder pass per target group.
        src_key_padding_mask: (B, N, S)     True = decoder must ignore.
        """
        B = ctx_mask.shape[0]
        S = self.config.total_patches
        D = self.config.decoder_d_model

        tgt = self.mask_token.expand(B, S, D).clone().to(contextual_features.dtype)
        tgt[~ctx_mask, :] = contextual_features.reshape(-1, D)
        tgt = tgt + self.pos_encoding_decoder

        # Repeat per-target: (B, S, D) -> (B*N, S, D)
        tgt = tgt.unsqueeze(1).expand(-1, nr_targets, -1, -1).reshape(B * nr_targets, S, D)
        pad = src_key_padding_mask.reshape(B * nr_targets, S)

        tgt = self.decoder(tgt, src_key_padding_mask=pad)
        return self.decoder_to_encoder_mapper(tgt)                   # (B*N, S, enc_d)

    # -------------------------------------------------------------------------
    # Main training forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        audio: torch.Tensor,
        ctx_masks: torch.Tensor,
        target_indices: torch.Tensor,
        ctx_and_target_masks: torch.Tensor,
    ) -> dict:
        """One training step.

        Args:
            audio                : (B, C, target_length) — clip of `process_audio_seconds`
                                   seconds at `sample_rate`, normalised to zero mean
                                   and unit variance by the dataloader.
            ctx_masks            : (B, S)    True = position hidden from student.
            target_indices       : (B, N, S) True = position the predictor must hit.
            ctx_and_target_masks : (B, N, S) True = decoder ignores this position.

        Returns dict with:
            loss                  : scalar tensor (masked MSE).
            local_features        : (B, S, encoder_d) — post-LN, post-mapper, pos-added.
            contextual_features   : (n_visible_total, decoder_d) — packed visible tokens.
            preds                 : (B*N, S, encoder_d) — predictor outputs.
            targets               : (B, S, encoder_d) — teacher targets.
        """
        # Student waveform path — IDENTICAL to inference up to the encoder call.
        local_features = self.extract_audio(audio)                  # (B, S, 512)
        local_features = self.feature_norms(local_features)
        if self.post_extraction_mapper is not None:
            local_features = self.post_extraction_mapper(local_features)
        local_features = local_features + self.pos_encoding_encoder  # (B, S, enc_d)

        # Student encoder over context tokens only.
        contextual = self._encoder_forward(local_features, src_key_padding_mask=ctx_masks)
        contextual = contextual[~ctx_masks]                          # (n_vis, enc_d)
        contextual = self.encoder_to_decoder_mapper(contextual)      # (n_vis, dec_d)

        # Predictor.
        preds = self._decoder_forward(
            contextual, ctx_masks,
            nr_targets=target_indices.shape[1],
            src_key_padding_mask=ctx_and_target_masks,
        )

        # Teacher targets (no grad).
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
