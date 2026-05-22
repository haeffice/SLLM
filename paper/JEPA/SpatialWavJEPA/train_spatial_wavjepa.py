"""Spatial WavJEPA (WavJEPA-Nat) SSL pre-training (PyTorch 2.8).

JEPA self-supervised pre-training of the binaural WavJEPA-Nat encoder on
AudioSet waveforms convolved with binaural room impulse responses (BRIRs).

Stack (per `paper/CLAUDE.md`):
    * `transformers.PreTrainedModel` wrapper around `SpatialWavJEPATrainer`
    * `transformers.Trainer` training loop
    * `torch.utils.data.Dataset` / `DataLoader` (BAT-style reverb pipeline)
    * `accelerate` + `torchrun` for multi-GPU (CPU-compatible)
    * logs: per-module trainable params; first batch's first sample wav
      PATH before the model is fed; step / train_loss / valid_loss / lr /
      ema-decay every `logging_steps`; checkpoints every `save_steps` with
      the step number in the filename; abort if a resume checkpoint fails
      to load.

Run via `run_train_SpatialWavJEPA.sh config.yaml` (never invoke directly in
production — the shell script enforces the "no pre-existing checkpoints"
guard). All arguments live in a single `config.yaml`.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy import signal
from torch.utils.data import Dataset
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.optimization import get_cosine_schedule_with_warmup

import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_WAVJEPA_DIR = os.path.join(os.path.dirname(_HERE), "WavJEPA")
for _p in (_HERE, _WAVJEPA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from SpatialWavJEPA import (  # noqa: E402
    N_CHANNELS,
    SAMPLE_RATE,
    TARGET_LENGTH,
    TOTAL_PATCHES,
)
from SpatialWavJEPA_Trainer import SpatialWavJEPATrainer  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_spatial_wavjepa")


# =============================================================================
# Dataset — AudioSet wav (x) BRIR -> binaural 2.01 s clip (BAT-style)
# =============================================================================

def _rms_normalize(audio: np.ndarray, target_dbfs: float = -14.0) -> np.ndarray:
    """RMS-to-target-dBFS gain (port of BAT.normalize_audio)."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms == 0.0:
        return audio
    gain = 10.0 ** ((target_dbfs - 20.0 * np.log10(rms)) / 20.0)
    return audio * gain


class SpatialWavJEPADataset(Dataset):
    """Manifest-driven binaural SSL dataset.

    Manifest JSON: ``{"data": [{"audio_id": str, "reverb_id": str}, ...]}``.
    ``audio_path  = audio_root/audio_id`` (mono AudioSet wav, any SR),
    ``reverb_path = reverb_root/reverb_id`` (binaural BRIR ``.npy``, shape
    ``(2, R)``). No labels (pure SSL).
    """

    def __init__(self, manifest: str, audio_root: str, reverb_root: str,
                 sample_rate: int = SAMPLE_RATE, clip_samples: int = TARGET_LENGTH,
                 train: bool = True):
        with open(manifest, "r") as f:
            self.samples = json.load(f)["data"]
        self.audio_root = audio_root
        self.reverb_root = reverb_root
        self.sample_rate = sample_rate
        self.clip_samples = clip_samples
        self.train = train
        if not self.samples:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_or_pad(self, wave: np.ndarray) -> np.ndarray:
        """(C, L) -> (C, clip_samples): random crop in train, head in eval."""
        c, L = wave.shape
        n = self.clip_samples
        if L == n:
            return wave
        if L > n:
            start = random.randint(0, L - n) if self.train else 0
            return wave[:, start:start + n]
        return np.pad(wave, ((0, 0), (0, n - L)))

    def _load_waveform(self, audio_path: str, reverb_path: str) -> torch.Tensor:
        wave, sr = sf.read(audio_path)
        if wave.ndim > 1:                       # force mono source
            wave = wave[:, 0]
        if sr != self.sample_rate:
            wave = signal.resample_poly(wave, self.sample_rate, sr)
        wave = _rms_normalize(np.asarray(wave, dtype=np.float32))
        wave = wave.reshape(1, -1)              # (1, T)

        reverb = np.load(reverb_path).astype(np.float32)   # binaural BRIR (2, R)
        if reverb.ndim != 2 or reverb.shape[0] != N_CHANNELS:
            raise ValueError(
                f"BRIR {reverb_path} must be shape (2, R); got {reverb.shape}"
            )
        wave = signal.fftconvolve(wave, reverb, mode="full")   # (2, T+R-1)
        wave = self._crop_or_pad(np.ascontiguousarray(wave, dtype=np.float32))

        # Joint (channel, time) zero-mean / unit-std — keep the inter-channel
        # level cue (ILD); matches WavJEPA / SpatialWavJEPA normalisation.
        t = torch.from_numpy(wave).float()
        t = (t - t.mean()) / (t.std() + 1e-5)
        return t                                # (2, clip_samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        audio_path = os.path.join(self.audio_root, s["audio_id"])
        reverb_path = os.path.join(self.reverb_root, s["reverb_id"])
        return {
            "audio": self._load_waveform(audio_path, reverb_path),
            "audio_path": audio_path,
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "audio": torch.stack([b["audio"] for b in batch]),     # (B, 2, 32160)
        "audio_path": [b["audio_path"] for b in batch],
    }


# =============================================================================
# transformers wrapper
# =============================================================================

class SpatialWavJEPAHFConfig(PretrainedConfig):
    model_type = "spatial_wavjepa"

    def __init__(self, trainer_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.trainer_kwargs = trainer_kwargs or {}


class SpatialWavJEPAModel(PreTrainedModel):
    """Thin HF wrapper: owns a `SpatialWavJEPATrainer`, samples masks per
    step, returns `{"loss": ...}` so `Trainer` works without a custom
    `compute_loss` (we still override it for first-batch logging)."""

    config_class = SpatialWavJEPAHFConfig

    def __init__(self, config: SpatialWavJEPAHFConfig):
        super().__init__(config)
        self.spatial = SpatialWavJEPATrainer(**config.trainer_kwargs)
        self.post_init()

    def _init_weights(self, module):  # SpatialWavJEPATrainer self-initialises.
        pass

    def forward(self, audio: torch.Tensor = None, **_) -> dict:
        ctx, tgt, c_or_t = self.spatial.generate_masks(
            audio.shape[0], device=audio.device
        )
        out = self.spatial(audio, ctx, tgt, c_or_t)
        return {"loss": out["loss"]}

    @torch.no_grad()
    def ema_step(self, global_step: int) -> float:
        return self.spatial.ema_step(global_step)


# =============================================================================
# Callbacks
# =============================================================================

def _unwrap(model) -> SpatialWavJEPAModel:
    return model.module if hasattr(model, "module") else model


def _count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class ParamCountCallback(TrainerCallback):
    """Log per-module trainable parameter counts once at train start."""

    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        sp = _unwrap(model).spatial
        groups = {
            "extract_audio_1": sp.extract_audio_1,
            "extract_audio_2": sp.extract_audio_2,
            "feature_norms": sp.feature_norms,
            "post_extraction_mapper": sp.post_extraction_mapper,
            "encoder (student/deployed)": sp.encoder,
            "decoder (predictor)": sp.decoder,
            "encoder_to_decoder_mapper": sp.encoder_to_decoder_mapper,
            "decoder_to_encoder_mapper": sp.decoder_to_encoder_mapper,
            "teacher_encoder (EMA, frozen)": sp.teacher_encoder,
        }
        logger.info("==== trainable parameters per module ====")
        for name, mod in groups.items():
            if mod is None:
                continue
            logger.info("  %-32s %15s", name, f"{_count_trainable(mod):,}")
        logger.info("  %-32s %15s", "mask_token", f"{sp.mask_token.numel():,}")
        logger.info("  %-32s %15s", "TOTAL trainable", f"{_count_trainable(sp):,}")
        logger.info("=========================================")


class FirstSampleLogCallback(TrainerCallback):
    """Log the first batch's first sample wav PATH before the model is fed."""

    def __init__(self):
        self.done = False

    def mark(self, audio_path: list[str], audio: torch.Tensor):
        if self.done:
            return
        self.done = True
        logger.info(
            "==== first batch / first sample (pre-feed) ====\n"
            "  wav path : %s\n  audio    : shape=%s dtype=%s",
            audio_path[0] if audio_path else "<unknown>",
            tuple(audio.shape), audio.dtype,
        )


class EMACallback(TrainerCallback):
    """Polyak-update the EMA teacher after every optimizer step; expose the
    applied decay so it shows up in the periodic train log."""

    def __init__(self):
        self.last_decay = None

    def on_step_end(self, args, state, control, model=None, **kw):
        self.last_decay = _unwrap(model).ema_step(state.global_step)

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is not None and self.last_decay is not None:
            logs["ema_decay"] = round(float(self.last_decay), 8)


class SaveStudentCallback(TrainerCallback):
    """On every HF checkpoint, also dump a student-only `.pt` whose filename
    carries the step number (loadable via `SpatialWavJEPA.from_checkpoint`)."""

    _STUDENT_PREFIXES = (
        "extract_audio_1.", "extract_audio_2.", "feature_norms.",
        "post_extraction_mapper.", "encoder.", "pos_encoding_encoder",
    )

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        sp = _unwrap(model).spatial
        sd = {k: v.detach().cpu()
              for k, v in sp.state_dict().items()
              if any(k == p or k.startswith(p) for p in self._STUDENT_PREFIXES)}
        out = os.path.join(
            args.output_dir, f"spatial_wavjepa_student_step{state.global_step}.pt"
        )
        torch.save(sd, out)
        logger.info("saved student checkpoint (%d tensors) -> %s", len(sd), out)


# =============================================================================
# Trainer subclass — paper optimiser/schedule + first-sample logging
# =============================================================================

class SpatialWavJEPAHFTrainer(Trainer):
    def __init__(self, *a, first_sample_cb: FirstSampleLogCallback = None, **kw):
        super().__init__(*a, **kw)
        self._first_sample_cb = first_sample_cb

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_sample_cb is not None:
            self._first_sample_cb.mark(inputs.get("audio_path"), inputs["audio"])
        outputs = model(audio=inputs["audio"])      # audio_path NOT fed to model
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            decay, no_decay = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if p.ndim <= 1 else decay).append(p)
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": self.args.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.optimizer, self.lr_scheduler


# =============================================================================
# Config + main
# =============================================================================

@dataclass
class _Cfg:
    raw: dict

    def sec(self, k: str) -> dict:
        return self.raw.get(k, {}) or {}


def _build_trainer_kwargs(c: _Cfg) -> dict:
    m, mk, em = c.sec("model"), c.sec("masking"), c.sec("ema")
    d = c.sec("data")
    kw = dict(
        sample_rate=int(d.get("sample_rate", SAMPLE_RATE)),
        process_audio_seconds=float(d.get("process_audio_seconds", 2.01)),
        total_patches=TOTAL_PATCHES,
        in_channels=N_CHANNELS,
    )
    for src, keys in (
        (m, ("encoder_d_model", "encoder_nhead", "encoder_dim_feedforward",
             "encoder_num_layers", "decoder_d_model", "decoder_nhead",
             "decoder_dim_feedforward", "decoder_num_layers",
             "average_top_k_layers")),
        (mk, ("context_mask_prob", "context_mask_length", "target_prob",
              "target_length", "target_masks_per_context", "ratio_cutoff",
              "masker_kind")),
        (em, ("ema_decay", "ema_end_decay", "ema_anneal_end_step")),
    ):
        for k in keys:
            if k in src:
                kw[k] = src[k]
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = _Cfg(yaml.safe_load(f))
    d, t, o = cfg.sec("data"), cfg.sec("train"), cfg.sec("optim")

    seed = int(t.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainer_kwargs = _build_trainer_kwargs(cfg)
    logger.info("SpatialWavJEPATrainer kwargs: %s", trainer_kwargs)

    model = SpatialWavJEPAModel(SpatialWavJEPAHFConfig(trainer_kwargs=trainer_kwargs))

    # Optional warm-start from a student `.pt` — abort if nothing loads.
    init_from = t.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location="cpu", weights_only=True)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.spatial.load_state_dict(sd, strict=False)
        applied = len(sd) - len(unexpected)
        if applied == 0:
            raise RuntimeError(
                f"init_from={init_from!r}: 0/{len(sd)} keys applied — aborting."
            )
        logger.info("init_from %s: applied %d/%d keys (missing=%d, unexpected=%d)",
                    init_from, applied, len(sd), len(missing), len(unexpected))

    train_ds = SpatialWavJEPADataset(
        d["train_manifest"], d["audio_root"], d["reverb_root"],
        sample_rate=trainer_kwargs["sample_rate"], train=True,
    )
    eval_ds = None
    if d.get("valid_manifest"):
        eval_ds = SpatialWavJEPADataset(
            d["valid_manifest"], d["audio_root"], d["reverb_root"],
            sample_rate=trainer_kwargs["sample_rate"], train=False,
        )

    use_cpu = not torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=t["output_dir"],
        max_steps=int(o.get("max_steps", 375_000)),
        per_device_train_batch_size=int(o.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(o.get("per_device_eval_batch_size",
                                              o.get("per_device_train_batch_size", 16))),
        gradient_accumulation_steps=int(o.get("gradient_accumulation_steps", 8)),
        learning_rate=float(o.get("learning_rate", 2.0e-4)),
        weight_decay=float(o.get("weight_decay", 0.04)),
        adam_beta1=float(o.get("adam_beta1", 0.9)),
        adam_beta2=float(o.get("adam_beta2", 0.98)),
        warmup_steps=int(o.get("warmup_steps", 100_000)),
        logging_steps=int(t.get("logging_steps", 100)),
        save_steps=int(t.get("save_steps", 5_000)),
        save_total_limit=t.get("save_total_limit"),
        eval_strategy=("steps" if eval_ds is not None else "no"),
        eval_steps=int(t.get("eval_steps", 5_000)),
        dataloader_num_workers=int(t.get("num_workers", 4)),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,        # keep "audio"/"audio_path"
        label_names=[],
        use_cpu=use_cpu,
        ddp_backend=("gloo" if use_cpu else None),
        max_grad_norm=float(o.get("max_grad_norm", 1.0)),
    )

    first_cb = FirstSampleLogCallback()
    trainer = SpatialWavJEPAHFTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        first_sample_cb=first_cb,
        callbacks=[ParamCountCallback(), first_cb, EMACallback(),
                   SaveStudentCallback()],
    )

    trainer.train()
    trainer.save_model(os.path.join(targs.output_dir, "final"))
    logger.info("training complete; final HF model -> %s/final", targs.output_dir)


if __name__ == "__main__":
    main()
