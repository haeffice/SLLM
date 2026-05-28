"""JEPA-Neural-Tokenizer training (PyTorch 2.8) — two stages, one config.

`config.yaml` carries a top-level `stage: 1|2`; this script branches on it:

  stage 1  — masked-latent JEPA SSL of the Conv+DAAM+Conformer encoder
             (`Stage1Trainer`), via the standard `transformers.Trainer`
             with an EMA-teacher callback.
  stage 2  — FSQ + HiFi-GAN reconstruction (`Stage2Trainer`) with a frozen
             Stage-1 encoder, trained adversarially via a custom
             `transformers.Trainer` subclass that owns the discriminator
             optimizer and steps G / D inside `training_step`.

Stack (per `paper/CLAUDE.md`):
    * `transformers.PreTrainedModel` wrapper around the stage trainer
    * `transformers.Trainer` training loop (+ GAN subclass for stage 2)
    * `torch.utils.data.Dataset` / `DataLoader` waveform pipeline
    * `accelerate` + `torchrun` for multi-GPU (CPU-compatible)
    * logs: per-module trainable params; first batch's first sample WAV PATH
      before the model is fed; step / train_loss / sub-losses / lr every
      `logging_steps`; checkpoints every `save_steps` with the step number in
      the filename (loadable via `NeuralTokenizer.from_checkpoint`); abort if
      an init / Stage-1 checkpoint fails to load.

Run via `run_train_NeuralTokenizer.sh config.yaml`. All args live in config.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
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

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from NeuralTokenizer import NeuralTokenizerConfig  # noqa: E402
from NeuralTokenizer_Trainer import Stage1Trainer, Stage2Trainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_neural_tokenizer")


# =============================================================================
# Dataset — manifest-driven waveform loader (soundfile + scipy resample)
# =============================================================================

def _rms_normalize(wave: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    rms = np.sqrt(np.mean(wave ** 2) + 1e-9)
    gain = (10.0 ** (target_db / 20.0)) / (rms + 1e-9)
    return wave * gain


class WaveformDataset(Dataset):
    """Manifest JSON: ``{"data": [{"audio_id": str}, ...]}``.

    ``audio_path = audio_root/audio_id`` (any-SR mono/stereo wav). Output:
    float tensor (1, clip_samples) at `sample_rate`, RMS-normalised.
    """

    def __init__(self, manifest: str, audio_root: str, sample_rate: int,
                 clip_samples: int, train: bool = True):
        with open(manifest) as f:
            self.samples = json.load(f)["data"]
        if not self.samples:
            raise ValueError(f"Empty manifest: {manifest}")
        self.audio_root = audio_root
        self.sample_rate = sample_rate
        self.clip_samples = clip_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_or_pad(self, wave: np.ndarray) -> np.ndarray:
        n = self.clip_samples
        L = wave.shape[0]
        if L == n:
            return wave
        if L > n:
            start = random.randint(0, L - n) if self.train else 0
            return wave[start:start + n]
        return np.pad(wave, (0, n - L))

    def _load(self, path: str) -> torch.Tensor:
        wave, sr = sf.read(path)
        if wave.ndim > 1:
            wave = wave[:, 0]                                 # force mono
        if sr != self.sample_rate:
            wave = signal.resample_poly(wave, self.sample_rate, sr)
        wave = _rms_normalize(np.asarray(wave, dtype=np.float32))
        wave = self._crop_or_pad(np.ascontiguousarray(wave, dtype=np.float32))
        return torch.from_numpy(wave).float().reshape(1, -1)  # (1, clip_samples)

    def __getitem__(self, idx: int) -> dict:
        path = os.path.join(self.audio_root, self.samples[idx]["audio_id"])
        return {"audio": self._load(path), "audio_path": path}


def collate_fn(batch: list[dict]) -> dict:
    return {
        "audio": torch.stack([b["audio"] for b in batch]),   # (B, 1, clip)
        "audio_path": [b["audio_path"] for b in batch],
    }


# =============================================================================
# Shared helpers
# =============================================================================

def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class FirstSampleLogCallback(TrainerCallback):
    def __init__(self):
        self.done = False

    def mark(self, audio_path, audio):
        if self.done:
            return
        self.done = True
        logger.info(
            "==== first batch / first sample (pre-feed) ====\n"
            "  wav path : %s\n  audio    : shape=%s dtype=%s",
            audio_path[0] if audio_path else "<unknown>",
            tuple(audio.shape), audio.dtype,
        )


@dataclass
class _Cfg:
    raw: dict

    def sec(self, k: str) -> dict:
        return self.raw.get(k, {}) or {}


def _build_model_kwargs(c: _Cfg) -> dict:
    m, d = c.sec("model"), c.sec("data")
    kw: dict = dict(
        sample_rate=int(d.get("sample_rate", 24_000)),
        clip_seconds=float(d.get("clip_seconds", 15.0)),
    )
    keys = ("conv_channels", "conv_strides", "stem_kernel", "use_daam",
            "daam_k", "daam_alpha", "num_layers", "num_heads", "ff_mult",
            "conv_kernel", "dropout", "pred_layers", "pred_heads", "mask_ratio",
            "mask_min_span", "ema_decay", "fsq_dim", "fsq_level", "pack_group",
            "dec_channels", "dec_min_channels", "lambda_stft", "lambda_gan",
            "disc_warmup")
    for k in keys:
        if k in m:
            v = m[k]
            kw[k] = tuple(v) if isinstance(v, list) else v
    return kw


def _common_targs(t: dict, o: dict, output_dir: str, seed: int, use_cpu: bool,
                  eval_enabled: bool, max_grad_norm: float) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        max_steps=int(o.get("max_steps", 24_000)),
        per_device_train_batch_size=int(o.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(o.get("per_device_eval_batch_size",
                                             o.get("per_device_train_batch_size", 8))),
        gradient_accumulation_steps=int(o.get("gradient_accumulation_steps", 1)),
        learning_rate=float(o.get("learning_rate", 1.5e-4)),
        weight_decay=float(o.get("weight_decay", 1e-3)),
        adam_beta1=float(o.get("adam_beta1", 0.8)),
        adam_beta2=float(o.get("adam_beta2", 0.99)),
        warmup_steps=int(o.get("warmup_steps", 2_000)),
        logging_steps=int(t.get("logging_steps", 50)),
        save_steps=int(t.get("save_steps", 2_000)),
        save_total_limit=t.get("save_total_limit"),
        eval_strategy=("steps" if eval_enabled else "no"),
        eval_steps=int(t.get("eval_steps", 2_000)),
        dataloader_num_workers=int(t.get("num_workers", 4)),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
        label_names=[],
        use_cpu=use_cpu,
        ddp_backend=("gloo" if use_cpu else None),
        max_grad_norm=max_grad_norm,
    )


# =============================================================================
# Stage 1 — masked-latent JEPA SSL
# =============================================================================

class Stage1HFConfig(PretrainedConfig):
    model_type = "neural_tokenizer_stage1"

    def __init__(self, model_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.model_kwargs = model_kwargs or {}


class Stage1Model(PreTrainedModel):
    config_class = Stage1HFConfig

    def __init__(self, config: Stage1HFConfig):
        super().__init__(config)
        self.s1 = Stage1Trainer(NeuralTokenizerConfig(**config.model_kwargs))
        self.last_metrics: dict = {}
        self.post_init()

    def _init_weights(self, module):
        pass

    def forward(self, audio=None, **_) -> dict:
        out = self.s1(audio)
        self.last_metrics = {"mask_frac": float(out["mask_frac"])}
        return {"loss": out["loss"]}

    @torch.no_grad()
    def ema_step(self):
        self.s1.ema_step()


class Stage1ParamCount(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        s = _unwrap(model).s1
        groups = {"encoder (Conv+DAAM+Conformer)": s.encoder,
                  "predictor": s.predictor}
        logger.info("==== trainable parameters per module ====")
        for name, mod in groups.items():
            logger.info("  %-32s %15s", name, f"{_count(mod):,}")
        logger.info("  %-32s %15s", "TOTAL trainable", f"{_count(s):,}")
        logger.info("  %-32s %15s (EMA, frozen)", "teacher",
                    f"{sum(p.numel() for p in s.teacher.parameters()):,}")
        logger.info("  encoder hop=%d  frame_rate=%.3f Hz", s.encoder.hop,
                    s.config.sample_rate / s.encoder.hop)
        logger.info("=========================================")


class Stage1EMACallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kw):
        _unwrap(model).ema_step()


class Stage1Metrics(TrainerCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kw):
        if logs is not None and model is not None:
            for k, v in _unwrap(model).last_metrics.items():
                logs[k] = round(float(v), 6)


class Stage1Save(TrainerCallback):
    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        s = _unwrap(model).s1
        sd = {k: v.detach().cpu() for k, v in s.state_dict().items()}
        out = os.path.join(args.output_dir, f"nt_stage1_step{state.global_step}.pt")
        torch.save(sd, out)
        logger.info("saved Stage-1 checkpoint (%d tensors) -> %s", len(sd), out)


class Stage1HFTrainer(Trainer):
    def __init__(self, *a, first_cb=None, **kw):
        super().__init__(*a, **kw)
        self._first_cb = first_cb

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_cb is not None:
            self._first_cb.mark(inputs.get("audio_path"), inputs["audio"])
        out = model(audio=inputs["audio"])
        return (out["loss"], out) if return_outputs else out["loss"]

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            decay, no_decay = [], []
            for _, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if p.ndim <= 1 else decay).append(p)
            self.optimizer = torch.optim.AdamW(
                [{"params": decay, "weight_decay": self.args.weight_decay},
                 {"params": no_decay, "weight_decay": 0.0}],
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon)
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps)
        return self.optimizer, self.lr_scheduler


def _run_stage1(cfg: _Cfg, model_kwargs, d, t, o, seed, use_cpu):
    model = Stage1Model(Stage1HFConfig(model_kwargs=model_kwargs))
    init_from = t.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location="cpu", weights_only=True)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.s1.load_state_dict(sd, strict=False)
        if len(sd) - len(unexpected) == 0:
            raise RuntimeError(f"init_from={init_from!r}: 0 keys applied — aborting.")
        logger.info("init_from %s: applied %d/%d keys",
                    init_from, len(sd) - len(unexpected), len(sd))

    nt_cfg = model.s1.config
    train_ds = WaveformDataset(d["train_manifest"], d["audio_root"],
                               nt_cfg.sample_rate, nt_cfg.clip_samples, train=True)
    eval_ds = (WaveformDataset(d["valid_manifest"], d["audio_root"],
                               nt_cfg.sample_rate, nt_cfg.clip_samples, train=False)
               if d.get("valid_manifest") else None)

    targs = _common_targs(t, o, t["output_dir"], seed, use_cpu,
                          eval_ds is not None, float(o.get("max_grad_norm", 1.0)))
    first_cb = FirstSampleLogCallback()
    trainer = Stage1HFTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collate_fn, first_cb=first_cb,
        callbacks=[Stage1ParamCount(), first_cb, Stage1EMACallback(),
                   Stage1Metrics(), Stage1Save()])
    trainer.train()
    trainer.save_model(os.path.join(targs.output_dir, "final"))
    logger.info("Stage-1 training complete -> %s/final", targs.output_dir)


# =============================================================================
# Stage 2 — FSQ + HiFi-GAN reconstruction (adversarial)
# =============================================================================

class Stage2HFConfig(PretrainedConfig):
    model_type = "neural_tokenizer_stage2"

    def __init__(self, model_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.model_kwargs = model_kwargs or {}


class Stage2Model(PreTrainedModel):
    config_class = Stage2HFConfig

    def __init__(self, config: Stage2HFConfig):
        super().__init__(config)
        self.gen = Stage2Trainer(NeuralTokenizerConfig(**config.model_kwargs))
        self.post_init()

    def _init_weights(self, module):
        pass

    def forward(self, audio=None, **_) -> dict:
        x_hat, x_real = self.gen.generator_forward(audio)
        return {"loss": self.gen.generator_loss(x_real, x_hat, False)["loss"]}


# deployable keys (encoder + FSQ proj + decoder) — drop discriminators / stft.
_DEPLOY_PREFIXES = ("encoder.", "proj_in.", "proj_out.", "decoder.")


class Stage2GANTrainer(Trainer):
    """Custom GAN loop: owns the discriminator optimizer; updates D then G
    inside `training_step` (HF steps the generator via `self.optimizer`)."""

    def __init__(self, *a, first_cb=None, max_grad_norm=1.0, **kw):
        super().__init__(*a, **kw)
        self._first_cb = first_cb
        self._clip = max_grad_norm
        self._metrics: dict = {}
        self.d_optimizer = None

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        gen = _unwrap(self.model).gen
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                gen.generator_parameters(), lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.weight_decay, eps=self.args.adam_epsilon)
        if self.d_optimizer is None:
            self.d_optimizer = torch.optim.AdamW(
                gen.discriminator_parameters(),
                lr=self.args.learning_rate * 0.5,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.weight_decay, eps=self.args.adam_epsilon)
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps)
        return self.optimizer, self.lr_scheduler

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        gen = _unwrap(model).gen
        wav = inputs["audio"].to(self.args.device)
        if self._first_cb is not None:
            self._first_cb.mark(inputs.get("audio_path"), wav)

        x_hat, x_real = gen.generator_forward(wav)
        use_gan = self.state.global_step >= gen.config.disc_warmup

        # ---- discriminator step ----
        if use_gan:
            d_loss = gen.discriminator_loss(x_real, x_hat)
            self.d_optimizer.zero_grad(set_to_none=True)
            self.accelerator.backward(d_loss)
            torch.nn.utils.clip_grad_norm_(gen.discriminator_parameters(), self._clip)
            self.d_optimizer.step()
            self._metrics["d_loss"] = round(float(d_loss.detach()), 6)

        # ---- generator step (backward only; HF loop calls optimizer.step) ----
        dps = gen.discriminator_parameters()
        for p in dps:                                        # freeze D for G backward
            p.requires_grad_(False)
        g = gen.generator_loss(x_real, x_hat, use_gan)
        g_loss = g["loss"] / self.args.gradient_accumulation_steps
        self.accelerator.backward(g_loss)
        for p in dps:
            p.requires_grad_(True)
        torch.nn.utils.clip_grad_norm_(gen.generator_parameters(), self._clip)
        self._metrics.update({k: round(float(v), 6) for k, v in g.items()
                              if k != "loss"})
        return g_loss.detach()

    def log(self, logs, *args, **kwargs):
        logs.update(self._metrics)
        super().log(logs, *args, **kwargs)


class Stage2Save(TrainerCallback):
    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        gen = _unwrap(model).gen
        sd = {k: v.detach().cpu() for k, v in gen.state_dict().items()
              if k.startswith(_DEPLOY_PREFIXES)}
        out = os.path.join(args.output_dir, f"nt_stage2_step{state.global_step}.pt")
        torch.save(sd, out)
        logger.info("saved Stage-2 (deployable) checkpoint (%d tensors) -> %s",
                    len(sd), out)


class Stage2ParamCount(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        gen = _unwrap(model).gen
        logger.info("==== trainable parameters per module ====")
        logger.info("  %-28s %15s (frozen)", "encoder (Stage-1)",
                    f"{sum(p.numel() for p in gen.encoder.parameters()):,}")
        for name, mod in (("proj_in", gen.proj_in), ("proj_out", gen.proj_out),
                          ("decoder (HiFi-GAN)", gen.decoder),
                          ("mpd (disc)", gen.mpd), ("msd (disc)", gen.msd)):
            logger.info("  %-28s %15s", name, f"{_count(mod):,}")
        logger.info("  %-28s %15s", "generator trainable",
                    f"{sum(p.numel() for p in gen.generator_parameters()):,}")
        logger.info("  token rate = %.1f tok/s (%d groups @ %.3f Hz)",
                    (gen.config.sample_rate / gen.encoder.hop) * gen.codec.n_groups,
                    gen.codec.n_groups, gen.config.sample_rate / gen.encoder.hop)
        logger.info("=========================================")


def _run_stage2(cfg: _Cfg, model_kwargs, d, t, o, seed, use_cpu):
    model = Stage2Model(Stage2HFConfig(model_kwargs=model_kwargs))
    stage1_ckpt = t.get("stage1_ckpt")
    if not stage1_ckpt:
        raise ValueError("stage 2 requires train.stage1_ckpt (frozen encoder).")
    blob = torch.load(stage1_ckpt, map_location="cpu", weights_only=True)
    sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    applied, total = model.gen.load_stage1_encoder(sd)
    logger.info("loaded Stage-1 encoder from %s: %d/%d keys",
                stage1_ckpt, applied, total)

    nt_cfg = model.gen.config
    train_ds = WaveformDataset(d["train_manifest"], d["audio_root"],
                               nt_cfg.sample_rate, nt_cfg.clip_samples, train=True)
    targs = _common_targs(t, o, t["output_dir"], seed, use_cpu,
                          eval_enabled=False, max_grad_norm=0.0)  # GAN clips manually
    first_cb = FirstSampleLogCallback()
    trainer = Stage2GANTrainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=collate_fn, first_cb=first_cb,
        max_grad_norm=float(o.get("max_grad_norm", 1.0)),
        callbacks=[Stage2ParamCount(), first_cb, Stage2Save()])
    trainer.train()
    trainer.save_model(os.path.join(targs.output_dir, "final"))
    logger.info("Stage-2 training complete -> %s/final", targs.output_dir)


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = _Cfg(yaml.safe_load(f))
    d, t, o = cfg.sec("data"), cfg.sec("train"), cfg.sec("optim")

    stage = int(cfg.raw.get("stage", 1))
    seed = int(t.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_kwargs = _build_model_kwargs(cfg)
    logger.info("stage=%d  NeuralTokenizer kwargs: %s", stage, model_kwargs)

    use_cpu = not torch.cuda.is_available()
    if stage == 1:
        _run_stage1(cfg, model_kwargs, d, t, o, seed, use_cpu)
    elif stage == 2:
        _run_stage2(cfg, model_kwargs, d, t, o, seed, use_cpu)
    else:
        raise ValueError(f"stage must be 1 or 2, got {stage}")


if __name__ == "__main__":
    main()
