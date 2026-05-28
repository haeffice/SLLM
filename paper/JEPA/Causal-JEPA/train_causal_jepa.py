"""Causal-JEPA object-centric self-supervised pre-training (PyTorch 2.8).

A Slot-Attention encoder turns each frame into object slots; a spatial-
broadcast decoder reconstructs frames (so slots stay object-like); and a
masked slot predictor must infer **masked whole objects** (a latent
intervention) and **future** slots from the visible ones — optimised with a
masked latent MSE plus the reconstruction aux loss.

Stack (per `paper/CLAUDE.md`):
    * `transformers.PreTrainedModel` wrapper around `CausalJEPA`
    * `transformers.Trainer` training loop
    * `torch.utils.data.Dataset` / `DataLoader` clip pipeline
    * `accelerate` + `torchrun` for multi-GPU (CPU-compatible)
    * logs: per-module trainable params; first batch's first sample video
      PATH before the model is fed; step / train_loss / pred_loss /
      recon_loss / history_loss / future_loss / lr every `logging_steps`;
      checkpoints every `save_steps` with the step number in the filename;
      abort if an init checkpoint fails to load.

Run via `run_train_CausalJEPA.sh config.yaml` (the shell enforces the
"no pre-existing checkpoints" guard). All arguments live in `config.yaml`.
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
import torch
import torch.nn as nn
import yaml
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

from CausalJEPA import CausalJEPA, CausalJEPAConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_causal_jepa")


# =============================================================================
# Dataset — manifest-driven clip loader (.npy)
# =============================================================================

class VideoClipDataset(Dataset):
    """Manifest JSON: ``{"data": [{"video_id": str}, ...]}``.

    ``video_path = video_root/video_id`` (``.npy`` (T, H, W, 3) uint8/float or
    (3, T, H, W) float). Output clip: float tensor (T, 3, img_size, img_size)
    in [0, 1] — slot encoders operate per-frame, so time is the leading axis.
    """

    def __init__(self, manifest: str, video_root: str, num_frames: int,
                 img_size: int, train: bool = True):
        with open(manifest) as f:
            self.samples = json.load(f)["data"]
        if not self.samples:
            raise ValueError(f"Empty manifest: {manifest}")
        self.video_root = video_root
        self.num_frames = num_frames
        self.img_size = img_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_frame_idx(self, total: int) -> np.ndarray:
        n = self.num_frames
        if total <= n:
            return np.clip(np.arange(n), 0, total - 1)
        start = random.randint(0, total - n) if self.train else (total - n) // 2
        return np.arange(start, start + n)

    def _resize(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (T, 3, H, W) float in [0, 1] -> (T, 3, S, S)
        return torch.nn.functional.interpolate(
            clip, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False)

    def _load(self, path: str) -> torch.Tensor:
        arr = np.load(path)
        if arr.ndim != 4:
            raise ValueError(f"{path}: expected 4D array, got {arr.shape}")
        t = torch.from_numpy(arr).float()
        if arr.shape[0] == 3 and arr.shape[1] != 3:       # (3, T, H, W)
            clip = t.permute(1, 0, 2, 3)                  # (T, 3, H, W)
        else:                                             # (T, H, W, 3)
            clip = t.permute(0, 3, 1, 2)
        if clip.max() > 1.5:
            clip = clip / 255.0
        idx = self._sample_frame_idx(clip.shape[0])
        return self._resize(clip[idx])                    # (T, 3, S, S)

    def __getitem__(self, idx: int) -> dict:
        path = os.path.join(self.video_root, self.samples[idx]["video_id"])
        return {"video": self._load(path), "video_path": path}


def collate_fn(batch: list[dict]) -> dict:
    return {
        "video": torch.stack([b["video"] for b in batch]),   # (B, T, 3, S, S)
        "video_path": [b["video_path"] for b in batch],
    }


# =============================================================================
# transformers wrapper
# =============================================================================

class CausalJEPAHFConfig(PretrainedConfig):
    model_type = "causal_jepa"

    def __init__(self, model_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.model_kwargs = model_kwargs or {}


class CausalJEPAModel(PreTrainedModel):
    config_class = CausalJEPAHFConfig

    def __init__(self, config: CausalJEPAHFConfig):
        super().__init__(config)
        self.cjepa = CausalJEPA(CausalJEPAConfig(**config.model_kwargs))
        self.post_init()

    def _init_weights(self, module):  # CausalJEPA self-initialises.
        pass

    def forward(self, video=None, **_) -> dict:
        return self.cjepa(video)


# =============================================================================
# Callbacks
# =============================================================================

def _unwrap(model) -> CausalJEPAModel:
    return model.module if hasattr(model, "module") else model


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class ParamCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        c = _unwrap(model).cjepa
        groups = {
            "encoder (CNN frame enc)": c.encoder,
            "slot_attn (Slot Attention)": c.slot_attn,
            "decoder (broadcast)": c.decoder,
            "predictor (slot ViT)": c.predictor,
        }
        logger.info("==== trainable parameters per module ====")
        for name, mod in groups.items():
            logger.info("  %-30s %15s", name, f"{_count(mod):,}")
        logger.info("  %-30s %15s", "TOTAL trainable", f"{_count(c):,}")
        logger.info("  slots=%d  history_len=%d  num_frames=%d",
                    c.config.num_slots, c.config.history_len, c.config.num_frames)
        logger.info("=========================================")


class FirstSampleLogCallback(TrainerCallback):
    def __init__(self):
        self.done = False

    def mark(self, video_path, video):
        if self.done:
            return
        self.done = True
        logger.info(
            "==== first batch / first sample (pre-feed) ====\n"
            "  video path : %s\n  video      : shape=%s dtype=%s",
            video_path[0] if video_path else "<unknown>",
            tuple(video.shape), video.dtype,
        )


class LossLogCallback(TrainerCallback):
    """Surface the sub-losses (pred / recon / history / future) in the logs."""

    _KEYS = ("pred_loss", "recon_loss", "history_loss", "future_loss")

    def __init__(self):
        self.metrics: dict[str, float] = {}

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is not None:
            for k in self._KEYS:
                if k in self.metrics:
                    logs[k] = round(float(self.metrics[k]), 6)


class SaveModelCallback(TrainerCallback):
    """Dump a model-only `.pt` (filename carries the step) on each save."""

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        c = _unwrap(model).cjepa
        sd = {k: v.detach().cpu() for k, v in c.state_dict().items()}
        out = os.path.join(args.output_dir,
                           f"causal_jepa_step{state.global_step}.pt")
        torch.save(sd, out)
        logger.info("saved model checkpoint (%d tensors) -> %s", len(sd), out)


# =============================================================================
# Trainer subclass
# =============================================================================

class CausalJEPAHFTrainer(Trainer):
    def __init__(self, *a, first_cb=None, loss_cb=None, **kw):
        super().__init__(*a, **kw)
        self._first_cb = first_cb
        self._loss_cb = loss_cb

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_cb is not None:
            self._first_cb.mark(inputs.get("video_path"), inputs["video"])
        outputs = model(video=inputs["video"])
        if self._loss_cb is not None:
            self._loss_cb.metrics = {
                k: float(outputs[k]) for k in
                ("pred_loss", "recon_loss", "history_loss", "future_loss")
                if k in outputs
            }
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            decay, no_decay = [], []
            for _, p in self.model.named_parameters():
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


def _build_model_kwargs(c: _Cfg) -> dict:
    m, msk, ob = c.sec("model"), c.sec("masking"), c.sec("objective")
    d = c.sec("data")
    kw: dict = dict(
        img_size=int(d.get("img_size", 64)),
        num_frames=int(d.get("num_frames", 16)),
    )
    for src, keys in (
        (m, ("in_chans", "enc_channels", "num_slots", "slot_dim", "slot_iters",
             "slot_hidden", "dec_hidden", "pred_dim", "pred_depth", "pred_heads",
             "pred_mlp_ratio", "history_len", "freeze_encoder")),
        (msk, ("max_masked_slots",)),
        (ob, ("recon_weight", "history_weight", "future_weight")),
    ):
        for k in keys:
            if k in src:
                v = src[k]
                kw[k] = tuple(v) if isinstance(v, list) else v
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = _Cfg(yaml.safe_load(f))
    d, t, o = cfg.sec("data"), cfg.sec("train"), cfg.sec("optim")

    seed = int(t.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_kwargs = _build_model_kwargs(cfg)
    logger.info("CausalJEPA kwargs: %s", model_kwargs)
    model = CausalJEPAModel(CausalJEPAHFConfig(model_kwargs=model_kwargs))

    init_from = t.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location="cpu", weights_only=True)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.cjepa.load_state_dict(sd, strict=False)
        if len(sd) - len(unexpected) == 0:
            raise RuntimeError(
                f"init_from={init_from!r}: 0/{len(sd)} keys applied — aborting.")
        logger.info("init_from %s: applied %d/%d keys",
                    init_from, len(sd) - len(unexpected), len(sd))

    c_cfg = model.cjepa.config
    train_ds = VideoClipDataset(d["train_manifest"], d["video_root"],
                                c_cfg.num_frames, c_cfg.img_size, train=True)
    eval_ds = (VideoClipDataset(d["valid_manifest"], d["video_root"],
                                c_cfg.num_frames, c_cfg.img_size, train=False)
               if d.get("valid_manifest") else None)

    use_cpu = not torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=t["output_dir"],
        max_steps=int(o.get("max_steps", 100_000)),
        per_device_train_batch_size=int(o.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(o.get("per_device_eval_batch_size",
                                             o.get("per_device_train_batch_size", 16))),
        gradient_accumulation_steps=int(o.get("gradient_accumulation_steps", 1)),
        learning_rate=float(o.get("learning_rate", 5.0e-4)),
        weight_decay=float(o.get("weight_decay", 0.0)),
        adam_beta1=float(o.get("adam_beta1", 0.9)),
        adam_beta2=float(o.get("adam_beta2", 0.999)),
        warmup_steps=int(o.get("warmup_steps", 5_000)),
        logging_steps=int(t.get("logging_steps", 50)),
        save_steps=int(t.get("save_steps", 2_000)),
        save_total_limit=t.get("save_total_limit"),
        eval_strategy=("steps" if eval_ds is not None else "no"),
        eval_steps=int(t.get("eval_steps", 2_000)),
        dataloader_num_workers=int(t.get("num_workers", 4)),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
        label_names=[],
        use_cpu=use_cpu,
        ddp_backend=("gloo" if use_cpu else None),
        max_grad_norm=float(o.get("max_grad_norm", 1.0)),
    )

    first_cb, loss_cb = FirstSampleLogCallback(), LossLogCallback()
    trainer = CausalJEPAHFTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collate_fn,
        first_cb=first_cb, loss_cb=loss_cb,
        callbacks=[ParamCountCallback(), first_cb, loss_cb, SaveModelCallback()],
    )

    trainer.train()
    trainer.save_model(os.path.join(targs.output_dir, "final"))
    logger.info("training complete; final HF model -> %s/final", targs.output_dir)


if __name__ == "__main__":
    main()
