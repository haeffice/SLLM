"""EB-JEPA self-supervised image pre-training (PyTorch 2.8).

Energy-based JEPA pre-training (arXiv:2602.03604): a ResNet encoder + MLP
projector are trained so a predictor maps one augmented view's embedding to
the other's, with VICReg-style variance + covariance regularization (the
"energy") preventing collapse — no EMA teacher / stop-gradient.

Stack (per `paper/CLAUDE.md`):
    * `transformers.PreTrainedModel` wrapper around `EBJEPA`
    * `transformers.Trainer` training loop
    * `torch.utils.data.Dataset` / `DataLoader` two-view image pipeline
    * `accelerate` + `torchrun` for multi-GPU (CPU-compatible)
    * logs: per-module trainable params; first batch's first sample image
      PATH before the model is fed; step / train_loss / inv_loss /
      var_loss / cov_loss / valid_loss / lr every `logging_steps`;
      checkpoints every `save_steps` with the step number in the filename;
      abort if an init checkpoint fails to load.

Run via `run_train_EBJEPA.sh config.yaml` (the shell enforces the
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
import torch.nn.functional as F
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

from EBJEPA import EBJEPA, EBJEPAConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_ebjepa")


# =============================================================================
# Two-view augmentations (pure torch — no torchvision dependency)
# =============================================================================

def _rand_resized_crop(img: torch.Tensor, size: int,
                       scale=(0.4, 1.0)) -> torch.Tensor:
    """img (C, H, W) in [0, 1] -> (C, size, size) random-resized crop."""
    C, H, W = img.shape
    area = H * W
    for _ in range(10):
        target = random.uniform(*scale) * area
        ar = random.uniform(0.75, 1.333)
        w = int(round((target * ar) ** 0.5))
        h = int(round((target / ar) ** 0.5))
        if w <= W and h <= H:
            x0 = random.randint(0, W - w)
            y0 = random.randint(0, H - h)
            crop = img[:, y0:y0 + h, x0:x0 + w]
            break
    else:
        crop = img
    return F.interpolate(crop.unsqueeze(0), size=(size, size),
                         mode="bilinear", align_corners=False).squeeze(0)


def _augment(img: torch.Tensor, size: int) -> torch.Tensor:
    """One stochastic view: rand-resized-crop, h-flip, brightness/contrast
    jitter, optional grayscale."""
    x = _rand_resized_crop(img, size)
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-1])                          # horizontal flip
    if random.random() < 0.8:                                 # brightness
        x = x * random.uniform(0.6, 1.4)
    if random.random() < 0.8:                                 # contrast
        m = x.mean(dim=(-1, -2), keepdim=True)
        x = (x - m) * random.uniform(0.6, 1.4) + m
    if random.random() < 0.2:                                 # grayscale
        g = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2])
        x = g.unsqueeze(0).expand_as(x)
    return x.clamp(0, 1)


# =============================================================================
# Dataset — manifest-driven image loader producing two views
# =============================================================================

class TwoViewImageDataset(Dataset):
    """Manifest JSON: ``{"data": [{"image_id": str}, ...]}``.

    ``image_path = image_root/image_id`` (``.npy`` (H, W, 3) uint8/float).
    Returns two independently-augmented views of size ``img_size``.
    """

    def __init__(self, manifest: str, image_root: str, img_size: int):
        with open(manifest) as f:
            self.samples = json.load(f)["data"]
        if not self.samples:
            raise ValueError(f"Empty manifest: {manifest}")
        self.image_root = image_root
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[..., None].repeat(3, -1)
        if arr.max() > 1.5:
            arr = arr / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()   # (C,H,W)

    def __getitem__(self, idx: int) -> dict:
        path = os.path.join(self.image_root, self.samples[idx]["image_id"])
        img = self._load(path)
        return {"view1": _augment(img, self.img_size),
                "view2": _augment(img, self.img_size),
                "image_path": path}


def collate_fn(batch: list[dict]) -> dict:
    return {
        "view1": torch.stack([b["view1"] for b in batch]),
        "view2": torch.stack([b["view2"] for b in batch]),
        "image_path": [b["image_path"] for b in batch],
    }


# =============================================================================
# transformers wrapper
# =============================================================================

class EBJEPAHFConfig(PretrainedConfig):
    model_type = "eb_jepa"

    def __init__(self, model_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.model_kwargs = model_kwargs or {}


class EBJEPAModel(PreTrainedModel):
    config_class = EBJEPAHFConfig

    def __init__(self, config: EBJEPAHFConfig):
        super().__init__(config)
        self.eb = EBJEPA(EBJEPAConfig(**config.model_kwargs))
        self.post_init()

    def _init_weights(self, module):  # EBJEPA self-initialises.
        pass

    def forward(self, view1=None, view2=None, **_) -> dict:
        return self.eb.compute_loss(view1, view2)


# =============================================================================
# Callbacks
# =============================================================================

def _unwrap(model) -> EBJEPAModel:
    return model.module if hasattr(model, "module") else model


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class ParamCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        e = _unwrap(model).eb
        groups = {"encoder (ResNet)": e.encoder, "projector": e.projector,
                  "predictor": e.predictor}
        logger.info("==== trainable parameters per module ====")
        for name, mod in groups.items():
            logger.info("  %-22s %15s", name, f"{_count(mod):,}")
        logger.info("  %-22s %15s", "TOTAL trainable", f"{_count(e):,}")
        logger.info("=========================================")


class FirstSampleLogCallback(TrainerCallback):
    def __init__(self):
        self.done = False

    def mark(self, image_path: list[str], view1: torch.Tensor):
        if self.done:
            return
        self.done = True
        logger.info(
            "==== first batch / first sample (pre-feed) ====\n"
            "  image path : %s\n  view       : shape=%s dtype=%s",
            image_path[0] if image_path else "<unknown>",
            tuple(view1.shape), view1.dtype,
        )


class LossLogCallback(TrainerCallback):
    def __init__(self):
        self.inv = self.var = self.cov = None

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is not None:
            for name in ("inv", "var", "cov"):
                v = getattr(self, name)
                if v is not None:
                    logs[f"{name}_loss"] = round(float(v), 6)


class SaveModelCallback(TrainerCallback):
    """On every HF checkpoint also dump a model-only `.pt` whose filename
    carries the step number (loadable via `EBJEPA.from_checkpoint`)."""

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        e = _unwrap(model).eb
        sd = {k: v.detach().cpu() for k, v in e.state_dict().items()}
        out = os.path.join(args.output_dir, f"ebjepa_step{state.global_step}.pt")
        torch.save(sd, out)
        logger.info("saved model checkpoint (%d tensors) -> %s", len(sd), out)


# =============================================================================
# Trainer subclass
# =============================================================================

class EBJEPAHFTrainer(Trainer):
    def __init__(self, *a, first_cb=None, loss_cb=None, **kw):
        super().__init__(*a, **kw)
        self._first_cb = first_cb
        self._loss_cb = loss_cb

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_cb is not None:
            self._first_cb.mark(inputs.get("image_path"), inputs["view1"])
        outputs = model(view1=inputs["view1"], view2=inputs["view2"])
        if self._loss_cb is not None:
            self._loss_cb.inv = outputs.get("inv_loss")
            self._loss_cb.var = outputs.get("var_loss")
            self._loss_cb.cov = outputs.get("cov_loss")
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
    m, ob = c.sec("model"), c.sec("objective")
    kw: dict = {}
    for k in ("in_chans", "resnet_layers", "resnet_width", "proj_hidden",
              "proj_out", "pred_hidden"):
        if k in m:
            v = m[k]
            kw[k] = tuple(v) if isinstance(v, list) else v
    for k in ("std_coeff", "cov_coeff", "var_gamma", "eps"):
        if k in ob:
            kw[k] = ob[k]
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
    logger.info("EBJEPA kwargs: %s", model_kwargs)
    model = EBJEPAModel(EBJEPAHFConfig(model_kwargs=model_kwargs))

    init_from = t.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location="cpu", weights_only=True)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.eb.load_state_dict(sd, strict=False)
        if len(sd) - len(unexpected) == 0:
            raise RuntimeError(
                f"init_from={init_from!r}: 0/{len(sd)} keys applied — aborting.")
        logger.info("init_from %s: applied %d/%d keys",
                    init_from, len(sd) - len(unexpected), len(sd))

    img_size = int(d.get("img_size", 32))
    train_ds = TwoViewImageDataset(d["train_manifest"], d["image_root"], img_size)
    eval_ds = (TwoViewImageDataset(d["valid_manifest"], d["image_root"], img_size)
               if d.get("valid_manifest") else None)

    use_cpu = not torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=t["output_dir"],
        max_steps=int(o.get("max_steps", 50_000)),
        per_device_train_batch_size=int(o.get("per_device_train_batch_size", 256)),
        per_device_eval_batch_size=int(o.get("per_device_eval_batch_size",
                                             o.get("per_device_train_batch_size", 256))),
        gradient_accumulation_steps=int(o.get("gradient_accumulation_steps", 1)),
        learning_rate=float(o.get("learning_rate", 1.0e-3)),
        weight_decay=float(o.get("weight_decay", 1.0e-4)),
        adam_beta1=float(o.get("adam_beta1", 0.9)),
        adam_beta2=float(o.get("adam_beta2", 0.999)),
        warmup_steps=int(o.get("warmup_steps", 2_000)),
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
    trainer = EBJEPAHFTrainer(
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
