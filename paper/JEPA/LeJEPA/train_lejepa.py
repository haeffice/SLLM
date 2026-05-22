"""LeJEPA self-supervised image pre-training (PyTorch 2.8).

LeJEPA pre-training (arXiv:2511.08544): a ViT encoder + small JEPA predictor
are trained so each view's embedding predicts the *global* views' embeddings,
while **SIGReg** drives the pooled embedding distribution toward an isotropic
Gaussian — no EMA teacher / stop-gradient / scheduler, a single trade-off
hyper-parameter `sigreg_coeff`.

Stack (per `paper/CLAUDE.md`):
    * `transformers.PreTrainedModel` wrapper around `LeJEPA`
    * `transformers.Trainer` training loop
    * `torch.utils.data.Dataset` / `DataLoader` multi-crop image pipeline
    * `accelerate` + `torchrun` for multi-GPU (CPU-compatible)
    * logs: per-module trainable params; first batch's first sample image
      PATH before the model is fed; step / train_loss / pred_loss /
      reg_loss / valid_loss / lr every `logging_steps`; checkpoints every
      `save_steps` with the step number in the filename; abort if an init
      checkpoint fails to load.

Run via `run_train_LeJEPA.sh config.yaml` (the shell enforces the
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

from LeJEPA import LeJEPA, LeJEPAConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_lejepa")


# =============================================================================
# Multi-crop augmentations (pure torch — no torchvision dependency)
# =============================================================================

def _rand_resized_crop(img: torch.Tensor, size: int, scale) -> torch.Tensor:
    """img (C, H, W) in [0, 1] -> (C, size, size) random-resized crop."""
    C, H, W = img.shape
    area = H * W
    for _ in range(10):
        target = random.uniform(*scale) * area
        ar = random.uniform(0.75, 1.333)
        w = int(round((target * ar) ** 0.5))
        h = int(round((target / ar) ** 0.5))
        if 0 < w <= W and 0 < h <= H:
            x0 = random.randint(0, W - w)
            y0 = random.randint(0, H - h)
            crop = img[:, y0:y0 + h, x0:x0 + w]
            break
    else:
        crop = img
    return F.interpolate(crop.unsqueeze(0), size=(size, size),
                         mode="bilinear", align_corners=False).squeeze(0)


def _photometric(x: torch.Tensor) -> torch.Tensor:
    """Shared photometric jitter: h-flip, brightness/contrast, grayscale."""
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


def _view(img, size, scale):
    """One stochastic multi-crop view (global or local, set by size/scale)."""
    return _photometric(_rand_resized_crop(img, size, scale))


# =============================================================================
# Dataset — manifest-driven image loader producing multi-crop views
# =============================================================================

class MultiCropImageDataset(Dataset):
    """Manifest JSON: ``{"data": [{"image_id": str}, ...]}``.

    ``image_path = image_root/image_id`` (``.npy`` (H, W, 3) uint8/float).
    Returns ``num_global`` global crops (``global_size``) and ``num_local``
    local crops (``local_size``) of each image (DINO-style multi-crop).
    """

    def __init__(self, manifest: str, image_root: str, *, num_global: int,
                 num_local: int, global_size: int, local_size: int,
                 global_scale, local_scale):
        with open(manifest) as f:
            self.samples = json.load(f)["data"]
        if not self.samples:
            raise ValueError(f"Empty manifest: {manifest}")
        self.image_root = image_root
        self.num_global = num_global
        self.num_local = num_local
        self.global_size = global_size
        self.local_size = local_size
        self.global_scale = tuple(global_scale)
        self.local_scale = tuple(local_scale)

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
        globals_ = torch.stack([_view(img, self.global_size, self.global_scale)
                                for _ in range(self.num_global)])
        locals_ = (torch.stack([_view(img, self.local_size, self.local_scale)
                                for _ in range(self.num_local)])
                   if self.num_local > 0 else torch.empty(0))
        return {"globals": globals_, "locals": locals_, "image_path": path}


def collate_fn(batch: list[dict]) -> dict:
    out = {
        "globals": torch.stack([b["globals"] for b in batch]),   # (B,ng,C,H,W)
        "image_path": [b["image_path"] for b in batch],
    }
    locals_ = [b["locals"] for b in batch]
    out["locals"] = (torch.stack(locals_) if locals_[0].numel() > 0
                     else torch.empty(0))
    return out


# =============================================================================
# transformers wrapper
# =============================================================================

class LeJEPAHFConfig(PretrainedConfig):
    model_type = "lejepa"

    def __init__(self, model_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.model_kwargs = model_kwargs or {}


class LeJEPAModel(PreTrainedModel):
    config_class = LeJEPAHFConfig

    def __init__(self, config: LeJEPAHFConfig):
        super().__init__(config)
        self.le = LeJEPA(LeJEPAConfig(**config.model_kwargs))
        self.post_init()

    def _init_weights(self, module):  # LeJEPA self-initialises.
        pass

    def forward(self, globals=None, locals=None, **_) -> dict:
        loc = locals if (locals is not None and locals.numel() > 0) else None
        return self.le.compute_loss(globals, loc)


# =============================================================================
# Callbacks
# =============================================================================

def _unwrap(model) -> LeJEPAModel:
    return model.module if hasattr(model, "module") else model


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class ParamCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        le = _unwrap(model).le
        groups = {"encoder (ViT)": le.encoder, "predictor": le.predictor}
        logger.info("==== trainable parameters per module ====")
        for name, mod in groups.items():
            logger.info("  %-22s %15s", name, f"{_count(mod):,}")
        logger.info("  %-22s %15s", "TOTAL trainable", f"{_count(le):,}")
        logger.info("=========================================")


class FirstSampleLogCallback(TrainerCallback):
    def __init__(self):
        self.done = False

    def mark(self, image_path: list[str], globals_: torch.Tensor):
        if self.done:
            return
        self.done = True
        logger.info(
            "==== first batch / first sample (pre-feed) ====\n"
            "  image path : %s\n  globals    : shape=%s dtype=%s",
            image_path[0] if image_path else "<unknown>",
            tuple(globals_.shape), globals_.dtype,
        )


class LossLogCallback(TrainerCallback):
    def __init__(self):
        self.pred = self.reg = None

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is not None:
            for name in ("pred", "reg"):
                v = getattr(self, name)
                if v is not None:
                    logs[f"{name}_loss"] = round(float(v), 6)


class SaveModelCallback(TrainerCallback):
    """On every HF checkpoint also dump a model-only `.pt` whose filename
    carries the step number (loadable via `LeJEPA.from_checkpoint`)."""

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        le = _unwrap(model).le
        sd = {k: v.detach().cpu() for k, v in le.state_dict().items()}
        out = os.path.join(args.output_dir, f"lejepa_step{state.global_step}.pt")
        torch.save(sd, out)
        logger.info("saved model checkpoint (%d tensors) -> %s", len(sd), out)


# =============================================================================
# Trainer subclass
# =============================================================================

class LeJEPAHFTrainer(Trainer):
    def __init__(self, *a, first_cb=None, loss_cb=None, **kw):
        super().__init__(*a, **kw)
        self._first_cb = first_cb
        self._loss_cb = loss_cb

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_cb is not None:
            self._first_cb.mark(inputs.get("image_path"), inputs["globals"])
        outputs = model(globals=inputs["globals"], locals=inputs.get("locals"))
        if self._loss_cb is not None:
            self._loss_cb.pred = outputs.get("pred_loss")
            self._loss_cb.reg = outputs.get("reg_loss")
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
    m, v, ob = c.sec("model"), c.sec("views"), c.sec("objective")
    kw: dict = {}
    for k in ("in_chans", "img_size", "patch_size", "embed_dim", "depth",
              "num_heads", "mlp_ratio", "pred_hidden"):
        if k in m:
            kw[k] = m[k]
    for k in ("num_global", "num_local"):
        if k in v:
            kw[k] = v[k]
    for k in ("sigreg_coeff", "num_slices", "num_points", "probe_last_layers"):
        if k in ob:
            kw[k] = ob[k]
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = _Cfg(yaml.safe_load(f))
    d, v, t, o = cfg.sec("data"), cfg.sec("views"), cfg.sec("train"), cfg.sec("optim")

    seed = int(t.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_kwargs = _build_model_kwargs(cfg)
    logger.info("LeJEPA kwargs: %s", model_kwargs)
    model = LeJEPAModel(LeJEPAHFConfig(model_kwargs=model_kwargs))

    init_from = t.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location="cpu", weights_only=True)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.le.load_state_dict(sd, strict=False)
        if len(sd) - len(unexpected) == 0:
            raise RuntimeError(
                f"init_from={init_from!r}: 0/{len(sd)} keys applied — aborting.")
        logger.info("init_from %s: applied %d/%d keys",
                    init_from, len(sd) - len(unexpected), len(sd))

    ds_kw = dict(
        num_global=int(v.get("num_global", 2)),
        num_local=int(v.get("num_local", 6)),
        global_size=int(v.get("global_size", 224)),
        local_size=int(v.get("local_size", 98)),
        global_scale=v.get("global_scale", [0.3, 1.0]),
        local_scale=v.get("local_scale", [0.05, 0.3]),
    )
    train_ds = MultiCropImageDataset(d["train_manifest"], d["image_root"], **ds_kw)
    eval_ds = (MultiCropImageDataset(d["valid_manifest"], d["image_root"], **ds_kw)
               if d.get("valid_manifest") else None)

    use_cpu = not torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=t["output_dir"],
        max_steps=int(o.get("max_steps", 100_000)),
        per_device_train_batch_size=int(o.get("per_device_train_batch_size", 128)),
        per_device_eval_batch_size=int(o.get("per_device_eval_batch_size",
                                             o.get("per_device_train_batch_size", 128))),
        gradient_accumulation_steps=int(o.get("gradient_accumulation_steps", 1)),
        learning_rate=float(o.get("learning_rate", 5.0e-4)),
        weight_decay=float(o.get("weight_decay", 5.0e-2)),
        adam_beta1=float(o.get("adam_beta1", 0.9)),
        adam_beta2=float(o.get("adam_beta2", 0.999)),
        warmup_steps=int(o.get("warmup_steps", 5_000)),
        logging_steps=int(t.get("logging_steps", 50)),
        save_steps=int(t.get("save_steps", 2_000)),
        save_total_limit=t.get("save_total_limit"),
        eval_strategy=("steps" if eval_ds is not None else "no"),
        eval_steps=int(t.get("eval_steps", 2_000)),
        dataloader_num_workers=int(t.get("num_workers", 4)),
        bf16=bool(o.get("bf16", False)) and not use_cpu,
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
        label_names=[],
        use_cpu=use_cpu,
        ddp_backend=("gloo" if use_cpu else None),
        max_grad_norm=float(o.get("max_grad_norm", 1.0)),
    )

    first_cb, loss_cb = FirstSampleLogCallback(), LossLogCallback()
    trainer = LeJEPAHFTrainer(
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
