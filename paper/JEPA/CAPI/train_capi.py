"""CAPI self-supervised image pre-training (PyTorch 2.8).

CAPI pre-training (arXiv:2502.08769): a student ViT encodes the *visible*
patches, a cross-attention predictor fills in the *masked* patches, and the
training target is the cluster assignment that an EMA teacher gives those
masked patches — balanced online by Sinkhorn-Knopp over learnable prototypes.
No pixel decoder, no contrastive pairs.

Stack (per `paper/CLAUDE.md`):
    * `transformers.PreTrainedModel` wrapper around `CAPI`
    * `transformers.Trainer` training loop
    * `torch.utils.data.Dataset` / `DataLoader` masked-patch image pipeline
    * `accelerate` + `torchrun` for multi-GPU (CPU-compatible)
    * EMA teacher updated every optimiser step (mu = 1 - lr by default)
    * clustering prototypes optimised at 0.5x the backbone learning rate
    * logs: per-module trainable params; first batch's first sample image
      PATH before the model is fed; step / train_loss / ce_loss /
      target_entropy / valid_loss / lr every `logging_steps`; checkpoints
      every `save_steps` with the step number in the filename; abort if an
      init checkpoint fails to load.

Run via `run_train_CAPI.sh config.yaml` (the shell enforces the
"no pre-existing checkpoints" guard). All arguments live in `config.yaml`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

from CAPI import CAPI, CAPIConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_capi")


# =============================================================================
# Augmentation + inverse-block masking (pure torch — no torchvision)
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
    x = F.interpolate(crop.unsqueeze(0), size=(size, size),
                      mode="bilinear", align_corners=False).squeeze(0)
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-1])                          # horizontal flip
    return x.clamp(0, 1)


def _inverse_block_mask(grid: int, num_keep: int):
    """Keep a contiguous block (with toroidal roll) visible, mask the rest.

    Returns (visible_idx (num_keep,), masked_idx (N - num_keep,)) as long
    tensors of patch indices on the row-major grid."""
    n = grid * grid
    ar = random.uniform(0.5, 2.0)                             # block aspect
    hb = max(1, min(grid, int(round((num_keep * ar) ** 0.5))))
    wb = max(1, min(grid, math.ceil(num_keep / hb)))
    top = random.randint(0, grid - 1)
    left = random.randint(0, grid - 1)
    rows = [(top + i) % grid for i in range(hb)]
    cols = [(left + j) % grid for j in range(wb)]
    block = [r * grid + c for r in rows for c in cols]        # >= num_keep
    random.shuffle(block)
    visible = sorted(block[:num_keep])
    vis_set = set(visible)
    masked = [i for i in range(n) if i not in vis_set]
    return (torch.tensor(visible, dtype=torch.long),
            torch.tensor(masked, dtype=torch.long))


# =============================================================================
# Dataset — manifest-driven image loader producing one view + a patch mask
# =============================================================================

class MaskedImageDataset(Dataset):
    """Manifest JSON: ``{"data": [{"image_id": str}, ...]}``.

    ``image_path = image_root/image_id`` (``.npy`` (H, W, 3) uint8/float).
    Returns one random-resized-crop view and a visible/masked patch split.
    """

    def __init__(self, manifest: str, image_root: str, *, img_size: int,
                 patch_size: int, mask_ratio: float, crop_scale):
        with open(manifest) as f:
            self.samples = json.load(f)["data"]
        if not self.samples:
            raise ValueError(f"Empty manifest: {manifest}")
        self.image_root = image_root
        self.img_size = img_size
        self.grid = img_size // patch_size
        n = self.grid * self.grid
        self.num_keep = max(1, round((1.0 - mask_ratio) * n))
        self.crop_scale = tuple(crop_scale)

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
        img = _rand_resized_crop(self._load(path), self.img_size,
                                 self.crop_scale)
        visible_idx, masked_idx = _inverse_block_mask(self.grid, self.num_keep)
        return {"image": img, "visible_idx": visible_idx,
                "masked_idx": masked_idx, "image_path": path}


def collate_fn(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "visible_idx": torch.stack([b["visible_idx"] for b in batch]),
        "masked_idx": torch.stack([b["masked_idx"] for b in batch]),
        "image_path": [b["image_path"] for b in batch],
    }


# =============================================================================
# transformers wrapper
# =============================================================================

class CAPIHFConfig(PretrainedConfig):
    model_type = "capi"

    def __init__(self, model_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.model_kwargs = model_kwargs or {}


class CAPIModel(PreTrainedModel):
    config_class = CAPIHFConfig

    def __init__(self, config: CAPIHFConfig):
        super().__init__(config)
        self.cap = CAPI(CAPIConfig(**config.model_kwargs))
        self.post_init()

    def _init_weights(self, module):  # CAPI self-initialises.
        pass

    def forward(self, image=None, visible_idx=None, masked_idx=None, **_) -> dict:
        return self.cap.compute_loss(image, visible_idx, masked_idx)


# =============================================================================
# Callbacks
# =============================================================================

def _unwrap(model) -> CAPIModel:
    return model.module if hasattr(model, "module") else model


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class ParamCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        cap = _unwrap(model).cap
        groups = {"encoder (ViT student)": cap.encoder,
                  "predictor (cross-attn)": cap.predictor}
        logger.info("==== trainable parameters per module ====")
        for name, mod in groups.items():
            logger.info("  %-24s %15s", name, f"{_count(mod):,}")
        logger.info("  %-24s %15s", "prototypes",
                    f"{cap.prototypes.numel():,}")
        logger.info("  %-24s %15s", "TOTAL trainable", f"{_count(cap):,}")
        logger.info("  %-24s %15s (EMA, frozen)", "teacher",
                    f"{sum(p.numel() for p in cap.teacher.parameters()):,}")
        logger.info("=========================================")


class FirstSampleLogCallback(TrainerCallback):
    def __init__(self):
        self.done = False

    def mark(self, image_path, image, visible_idx, masked_idx):
        if self.done:
            return
        self.done = True
        n_vis, n_mask = visible_idx.shape[1], masked_idx.shape[1]
        logger.info(
            "==== first batch / first sample (pre-feed) ====\n"
            "  image path : %s\n  image      : shape=%s dtype=%s\n"
            "  patches    : %d visible / %d masked (of %d)",
            image_path[0] if image_path else "<unknown>",
            tuple(image.shape), image.dtype, n_vis, n_mask, n_vis + n_mask,
        )


class LossLogCallback(TrainerCallback):
    def __init__(self):
        self.ce = self.ent = None

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is not None:
            if self.ce is not None:
                logs["ce_loss"] = round(float(self.ce), 6)
            if self.ent is not None:
                logs["target_entropy"] = round(float(self.ent), 4)


class EMACallback(TrainerCallback):
    """Update the EMA teacher after every optimiser step (mu = 1 - lr by
    default unless `ema_momentum` is pinned in the config)."""

    def __init__(self, model: CAPIModel, follow_lr: bool):
        self.cap = model.cap
        self.follow_lr = follow_lr

    def on_step_end(self, args, state, control, **kw):
        m = None
        if self.follow_lr:
            lrs = kw.get("lr_scheduler")
            if lrs is not None:
                last = lrs.get_last_lr()
                if last:
                    m = max(0.0, min(1.0, 1.0 - float(last[0])))
        self.cap.update_teacher(m)


class SaveModelCallback(TrainerCallback):
    """On every HF checkpoint also dump a model-only `.pt` whose filename
    carries the step number (loadable via `CAPI.from_checkpoint`)."""

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        cap = _unwrap(model).cap
        sd = {k: v.detach().cpu() for k, v in cap.state_dict().items()}
        out = os.path.join(args.output_dir, f"capi_step{state.global_step}.pt")
        torch.save(sd, out)
        logger.info("saved model checkpoint (%d tensors) -> %s", len(sd), out)


# =============================================================================
# Trainer subclass
# =============================================================================

class CAPIHFTrainer(Trainer):
    def __init__(self, *a, first_cb=None, loss_cb=None, proto_lr_scale=0.5, **kw):
        super().__init__(*a, **kw)
        self._first_cb = first_cb
        self._loss_cb = loss_cb
        self._proto_lr_scale = proto_lr_scale

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_cb is not None:
            self._first_cb.mark(inputs.get("image_path"), inputs["image"],
                                inputs["visible_idx"], inputs["masked_idx"])
        outputs = model(image=inputs["image"], visible_idx=inputs["visible_idx"],
                        masked_idx=inputs["masked_idx"])
        if self._loss_cb is not None:
            self._loss_cb.ce = outputs.get("ce_loss")
            self._loss_cb.ent = outputs.get("target_entropy")
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            decay, no_decay, proto = [], [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if n.endswith("prototypes"):
                    proto.append(p)
                elif p.ndim <= 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
            lr = self.args.learning_rate
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": self.args.weight_decay,
                     "lr": lr},
                    {"params": no_decay, "weight_decay": 0.0, "lr": lr},
                    {"params": proto, "weight_decay": 0.0,
                     "lr": lr * self._proto_lr_scale},   # clustering at 0.5x lr
                ],
                lr=lr,
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
    for k in ("in_chans", "img_size", "patch_size", "embed_dim", "depth",
              "num_heads", "mlp_ratio", "num_registers", "pred_depth",
              "pred_heads"):
        if k in m:
            kw[k] = m[k]
    for k in ("num_prototypes", "mask_ratio", "sinkhorn_eps", "sinkhorn_iters",
              "student_temp", "ema_momentum"):
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
    logger.info("CAPI kwargs: %s", model_kwargs)
    model = CAPIModel(CAPIHFConfig(model_kwargs=model_kwargs))

    init_from = t.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location="cpu", weights_only=True)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.cap.load_state_dict(sd, strict=False)
        if len(sd) - len(unexpected) == 0:
            raise RuntimeError(
                f"init_from={init_from!r}: 0/{len(sd)} keys applied — aborting.")
        logger.info("init_from %s: applied %d/%d keys",
                    init_from, len(sd) - len(unexpected), len(sd))

    cap_cfg = model.cap.config
    ds_kw = dict(
        img_size=cap_cfg.img_size, patch_size=cap_cfg.patch_size,
        mask_ratio=cap_cfg.mask_ratio, crop_scale=d.get("crop_scale", [0.6, 1.0]),
    )
    train_ds = MaskedImageDataset(d["train_manifest"], d["image_root"], **ds_kw)
    eval_ds = (MaskedImageDataset(d["valid_manifest"], d["image_root"], **ds_kw)
               if d.get("valid_manifest") else None)

    use_cpu = not torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=t["output_dir"],
        max_steps=int(o.get("max_steps", 100_000)),
        per_device_train_batch_size=int(o.get("per_device_train_batch_size", 128)),
        per_device_eval_batch_size=int(o.get("per_device_eval_batch_size",
                                             o.get("per_device_train_batch_size", 128))),
        gradient_accumulation_steps=int(o.get("gradient_accumulation_steps", 1)),
        learning_rate=float(o.get("learning_rate", 1.0e-3)),
        weight_decay=float(o.get("weight_decay", 5.0e-2)),
        adam_beta1=float(o.get("adam_beta1", 0.9)),
        adam_beta2=float(o.get("adam_beta2", 0.95)),
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
    follow_lr = o.get("ema_momentum") is None    # mu = 1 - lr unless pinned
    trainer = CAPIHFTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collate_fn,
        first_cb=first_cb, loss_cb=loss_cb,
        proto_lr_scale=float(o.get("proto_lr_scale", 0.5)),
        callbacks=[ParamCountCallback(), first_cb, loss_cb,
                   EMACallback(model, follow_lr), SaveModelCallback()],
    )

    trainer.train()
    trainer.save_model(os.path.join(targs.output_dir, "final"))
    logger.info("training complete; final HF model -> %s/final", targs.output_dir)


if __name__ == "__main__":
    main()
