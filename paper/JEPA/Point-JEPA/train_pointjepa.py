"""Point-JEPA self-supervised pre-training (PyTorch 2.8).

JEPA pre-training of the Point-JEPA encoder on point clouds: a student ViT
encodes the context tokens, a narrow predictor reconstructs the EMA-teacher
features at the target tokens (selected as contiguous spans of the
proximity-ordered sequence), optimised with a smooth-L1 latent loss.

Stack (per `paper/CLAUDE.md`):
    * `transformers.PreTrainedModel` wrapper around `PointJEPATrainer`
    * `transformers.Trainer` training loop
    * `torch.utils.data.Dataset` / `DataLoader` point-cloud pipeline
    * `accelerate` + `torchrun` for multi-GPU (CPU-compatible)
    * logs: per-module trainable params; first batch's first sample
      point-cloud PATH before the model is fed; step / train_loss /
      valid_loss / lr / ema-decay every `logging_steps`; checkpoints every
      `save_steps` with the step number in the filename; abort if an init
      checkpoint fails to load.

Run via `run_train_PointJEPA.sh config.yaml` (the shell enforces the
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

from PointJEPA_Trainer import PointJEPATrainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_pointjepa")


# =============================================================================
# Dataset — manifest-driven point-cloud loader
# =============================================================================

def _normalize_unit_sphere(pc: np.ndarray) -> np.ndarray:
    pc = pc - pc.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pc, axis=1))
    return pc / (scale + 1e-6)


class PointCloudDataset(Dataset):
    """Manifest JSON: ``{"data": [{"points_id": str}, ...]}``.

    ``points_path = points_root/points_id`` (``.npy``/``.npz`` array
    (P, 3) or (P, >=3) — only XYZ used). Normalised to the unit sphere,
    then `num_points` are sampled (random in train, head in eval).
    """

    def __init__(self, manifest: str, points_root: str, num_points: int,
                 train: bool = True):
        with open(manifest) as f:
            self.samples = json.load(f)["data"]
        if not self.samples:
            raise ValueError(f"Empty manifest: {manifest}")
        self.points_root = points_root
        self.num_points = num_points
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str) -> np.ndarray:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[arr.files[0]]
        arr = np.asarray(arr, dtype=np.float32)[:, :3]
        n = self.num_points
        if arr.shape[0] >= n:
            if self.train:
                sel = np.random.choice(arr.shape[0], n, replace=False)
            else:
                sel = np.arange(n)
            arr = arr[sel]
        else:
            pad = np.random.choice(arr.shape[0], n - arr.shape[0], replace=True)
            arr = np.concatenate([arr, arr[pad]], axis=0)
        return _normalize_unit_sphere(arr)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        path = os.path.join(self.points_root, s["points_id"])
        return {"points": torch.from_numpy(self._load(path)).float(),
                "points_path": path}


def collate_fn(batch: list[dict]) -> dict:
    return {
        "points": torch.stack([b["points"] for b in batch]),
        "points_path": [b["points_path"] for b in batch],
    }


# =============================================================================
# transformers wrapper
# =============================================================================

class PointJEPAHFConfig(PretrainedConfig):
    model_type = "point_jepa"

    def __init__(self, trainer_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.trainer_kwargs = trainer_kwargs or {}


class PointJEPAModel(PreTrainedModel):
    config_class = PointJEPAHFConfig

    def __init__(self, config: PointJEPAHFConfig):
        super().__init__(config)
        self.pj = PointJEPATrainer(**config.trainer_kwargs)
        self.post_init()

    def _init_weights(self, module):  # PointJEPATrainer self-initialises.
        pass

    def forward(self, points: torch.Tensor = None, **_) -> dict:
        out = self.pj(points)
        return {"loss": out["loss"]}

    @torch.no_grad()
    def ema_step(self, step: int) -> float:
        return self.pj.ema_step(step)


# =============================================================================
# Callbacks
# =============================================================================

def _unwrap(model) -> PointJEPAModel:
    return model.module if hasattr(model, "module") else model


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class ParamCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        p = _unwrap(model).pj
        groups = {
            "tokenizer": p.tokenizer,
            "pos_embed": p.pos_embed,
            "encoder (student)": p.encoder,
            "predictor": p.predictor,
            "teacher_encoder (EMA, frozen)": p.teacher_encoder,
        }
        logger.info("==== trainable parameters per module ====")
        for name, mod in groups.items():
            logger.info("  %-30s %15s", name, f"{_count(mod):,}")
        logger.info("  %-30s %15s", "TOTAL trainable", f"{_count(p):,}")
        logger.info("=========================================")


class FirstSampleLogCallback(TrainerCallback):
    def __init__(self):
        self.done = False

    def mark(self, points_path: list[str], points: torch.Tensor):
        if self.done:
            return
        self.done = True
        logger.info(
            "==== first batch / first sample (pre-feed) ====\n"
            "  points path : %s\n  points      : shape=%s dtype=%s",
            points_path[0] if points_path else "<unknown>",
            tuple(points.shape), points.dtype,
        )


class EMACallback(TrainerCallback):
    def __init__(self):
        self.last_decay = None

    def on_step_end(self, args, state, control, model=None, **kw):
        self.last_decay = _unwrap(model).ema_step(state.global_step)

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is not None and self.last_decay is not None:
            logs["ema_decay"] = round(float(self.last_decay), 8)


class SaveStudentCallback(TrainerCallback):
    """On every HF checkpoint also dump a student-only `.pt` whose filename
    carries the step number (loadable via `PointJEPA.from_checkpoint`)."""

    _STUDENT_PREFIXES = ("tokenizer.", "pos_embed.", "encoder.")

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        p = _unwrap(model).pj
        sd = {k: v.detach().cpu()
              for k, v in p.state_dict().items()
              if any(k.startswith(pre) for pre in self._STUDENT_PREFIXES)}
        out = os.path.join(
            args.output_dir, f"pointjepa_student_step{state.global_step}.pt")
        torch.save(sd, out)
        logger.info("saved student checkpoint (%d tensors) -> %s", len(sd), out)


# =============================================================================
# Trainer subclass
# =============================================================================

class PointJEPAHFTrainer(Trainer):
    def __init__(self, *a, first_sample_cb: FirstSampleLogCallback = None, **kw):
        super().__init__(*a, **kw)
        self._first_sample_cb = first_sample_cb

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_sample_cb is not None:
            self._first_sample_cb.mark(inputs.get("points_path"),
                                       inputs["points"])
        outputs = model(points=inputs["points"])
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


def _build_trainer_kwargs(c: _Cfg) -> dict:
    m, mk, em = c.sec("model"), c.sec("masking"), c.sec("ema")
    d = c.sec("data")
    kw: dict = dict(num_points=int(d.get("num_points", 1024)))
    for src, keys in (
        (m, ("num_groups", "group_size", "embed_dim", "depth", "num_heads",
             "mlp_ratio", "predictor_dim", "predictor_depth")),
        (mk, ("num_target_blocks", "target_ratio_min", "target_ratio_max",
              "context_ratio_min", "context_ratio_max", "smooth_l1_beta")),
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

    with open(args.config) as f:
        cfg = _Cfg(yaml.safe_load(f))
    d, t, o = cfg.sec("data"), cfg.sec("train"), cfg.sec("optim")

    seed = int(t.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainer_kwargs = _build_trainer_kwargs(cfg)
    logger.info("PointJEPATrainer kwargs: %s", trainer_kwargs)
    model = PointJEPAModel(PointJEPAHFConfig(trainer_kwargs=trainer_kwargs))

    init_from = t.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location="cpu", weights_only=True)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.pj.load_state_dict(sd, strict=False)
        if len(sd) - len(unexpected) == 0:
            raise RuntimeError(
                f"init_from={init_from!r}: 0/{len(sd)} keys applied — aborting.")
        logger.info("init_from %s: applied %d/%d keys",
                    init_from, len(sd) - len(unexpected), len(sd))

    num_points = trainer_kwargs["num_points"]
    train_ds = PointCloudDataset(d["train_manifest"], d["points_root"],
                                 num_points, train=True)
    eval_ds = (PointCloudDataset(d["valid_manifest"], d["points_root"],
                                 num_points, train=False)
               if d.get("valid_manifest") else None)

    use_cpu = not torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=t["output_dir"],
        max_steps=int(o.get("max_steps", 30_000)),
        per_device_train_batch_size=int(o.get("per_device_train_batch_size", 32)),
        per_device_eval_batch_size=int(o.get("per_device_eval_batch_size",
                                             o.get("per_device_train_batch_size", 32))),
        gradient_accumulation_steps=int(o.get("gradient_accumulation_steps", 1)),
        learning_rate=float(o.get("learning_rate", 1.0e-3)),
        weight_decay=float(o.get("weight_decay", 0.05)),
        adam_beta1=float(o.get("adam_beta1", 0.9)),
        adam_beta2=float(o.get("adam_beta2", 0.999)),
        warmup_steps=int(o.get("warmup_steps", 3_000)),
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

    first_cb = FirstSampleLogCallback()
    trainer = PointJEPAHFTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
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
