"""LeWorldModel (LeWM) self-supervised pre-training (PyTorch 2.8).

End-to-end JEPA world-model pre-training from pixels: a ViT encodes each
frame, an AR predictor forecasts the next-frame latent from past latents +
actions, and SIGReg keeps the latents isotropic-Gaussian (no EMA teacher /
stop-gradient). Objective: `L_pred + lambda * L_reg` (arXiv:2603.19312).

Stack (per `paper/CLAUDE.md`):
    * `transformers.PreTrainedModel` wrapper around `LeWorldModel`
    * `transformers.Trainer` training loop
    * `torch.utils.data.Dataset` / `DataLoader` episode pipeline
    * `accelerate` + `torchrun` for multi-GPU (CPU-compatible)
    * logs: per-module trainable params; first batch's first sample
      episode PATH before the model is fed; step / train_loss / pred_loss /
      reg_loss / valid_loss / lr every `logging_steps`; checkpoints every
      `save_steps` with the step number in the filename; abort if an init
      checkpoint fails to load.

Run via `run_train_LeWorldModel.sh config.yaml` (the shell enforces the
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

from LeWorldModel import LeWMConfig, LeWorldModel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_lewm")


# =============================================================================
# Dataset — manifest-driven episode loader (obs frames + actions)
# =============================================================================

class EpisodeDataset(Dataset):
    """Manifest JSON: ``{"data": [{"episode_id": str}, ...]}``.

    ``episode_path = data_root/episode_id`` (``.npz`` with arrays
    ``obs`` (T, H, W, 3) uint8/float and ``actions`` (T, action_dim)).
    A contiguous window of ``seq_len`` steps is sampled (random start in
    train, head in eval). Output:
        obs     : (seq_len, 3, H, W) float in [0, 1]
        actions : (seq_len, action_dim) float
    """

    def __init__(self, manifest: str, data_root: str, seq_len: int,
                 img_size: int, train: bool = True):
        with open(manifest) as f:
            self.samples = json.load(f)["data"]
        if not self.samples:
            raise ValueError(f"Empty manifest: {manifest}")
        self.data_root = data_root
        self.seq_len = seq_len
        self.img_size = img_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def _resize(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (T, 3, H, W) -> (T, 3, S, S)
        if obs.shape[-1] != self.img_size or obs.shape[-2] != self.img_size:
            obs = torch.nn.functional.interpolate(
                obs, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False)
        return obs

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        path = os.path.join(self.data_root, s["episode_id"])
        blob = np.load(path)
        obs = np.asarray(blob["obs"], dtype=np.float32)
        actions = np.asarray(blob["actions"], dtype=np.float32)
        T = obs.shape[0]
        n = self.seq_len
        if T >= n:
            start = random.randint(0, T - n) if self.train else 0
        else:
            start = 0
        sl = slice(start, start + n)
        obs, actions = obs[sl], actions[sl]
        if obs.shape[0] < n:                                  # pad short episode
            obs = np.concatenate(
                [obs, np.repeat(obs[-1:], n - obs.shape[0], 0)], 0)
            actions = np.concatenate(
                [actions, np.repeat(actions[-1:], n - actions.shape[0], 0)], 0)
        if obs.max() > 1.5:
            obs = obs / 255.0
        obs_t = torch.from_numpy(obs).permute(0, 3, 1, 2).contiguous()  # (T,3,H,W)
        obs_t = self._resize(obs_t)
        return {"obs": obs_t, "actions": torch.from_numpy(actions).float(),
                "episode_path": path}


def collate_fn(batch: list[dict]) -> dict:
    return {
        "obs": torch.stack([b["obs"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
        "episode_path": [b["episode_path"] for b in batch],
    }


# =============================================================================
# transformers wrapper
# =============================================================================

class LeWMHFConfig(PretrainedConfig):
    model_type = "leworldmodel"

    def __init__(self, model_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.model_kwargs = model_kwargs or {}


class LeWMModel(PreTrainedModel):
    config_class = LeWMHFConfig

    def __init__(self, config: LeWMHFConfig):
        super().__init__(config)
        self.lewm = LeWorldModel(LeWMConfig(**config.model_kwargs))
        self.post_init()

    def _init_weights(self, module):  # LeWorldModel self-initialises.
        pass

    def forward(self, obs: torch.Tensor = None, actions: torch.Tensor = None,
                **_) -> dict:
        out = self.lewm.compute_loss(obs, actions)
        return out


# =============================================================================
# Callbacks
# =============================================================================

def _unwrap(model) -> LeWMModel:
    return model.module if hasattr(model, "module") else model


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class ParamCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        m = _unwrap(model).lewm
        groups = {
            "encoder (ViT)": m.encoder,
            "action_embedder": m.action_embedder,
            "predictor (AR)": m.predictor,
        }
        logger.info("==== trainable parameters per module ====")
        for name, mod in groups.items():
            logger.info("  %-22s %15s", name, f"{_count(mod):,}")
        logger.info("  %-22s %15s", "TOTAL trainable", f"{_count(m):,}")
        logger.info("=========================================")


class FirstSampleLogCallback(TrainerCallback):
    def __init__(self):
        self.done = False

    def mark(self, episode_path: list[str], obs: torch.Tensor,
             actions: torch.Tensor):
        if self.done:
            return
        self.done = True
        logger.info(
            "==== first batch / first sample (pre-feed) ====\n"
            "  episode path : %s\n  obs          : shape=%s dtype=%s\n"
            "  actions      : shape=%s dtype=%s",
            episode_path[0] if episode_path else "<unknown>",
            tuple(obs.shape), obs.dtype, tuple(actions.shape), actions.dtype,
        )


class LossLogCallback(TrainerCallback):
    def __init__(self):
        self.pred = None
        self.reg = None

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is not None:
            if self.pred is not None:
                logs["pred_loss"] = round(float(self.pred), 6)
            if self.reg is not None:
                logs["reg_loss"] = round(float(self.reg), 6)


class SaveModelCallback(TrainerCallback):
    """On every HF checkpoint also dump a model-only `.pt` whose filename
    carries the step number (loadable via `LeWorldModel.from_checkpoint`)."""

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        m = _unwrap(model).lewm
        sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        out = os.path.join(args.output_dir, f"lewm_step{state.global_step}.pt")
        torch.save(sd, out)
        logger.info("saved model checkpoint (%d tensors) -> %s", len(sd), out)


# =============================================================================
# Trainer subclass
# =============================================================================

class LeWMHFTrainer(Trainer):
    def __init__(self, *a, first_cb=None, loss_cb=None, **kw):
        super().__init__(*a, **kw)
        self._first_cb = first_cb
        self._loss_cb = loss_cb

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_cb is not None:
            self._first_cb.mark(inputs.get("episode_path"),
                                inputs["obs"], inputs["actions"])
        outputs = model(obs=inputs["obs"], actions=inputs["actions"])
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
    m, d, ob = c.sec("model"), c.sec("data"), c.sec("objective")
    kw: dict = dict(
        img_size=int(d.get("img_size", 64)),
        action_dim=int(d.get("action_dim", 2)),
        max_seq_len=int(d.get("seq_len", 16)),
    )
    for k in ("patch_size", "embed_dim", "depth", "num_heads", "mlp_ratio",
              "latent_dim", "pred_depth", "pred_num_heads"):
        if k in m:
            kw[k] = m[k]
    for k in ("reg_weight", "num_proj"):
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
    logger.info("LeWorldModel kwargs: %s", model_kwargs)
    model = LeWMModel(LeWMHFConfig(model_kwargs=model_kwargs))

    init_from = t.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location="cpu", weights_only=True)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.lewm.load_state_dict(sd, strict=False)
        if len(sd) - len(unexpected) == 0:
            raise RuntimeError(
                f"init_from={init_from!r}: 0/{len(sd)} keys applied — aborting.")
        logger.info("init_from %s: applied %d/%d keys",
                    init_from, len(sd) - len(unexpected), len(sd))

    seq_len = int(d.get("seq_len", 16))
    img_size = model_kwargs["img_size"]
    train_ds = EpisodeDataset(d["train_manifest"], d["data_root"], seq_len,
                              img_size, train=True)
    eval_ds = (EpisodeDataset(d["valid_manifest"], d["data_root"], seq_len,
                              img_size, train=False)
               if d.get("valid_manifest") else None)

    use_cpu = not torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=t["output_dir"],
        max_steps=int(o.get("max_steps", 50_000)),
        per_device_train_batch_size=int(o.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(o.get("per_device_eval_batch_size",
                                             o.get("per_device_train_batch_size", 16))),
        gradient_accumulation_steps=int(o.get("gradient_accumulation_steps", 1)),
        learning_rate=float(o.get("learning_rate", 3.0e-4)),
        weight_decay=float(o.get("weight_decay", 0.05)),
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
    trainer = LeWMHFTrainer(
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
