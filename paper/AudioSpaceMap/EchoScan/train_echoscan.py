"""EchoScan supervised training (PyTorch 2.8).

Trains the `EchoScan` network (acoustic echoes → floorplan + height maps) on
the simulated dataset produced by `make_echoscan_dataset.py`.

Stack (per `paper/CLAUDE.md`):
    * `transformers.PreTrainedModel` wrapper around `EchoScan`
    * `transformers.Trainer` training loop, AdamW + cosine schedule
    * `torch.utils.data.Dataset` / `DataLoader` over the `.npz` manifest
    * `accelerate` + `torchrun` for multi-GPU (CPU-compatible)
    * logs: per-module trainable params; first batch's first sample RIR PATH
      before the model is fed; step / train_loss / valid_loss / lr every
      `logging_steps`; checkpoints every `save_steps` with the step number in
      the filename; abort if an `init_from` warm-start matches nothing.

Run via `run_train_EchoScan.sh config.yaml` (the shell script enforces the
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

from EchoScan import (  # noqa: E402
    EchoScan,
    HEIGHT_SIZE,
    FLOORPLAN_SIZE,
    N_MICS,
    RIR_LENGTH,
    echoscan_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_echoscan")


# =============================================================================
# Dataset — reads the .npz samples written by make_echoscan_dataset.py
# =============================================================================

class EchoScanDataset(Dataset):
    """Manifest-driven RIR → (floorplan, height) dataset.

    Each manifest entry points to a `.npz` holding ``rir`` (M, N) float32,
    ``floor_packed`` (np.packbits of the b×b mask) and ``height`` (h,) uint8.
    The packed floorplan is unpacked here (cheap) so the model sees a dense
    (1, b, b) float target.
    """

    def __init__(self, manifest: str, floorplan_size: int, height_size: int):
        self.root = os.path.dirname(os.path.abspath(manifest))
        with open(manifest, "r") as f:
            blob = json.load(f)
        self.samples = blob["data"]
        self.meta = blob.get("meta", {})
        self.b = floorplan_size
        self.h = height_size
        if not self.samples:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        path = os.path.join(self.root, s["rir_id"])
        with np.load(path) as z:
            rir = z["rir"].astype(np.float32)                       # (M, N)
            b = int(z["b"]) if "b" in z else self.b
            floor = np.unpackbits(z["floor_packed"])[: b * b].reshape(b, b)
            height = z["height"].astype(np.float32)                 # (h,)
        return {
            "rir": torch.from_numpy(rir),
            "floorplan": torch.from_numpy(floor.astype(np.float32))[None],  # (1, b, b)
            "height": torch.from_numpy(height),
            "rir_path": path,
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "rir": torch.stack([b["rir"] for b in batch]),              # (B, M, N)
        "floorplan": torch.stack([b["floorplan"] for b in batch]),  # (B, 1, b, b)
        "height": torch.stack([b["height"] for b in batch]),        # (B, h)
        "rir_path": [b["rir_path"] for b in batch],
    }


# =============================================================================
# transformers wrapper
# =============================================================================

class EchoScanHFConfig(PretrainedConfig):
    model_type = "echoscan"

    def __init__(self, model_kwargs: Optional[dict] = None,
                 loss_alpha: float = 0.3, loss_beta: float = 1.0, **kw):
        super().__init__(**kw)
        self.model_kwargs = model_kwargs or {}
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta


class EchoScanModel(PreTrainedModel):
    """Thin HF wrapper: owns an `EchoScan`, computes the combined loss when
    labels are present, returns `{"loss": ...}` for `Trainer`."""

    config_class = EchoScanHFConfig

    def __init__(self, config: EchoScanHFConfig):
        super().__init__(config)
        self.echoscan = EchoScan(**config.model_kwargs)
        self.loss_alpha = config.loss_alpha
        self.loss_beta = config.loss_beta
        self.post_init()

    def _init_weights(self, module):                # EchoScan self-initialises.
        pass

    def forward(self, rir=None, floorplan=None, height=None, **_) -> dict:
        out = self.echoscan(rir)
        result = {"floorplan_logits": out["floorplan_logits"],
                  "height_logits": out["height_logits"]}
        if floorplan is not None and height is not None:
            loss, parts = echoscan_loss(out, floorplan, height,
                                        alpha=self.loss_alpha, beta=self.loss_beta)
            result["loss"] = loss
            result.update(parts)
        return result


# =============================================================================
# Callbacks
# =============================================================================

def _unwrap(model) -> EchoScanModel:
    return model.module if hasattr(model, "module") else model


def _count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class ParamCountCallback(TrainerCallback):
    """Log per-module trainable parameter counts once at train start."""

    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        es = _unwrap(model).echoscan
        groups = {
            "encoder (1-D ResNet)": es.encoder,
            "multi_aggregation": es.ma,
            "floorplan_decoder": es.floorplan_decoder,
            "height_decoder": es.height_decoder,
        }
        logger.info("==== trainable parameters per module ====")
        for name, mod in groups.items():
            logger.info("  %-28s %15s", name, f"{_count_trainable(mod):,}")
        logger.info("  %-28s %15s", "TOTAL trainable", f"{_count_trainable(es):,}")
        logger.info("=========================================")


class FirstSampleLogCallback(TrainerCallback):
    """Log the first batch's first sample RIR PATH before the model is fed."""

    def __init__(self):
        self.done = False

    def mark(self, rir_path: list[str], rir: torch.Tensor,
             floorplan: torch.Tensor, height: torch.Tensor):
        if self.done:
            return
        self.done = True
        logger.info(
            "==== first batch / first sample (pre-feed) ====\n"
            "  rir path  : %s\n  rir       : shape=%s dtype=%s\n"
            "  floorplan : shape=%s interior_px=%d\n  height    : shape=%s interior_px=%d",
            rir_path[0] if rir_path else "<unknown>",
            tuple(rir.shape), rir.dtype,
            tuple(floorplan.shape), int(floorplan[0].sum().item()),
            tuple(height.shape), int(height[0].sum().item()),
        )


class SaveCheckpointCallback(TrainerCallback):
    """On every HF checkpoint, also dump a portable `.pt` whose filename
    carries the step number (loadable via `EchoScan.from_checkpoint`)."""

    def __init__(self, model_kwargs: dict):
        self.model_kwargs = model_kwargs

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        es = _unwrap(model).echoscan
        out = os.path.join(args.output_dir, f"echoscan_step{state.global_step}.pt")
        torch.save({"state_dict": {k: v.detach().cpu()
                                   for k, v in es.state_dict().items()},
                    "model_kwargs": self.model_kwargs}, out)
        logger.info("saved EchoScan checkpoint -> %s", out)


# =============================================================================
# Trainer subclass — paper optimiser/schedule + first-sample logging
# =============================================================================

class EchoScanHFTrainer(Trainer):
    def __init__(self, *a, first_sample_cb: FirstSampleLogCallback = None, **kw):
        super().__init__(*a, **kw)
        self._first_sample_cb = first_sample_cb

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_sample_cb is not None:
            self._first_sample_cb.mark(inputs.get("rir_path"), inputs["rir"],
                                       inputs["floorplan"], inputs["height"])
        outputs = model(rir=inputs["rir"], floorplan=inputs["floorplan"],
                        height=inputs["height"])           # rir_path NOT fed
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
    m, d = c.sec("model"), c.sec("data")
    return dict(
        n_mics=int(d.get("n_mics", N_MICS)),
        rir_length=int(d.get("rir_length", RIR_LENGTH)),
        floorplan_size=int(d.get("floorplan_size", FLOORPLAN_SIZE)),
        height_size=int(d.get("height_size", HEIGHT_SIZE)),
        ma_proj_dim=int(m.get("ma_proj_dim", 256)),
        decoder_init_size=int(m.get("decoder_init_size", 16)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = _Cfg(yaml.safe_load(f))
    d, t, o, lo = cfg.sec("data"), cfg.sec("train"), cfg.sec("optim"), cfg.sec("loss")

    seed = int(t.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_kwargs = _build_model_kwargs(cfg)
    logger.info("EchoScan model kwargs: %s", model_kwargs)
    hf_cfg = EchoScanHFConfig(
        model_kwargs=model_kwargs,
        loss_alpha=float(lo.get("alpha", 0.3)),
        loss_beta=float(lo.get("beta", 1.0)),
    )
    model = EchoScanModel(hf_cfg)

    # Optional warm-start — abort if nothing loads.
    init_from = t.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location="cpu", weights_only=False)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.echoscan.load_state_dict(sd, strict=False)
        applied = len(sd) - len(unexpected)
        if applied == 0:
            raise RuntimeError(f"init_from={init_from!r}: 0/{len(sd)} keys applied.")
        logger.info("init_from %s: applied %d/%d keys (missing=%d, unexpected=%d)",
                    init_from, applied, len(sd), len(missing), len(unexpected))

    train_ds = EchoScanDataset(d["train_manifest"], model_kwargs["floorplan_size"],
                               model_kwargs["height_size"])
    eval_ds = None
    if d.get("valid_manifest"):
        eval_ds = EchoScanDataset(d["valid_manifest"], model_kwargs["floorplan_size"],
                                  model_kwargs["height_size"])

    use_cpu = not torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=float(o.get("num_train_epochs", 3)),
        max_steps=int(o.get("max_steps", -1)),
        per_device_train_batch_size=int(o.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(o.get("per_device_eval_batch_size",
                                             o.get("per_device_train_batch_size", 16))),
        gradient_accumulation_steps=int(o.get("gradient_accumulation_steps", 1)),
        learning_rate=float(o.get("learning_rate", 1.0e-3)),
        weight_decay=float(o.get("weight_decay", 1.0e-4)),
        adam_beta1=float(o.get("adam_beta1", 0.9)),
        adam_beta2=float(o.get("adam_beta2", 0.999)),
        warmup_steps=int(o.get("warmup_steps", 1000)),
        logging_steps=int(t.get("logging_steps", 50)),
        save_steps=int(t.get("save_steps", 1000)),
        save_total_limit=t.get("save_total_limit"),
        eval_strategy=("steps" if eval_ds is not None else "no"),
        eval_steps=int(t.get("eval_steps", 1000)),
        dataloader_num_workers=int(t.get("num_workers", 4)),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,        # keep rir/floorplan/height/rir_path
        label_names=["floorplan", "height"],
        use_cpu=use_cpu,
        ddp_backend=("gloo" if use_cpu else None),
        max_grad_norm=float(o.get("max_grad_norm", 1.0)),
    )

    first_cb = FirstSampleLogCallback()
    trainer = EchoScanHFTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        first_sample_cb=first_cb,
        callbacks=[ParamCountCallback(), first_cb,
                   SaveCheckpointCallback(model_kwargs)],
    )

    trainer.train()
    final_pt = os.path.join(targs.output_dir, "echoscan_final.pt")
    torch.save({"state_dict": {k: v.detach().cpu()
                               for k, v in model.echoscan.state_dict().items()},
                "model_kwargs": model_kwargs}, final_pt)
    logger.info("training complete; final EchoScan -> %s", final_pt)


if __name__ == "__main__":
    main()
