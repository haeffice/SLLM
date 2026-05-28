"""V-JEPA 2.1 self-supervised video pre-training (PyTorch 2.8).

Same JEPA pre-training loop as V-JEPA 2, but with the 2.1 recipe
(`VJEPA21Trainer`): a Dense Predictive Loss over *all* tokens plus Deep
Self-Supervision against several intermediate teacher depths. The dataset,
collator, EMA callback and `Trainer` subclass are imported verbatim from the
sibling `../VJEPA2/train_vjepa2.py` so behaviour stays identical where it
should; only the model wrapper, the per-module parameter log, the
checkpoint name (`vjepa21_student_step*.pt`) and the recipe config differ.

Stack (per `paper/CLAUDE.md`):
    * `transformers.PreTrainedModel` wrapper around `VJEPA21Trainer`
    * `transformers.Trainer` training loop (reused `VJEPA2HFTrainer`)
    * `torch.utils.data.Dataset` / `DataLoader` clip pipeline (reused)
    * `accelerate` + `torchrun` for multi-GPU (CPU-compatible)
    * logs: per-module trainable params; first batch's first sample video
      PATH before the model is fed; step / train_loss / valid_loss / lr /
      ema-decay / main_loss / deep_loss every `logging_steps`; checkpoints
      every `save_steps` with the step number in the filename; abort if a
      warm-start checkpoint fails to load.

Run via `run_train_VJEPA21.sh config.yaml`. All arguments live in `config.yaml`.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import yaml
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    TrainerCallback,
    TrainingArguments,
)

# --- sibling import: reuse the V-JEPA 2 encoder + training machinery ---------
_HERE = os.path.dirname(os.path.abspath(__file__))
_VJEPA2_DIR = os.path.join(os.path.dirname(_HERE), "VJEPA2")
for _p in (_HERE, _VJEPA2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from VJEPA2 import VARIANTS  # noqa: E402
from VJEPA21_Trainer import VJEPA21Trainer  # noqa: E402
from train_vjepa2 import (  # noqa: E402
    EMACallback,
    FirstSampleLogCallback,
    VJEPA2HFTrainer,
    VJEPA2VideoDataset,
    _count,
    _unwrap,
    collate_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_vjepa21")


# =============================================================================
# transformers wrapper
# =============================================================================

class VJEPA21HFConfig(PretrainedConfig):
    model_type = "vjepa21"

    def __init__(self, trainer_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.trainer_kwargs = trainer_kwargs or {}


class VJEPA21Model(PreTrainedModel):
    """Thin HF wrapper around `VJEPA21Trainer` (masks sampled per step)."""

    config_class = VJEPA21HFConfig

    def __init__(self, config: VJEPA21HFConfig):
        super().__init__(config)
        self.vjepa = VJEPA21Trainer(**config.trainer_kwargs)
        self.last_metrics: dict[str, float] = {}
        self.post_init()

    def _init_weights(self, module):  # VJEPA21Trainer self-initialises.
        pass

    def forward(self, video: torch.Tensor = None, **_) -> dict:
        out = self.vjepa(video)
        self.last_metrics = {
            "main_loss": float(out["main_loss"]),
            "deep_loss": float(out["deep_loss"]),
        }
        return {"loss": out["loss"]}

    @torch.no_grad()
    def ema_step(self, step: int) -> float:
        return self.vjepa.ema_step(step)


# =============================================================================
# Callbacks (2.1-specific param log, checkpoint name, recipe metrics)
# =============================================================================

class ParamCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        v = _unwrap(model).vjepa
        groups = {
            "patch_embed": v.patch_embed,
            "blocks (student encoder)": v.blocks,
            "norm": v.norm,
            "predictor": v.predictor,
            "deep_heads (deep supervision)": v.deep_heads,
            "teacher (EMA, frozen)": v.teacher,
        }
        logger.info("==== trainable parameters per module ====")
        for name, mod in groups.items():
            logger.info("  %-32s %15s", name, f"{_count(mod):,}")
        logger.info("  %-32s %15s", "TOTAL trainable", f"{_count(v):,}")
        logger.info("  deep-supervision depths : %s", list(v._ds_layers))
        logger.info("=========================================")


class MetricsCallback(TrainerCallback):
    """Surface the 2.1 sub-losses in the training logs."""

    def on_log(self, args, state, control, logs=None, model=None, **kw):
        if logs is not None and model is not None:
            m = _unwrap(model).last_metrics
            for k, val in m.items():
                logs[k] = round(val, 6)


class SaveStudentCallback(TrainerCallback):
    """Dump a student-only `.pt` (filename carries the step) on each save.

    The 2.1 student is the same `patch_embed`/`blocks`/`norm` stack as
    V-JEPA 2, so the dump loads with `VJEPA21.from_checkpoint(...)`.
    """

    _STUDENT_PREFIXES = ("patch_embed.", "blocks.", "norm.")

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        v = _unwrap(model).vjepa
        sd = {k: val.detach().cpu()
              for k, val in v.state_dict().items()
              if any(k.startswith(p) for p in self._STUDENT_PREFIXES)}
        out = os.path.join(
            args.output_dir, f"vjepa21_student_step{state.global_step}.pt")
        torch.save(sd, out)
        logger.info("saved student checkpoint (%d tensors) -> %s", len(sd), out)


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
    rc, d = c.sec("recipe"), c.sec("data")
    variant = m.get("variant", "vit_large")
    if variant not in VARIANTS:
        raise ValueError(f"model.variant={variant!r} not in {list(VARIANTS)}")
    kw: dict = dict(
        variant=variant,
        img_size=int(d.get("img_size", 256)),
        num_frames=int(d.get("num_frames", 16)),
        patch_size=int(m.get("patch_size", 16)),
        tubelet_size=int(m.get("tubelet_size", 2)),
        use_rope=bool(m.get("use_rope", True)),
    )
    for src, keys in (
        (m, ("embed_dim", "depth", "num_heads", "mlp_ratio",
             "predictor_embed_dim", "predictor_depth",
             "predictor_num_heads", "predictor_mlp_ratio")),
        (mk, ("mask_ratios", "mask_blocks_per_ratio", "min_context_ratio")),
        (em, ("ema_decay", "ema_end_decay", "ema_anneal_end_step")),
        # the V-JEPA 2.1 recipe
        (rc, ("dense_loss", "context_loss_weight",
              "deep_supervision_layers", "deep_supervision_weight")),
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

    trainer_kwargs = _build_trainer_kwargs(cfg)
    logger.info("VJEPA21Trainer kwargs: %s", trainer_kwargs)

    model = VJEPA21Model(VJEPA21HFConfig(trainer_kwargs=trainer_kwargs))

    # Optional warm-start from a student `.pt` — abort if nothing loads.
    init_from = t.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location="cpu", weights_only=True)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.vjepa.load_state_dict(sd, strict=False)
        applied = len(sd) - len(unexpected)
        if applied == 0:
            raise RuntimeError(
                f"init_from={init_from!r}: 0/{len(sd)} keys applied — aborting.")
        logger.info("init_from %s: applied %d/%d keys (missing=%d, unexpected=%d)",
                    init_from, applied, len(sd), len(missing), len(unexpected))

    img_size = trainer_kwargs["img_size"]
    num_frames = trainer_kwargs["num_frames"]
    train_ds = VJEPA2VideoDataset(
        d["train_manifest"], d["video_root"], num_frames, img_size, train=True)
    eval_ds = None
    if d.get("valid_manifest"):
        eval_ds = VJEPA2VideoDataset(
            d["valid_manifest"], d["video_root"], num_frames, img_size,
            train=False)

    use_cpu = not torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=t["output_dir"],
        max_steps=int(o.get("max_steps", 90_000)),
        per_device_train_batch_size=int(o.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(o.get("per_device_eval_batch_size",
                                             o.get("per_device_train_batch_size", 2))),
        gradient_accumulation_steps=int(o.get("gradient_accumulation_steps", 8)),
        learning_rate=float(o.get("learning_rate", 6.25e-4)),
        weight_decay=float(o.get("weight_decay", 0.04)),
        adam_beta1=float(o.get("adam_beta1", 0.9)),
        adam_beta2=float(o.get("adam_beta2", 0.999)),
        warmup_steps=int(o.get("warmup_steps", 12_000)),
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
    trainer = VJEPA2HFTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        first_sample_cb=first_cb,
        callbacks=[ParamCountCallback(), first_cb, EMACallback(),
                   MetricsCallback(), SaveStudentCallback()],
    )

    trainer.train()
    trainer.save_model(os.path.join(targs.output_dir, "final"))
    logger.info("training complete; final HF model -> %s/final", targs.output_dir)


if __name__ == "__main__":
    main()
