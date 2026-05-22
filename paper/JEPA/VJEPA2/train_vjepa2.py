"""V-JEPA 2 self-supervised video pre-training (PyTorch 2.8).

JEPA pre-training of the V-JEPA 2 video encoder: a student ViT encodes the
visible (context) tokens of a clip, a narrow predictor reconstructs the
EMA-teacher features at the masked (target) tokens, optimised with a masked
smooth-L1 latent loss.

Stack (per `paper/CLAUDE.md`):
    * `transformers.PreTrainedModel` wrapper around `VJEPA2Trainer`
    * `transformers.Trainer` training loop
    * `torch.utils.data.Dataset` / `DataLoader` clip pipeline
    * `accelerate` + `torchrun` for multi-GPU (CPU-compatible)
    * logs: per-module trainable params; first batch's first sample video
      PATH before the model is fed; step / train_loss / valid_loss / lr /
      ema-decay every `logging_steps`; checkpoints every `save_steps` with
      the step number in the filename; abort if a resume/init checkpoint
      fails to load.

Run via `run_train_VJEPA2.sh config.yaml` (the shell script enforces the
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

from VJEPA2 import VARIANTS  # noqa: E402
from VJEPA2_Trainer import VJEPA2Trainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_vjepa2")


# =============================================================================
# Dataset — manifest-driven clip loader (.npy tensor or video via decord)
# =============================================================================

class VJEPA2VideoDataset(Dataset):
    """Manifest JSON: ``{"data": [{"video_id": str}, ...]}``.

    ``video_path = video_root/video_id``. Two backends:
      * ``*.npy``  : array (T, H, W, 3) uint8 or (3, T, H, W) float — used
                     by `make_synthetic_manifest.py` so the smoke test has
                     no ffmpeg/decord dependency.
      * otherwise  : decoded with `decord.VideoReader`.
    Output clip: float tensor (3, num_frames, img_size, img_size) in [0, 1].
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
        if self.train:
            start = random.randint(0, total - n)
        else:
            start = (total - n) // 2
        return np.arange(start, start + n)

    def _resize(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (T, 3, H, W) float in [0, 1] -> (3, T, S, S)
        clip = torch.nn.functional.interpolate(
            clip, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        return clip.permute(1, 0, 2, 3).contiguous()

    def _load_npy(self, path: str) -> torch.Tensor:
        arr = np.load(path)
        if arr.ndim != 4:
            raise ValueError(f"{path}: expected 4D array, got {arr.shape}")
        if arr.shape[0] == 3 and arr.shape[1] != 3:        # (3, T, H, W)
            t = torch.from_numpy(arr).float()
            if t.max() > 1.5:
                t = t / 255.0
            clip = t.permute(1, 0, 2, 3)                    # (T, 3, H, W)
        else:                                              # (T, H, W, 3)
            t = torch.from_numpy(arr).float()
            if t.max() > 1.5:
                t = t / 255.0
            clip = t.permute(0, 3, 1, 2)                    # (T, 3, H, W)
        idx = self._sample_frame_idx(clip.shape[0])
        return self._resize(clip[idx])

    def _load_video(self, path: str) -> torch.Tensor:
        import decord  # imported lazily; only needed for real videos
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(path)
        idx = self._sample_frame_idx(len(vr))
        frames = vr.get_batch(list(idx)).float() / 255.0   # (T, H, W, 3)
        clip = frames.permute(0, 3, 1, 2)                  # (T, 3, H, W)
        return self._resize(clip)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        path = os.path.join(self.video_root, s["video_id"])
        clip = (self._load_npy(path) if path.endswith(".npy")
                else self._load_video(path))
        return {"video": clip, "video_path": path}


def collate_fn(batch: list[dict]) -> dict:
    return {
        "video": torch.stack([b["video"] for b in batch]),
        "video_path": [b["video_path"] for b in batch],
    }


# =============================================================================
# transformers wrapper
# =============================================================================

class VJEPA2HFConfig(PretrainedConfig):
    model_type = "vjepa2"

    def __init__(self, trainer_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.trainer_kwargs = trainer_kwargs or {}


class VJEPA2Model(PreTrainedModel):
    """Thin HF wrapper around `VJEPA2Trainer` (masks sampled per step)."""

    config_class = VJEPA2HFConfig

    def __init__(self, config: VJEPA2HFConfig):
        super().__init__(config)
        self.vjepa = VJEPA2Trainer(**config.trainer_kwargs)
        self.post_init()

    def _init_weights(self, module):  # VJEPA2Trainer self-initialises.
        pass

    def forward(self, video: torch.Tensor = None, **_) -> dict:
        out = self.vjepa(video)
        return {"loss": out["loss"]}

    @torch.no_grad()
    def ema_step(self, step: int) -> float:
        return self.vjepa.ema_step(step)


# =============================================================================
# Callbacks
# =============================================================================

def _unwrap(model) -> VJEPA2Model:
    return model.module if hasattr(model, "module") else model


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


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
            "teacher (EMA, frozen)": v.teacher,
        }
        logger.info("==== trainable parameters per module ====")
        for name, mod in groups.items():
            logger.info("  %-28s %15s", name, f"{_count(mod):,}")
        logger.info("  %-28s %15s", "TOTAL trainable", f"{_count(v):,}")
        logger.info("=========================================")


class FirstSampleLogCallback(TrainerCallback):
    def __init__(self):
        self.done = False

    def mark(self, video_path: list[str], video: torch.Tensor):
        if self.done:
            return
        self.done = True
        logger.info(
            "==== first batch / first sample (pre-feed) ====\n"
            "  video path : %s\n  video      : shape=%s dtype=%s",
            video_path[0] if video_path else "<unknown>",
            tuple(video.shape), video.dtype,
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
    carries the step number (loadable via `VJEPA2.from_checkpoint`)."""

    _STUDENT_PREFIXES = ("patch_embed.", "blocks.", "norm.")

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        v = _unwrap(model).vjepa
        sd = {k: val.detach().cpu()
              for k, val in v.state_dict().items()
              if any(k.startswith(p) for p in self._STUDENT_PREFIXES)}
        out = os.path.join(
            args.output_dir, f"vjepa2_student_step{state.global_step}.pt")
        torch.save(sd, out)
        logger.info("saved student checkpoint (%d tensors) -> %s", len(sd), out)


# =============================================================================
# Trainer subclass — paper optimiser/schedule + first-sample logging
# =============================================================================

class VJEPA2HFTrainer(Trainer):
    def __init__(self, *a, first_sample_cb: FirstSampleLogCallback = None, **kw):
        super().__init__(*a, **kw)
        self._first_sample_cb = first_sample_cb

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_sample_cb is not None:
            self._first_sample_cb.mark(inputs.get("video_path"),
                                       inputs["video"])
        outputs = model(video=inputs["video"])
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
        # `embed_dim`/`depth`/`num_heads`/`mlp_ratio` override the variant
        # defaults (used by the CPU smoke test to shrink the encoder).
        (m, ("embed_dim", "depth", "num_heads", "mlp_ratio",
             "predictor_embed_dim", "predictor_depth",
             "predictor_num_heads", "predictor_mlp_ratio")),
        (mk, ("mask_ratios", "mask_blocks_per_ratio", "min_context_ratio")),
        (em, ("ema_decay", "ema_end_decay", "ema_anneal_end_step")),
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
    logger.info("VJEPA2Trainer kwargs: %s", trainer_kwargs)

    model = VJEPA2Model(VJEPA2HFConfig(trainer_kwargs=trainer_kwargs))

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
                   SaveStudentCallback()],
    )

    trainer.train()
    trainer.save_model(os.path.join(targs.output_dir, "final"))
    logger.info("training complete; final HF model -> %s/final", targs.output_dir)


if __name__ == "__main__":
    main()
