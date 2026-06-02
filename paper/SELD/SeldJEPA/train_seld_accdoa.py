"""Stage-2 Multi-ACCDOA SELD training on the frozen causal-Conformer encoder.

Trains `SeldMultiACCDOA` (frozen Stage-1 encoder + Multi-ACCDOA head, N=3 tracks,
2-D azimuth) with the **ADPIT** loss on **labeled** 2-channel audio. The encoder is
loaded frozen from the Stage-1 `.pt` (`model.encoder_ckpt`); set
`model.unfreeze_at_step` to switch to two-phase fine-tuning.

Stack (per `paper/CLAUDE.md`): `transformers.PreTrainedModel` + `Trainer` + torch
`Dataset`/`DataLoader` + accelerate/torchrun (CPU-compatible). Logs per-module
trainable params (encoder frozen/trainable, head); the first batch's first sample
AUDIO PATH + feature shape + label shape before the model is fed; step / train_loss
/ adpit_loss / valid_loss / lr every `logging_steps`; step-named checkpoints.

Run via `run_train_SeldACCDOA.sh config_seld.yaml` (the shell enforces the
"no pre-existing checkpoints" guard).
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

from features import AudioFeatureExtractor, FeatureConfig, lr_swap  # noqa: E402
from SeldMultiACCDOA import SeldMultiACCDOA  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_seld_accdoa")


# =============================================================================
# Dataset — labeled (feature, Multi-ACCDOA target) pairs
# =============================================================================

class SeldACCDOADataset(Dataset):
    """Manifest ``{"data": [{"audio_id", "label_id"}], "meta": {...}}``.

    Returns the 4-channel feature ``(4, T_f, M)`` and the per-class source target
    ``(T_label, C, 3, 2)`` (up-to-3 azimuth (x,y) vectors, active-first, zero-pad).
    Optional left/right-swap augmentation mirrors the azimuth label (phi -> -phi,
    i.e. y -> -y).
    """

    def __init__(self, manifest: str, feat_cfg: FeatureConfig, *, swap_prob: float = 0.0):
        self.root = os.path.dirname(os.path.abspath(manifest))
        with open(manifest) as f:
            blob = json.load(f)
        self.samples = blob["data"]
        self.meta = blob.get("meta", {})
        if not self.samples:
            raise ValueError(f"Empty manifest: {manifest}")
        self.fx = AudioFeatureExtractor(feat_cfg)
        self.swap_prob = swap_prob

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        rec = self.samples[idx]
        audio_path = os.path.join(self.root, rec["audio_id"])
        wav, _ = sf.read(audio_path, dtype="float32", always_2d=True)   # (T, C)
        wav = torch.from_numpy(wav).transpose(0, 1).contiguous()        # (C, T)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        with torch.no_grad():
            feat = self.fx(wav)                                          # (4, T_f, M)
        target = torch.from_numpy(
            np.load(os.path.join(self.root, rec["label_id"])).astype(np.float32))  # (Tl,C,3,2)

        if self.swap_prob > 0 and random.random() < self.swap_prob:
            feat = lr_swap(feat)
            target = target.clone()
            target[..., 1] = -target[..., 1]                            # azimuth phi -> -phi
        return {"feat": feat, "target": target, "audio_path": audio_path}


def collate_fn(batch: list[dict]) -> dict:
    return {
        "feat": torch.stack([b["feat"] for b in batch]),                # (B, 4, T_f, M)
        "target": torch.stack([b["target"] for b in batch]),            # (B, Tl, C, 3, 2)
        "audio_path": [b["audio_path"] for b in batch],
    }


# =============================================================================
# transformers wrapper
# =============================================================================

class SeldACCDOAHFConfig(PretrainedConfig):
    model_type = "seld_accdoa"

    def __init__(self, model_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.model_kwargs = model_kwargs or {}


class SeldACCDOAModel(PreTrainedModel):
    config_class = SeldACCDOAHFConfig

    def __init__(self, config: SeldACCDOAHFConfig):
        super().__init__(config)
        self.accdoa = SeldMultiACCDOA(**config.model_kwargs)
        self.post_init()

    def _init_weights(self, module):                                    # self-initialises
        pass

    def forward(self, feat=None, target=None, **_) -> dict:
        if target is not None:
            return self.accdoa.compute_loss(feat, target)
        return {"pred": self.accdoa(feat)}


# =============================================================================
# Callbacks
# =============================================================================

def _unwrap(model) -> SeldACCDOAModel:
    return model.module if hasattr(model, "module") else model


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class ParamCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        acc = _unwrap(model).accdoa
        enc_trainable = _count(acc.encoder)
        logger.info("==== trainable parameters per module ====")
        logger.info("  %-28s %15s", f"encoder ({'trainable' if enc_trainable else 'frozen'})",
                    f"{enc_trainable:,}")
        logger.info("  %-28s %15s", "accdoa head", f"{_count(acc.head):,}")
        logger.info("  %-28s %15s", "TOTAL trainable", f"{_count(acc):,}")
        logger.info("=========================================")


class FirstSampleLogCallback(TrainerCallback):
    def __init__(self):
        self.done = False

    def mark(self, audio_path, feat: torch.Tensor, target: torch.Tensor):
        if self.done:
            return
        self.done = True
        logger.info(
            "==== first batch / first sample (pre-feed) ====\n"
            "  audio path : %s\n  feat       : shape=%s dtype=%s\n  target     : shape=%s",
            audio_path[0] if audio_path else "<unknown>",
            tuple(feat.shape), feat.dtype, tuple(target.shape),
        )


class LossLogCallback(TrainerCallback):
    def __init__(self):
        self.adpit = None

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is not None and self.adpit is not None:
            logs["adpit_loss"] = round(float(self.adpit), 6)


class UnfreezeCallback(TrainerCallback):
    """Two-phase fine-tuning: unfreeze the encoder once `unfreeze_at_step` is hit."""

    def __init__(self, unfreeze_at_step: int):
        self.unfreeze_at_step = unfreeze_at_step
        self.done = False

    def on_step_begin(self, args, state, control, model=None, **kw):
        if self.done or self.unfreeze_at_step <= 0:
            return
        if state.global_step >= self.unfreeze_at_step:
            _unwrap(model).accdoa.unfreeze_encoder()
            self.done = True
            if state.is_world_process_zero:
                logger.info("unfroze encoder at step %d (two-phase fine-tuning)",
                            state.global_step)


class SaveCheckpointCallback(TrainerCallback):
    def __init__(self, model_kwargs: dict):
        self.model_kwargs = model_kwargs

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        acc = _unwrap(model).accdoa
        out = os.path.join(args.output_dir, f"seld_accdoa_step{state.global_step}.pt")
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in acc.state_dict().items()},
                    "model_kwargs": self.model_kwargs}, out)
        logger.info("saved SeldMultiACCDOA checkpoint -> %s", out)


# =============================================================================
# Trainer subclass
# =============================================================================

class SeldACCDOAHFTrainer(Trainer):
    def __init__(self, *a, first_cb=None, loss_cb=None, **kw):
        super().__init__(*a, **kw)
        self._first_cb = first_cb
        self._loss_cb = loss_cb

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_cb is not None:
            self._first_cb.mark(inputs.get("audio_path"), inputs["feat"], inputs["target"])
        outputs = model(feat=inputs["feat"], target=inputs["target"])
        if self._loss_cb is not None:
            self._loss_cb.adpit = outputs.get("adpit_loss")
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
                [{"params": decay, "weight_decay": self.args.weight_decay},
                 {"params": no_decay, "weight_decay": 0.0}],
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


def _feature_config(f: dict) -> FeatureConfig:
    return FeatureConfig(
        sample_rate=int(f.get("sample_rate", 24_000)),
        n_fft=int(f.get("n_fft", 512)),
        win_length=int(f.get("win_length", 512)),
        hop_length=int(f.get("hop_length", 240)),
        n_mels=int(f.get("n_mels", 64)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config_seld.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = _Cfg(yaml.safe_load(f))
    d, mo, t, o = cfg.sec("data"), cfg.sec("model"), cfg.sec("train"), cfg.sec("optim")

    seed = int(t.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    encoder_ckpt = mo["encoder_ckpt"]
    if not os.path.isfile(encoder_ckpt):
        raise FileNotFoundError(f"model.encoder_ckpt not found: {encoder_ckpt}")
    head = mo.get("head", {}) or {}
    enc_kwargs = SeldMultiACCDOA.read_encoder_kwargs(encoder_ckpt)
    model_kwargs = dict(
        encoder_kwargs=enc_kwargs,
        num_tracks=int(head.get("num_tracks", 3)),
        num_classes=int(head.get("num_classes", 12)),
        pool_factor=int(head.get("pool_factor", 5)),
        activity_threshold=float(head.get("activity_threshold", 0.5)),
        unify_angle_deg=float(head.get("unify_angle_deg", 30.0)),
    )
    logger.info("SeldMultiACCDOA kwargs: %s", {k: v for k, v in model_kwargs.items()
                                               if k != "encoder_kwargs"})
    model = SeldACCDOAModel(SeldACCDOAHFConfig(model_kwargs=model_kwargs))

    freeze_encoder = bool(mo.get("freeze_encoder", True))
    model.accdoa.load_encoder_weights(encoder_ckpt, freeze=freeze_encoder)

    feat_cfg = _feature_config(cfg.sec("features"))
    swap_prob = float(cfg.sec("loss").get("swap_prob", 0.0))
    train_ds = SeldACCDOADataset(d["train_manifest"], feat_cfg, swap_prob=swap_prob)
    eval_ds = (SeldACCDOADataset(d["valid_manifest"], feat_cfg)
               if d.get("valid_manifest") else None)
    logger.info("train clips: %d%s", len(train_ds),
                f" | valid clips: {len(eval_ds)}" if eval_ds else "")

    use_cpu = not torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=float(o.get("num_train_epochs", 50)),
        max_steps=int(o.get("max_steps", -1)),
        per_device_train_batch_size=int(o.get("per_device_train_batch_size", 32)),
        per_device_eval_batch_size=int(o.get("per_device_eval_batch_size",
                                             o.get("per_device_train_batch_size", 32))),
        gradient_accumulation_steps=int(o.get("gradient_accumulation_steps", 1)),
        learning_rate=float(o.get("learning_rate", 1.0e-4)),
        weight_decay=float(o.get("weight_decay", 1.0e-4)),
        adam_beta1=float(o.get("adam_beta1", 0.9)),
        adam_beta2=float(o.get("adam_beta2", 0.999)),
        warmup_steps=int(o.get("warmup_steps", 1_000)),
        logging_steps=int(t.get("logging_steps", 50)),
        save_steps=int(t.get("save_steps", 1_000)),
        save_total_limit=t.get("save_total_limit"),
        eval_strategy=("steps" if eval_ds is not None else "no"),
        eval_steps=int(t.get("eval_steps", 1_000)),
        dataloader_num_workers=int(t.get("num_workers", 4)),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
        label_names=["target"],
        use_cpu=use_cpu,
        ddp_backend=("gloo" if use_cpu else None),
        max_grad_norm=float(o.get("max_grad_norm", 1.0)),
    )

    first_cb, loss_cb = FirstSampleLogCallback(), LossLogCallback()
    callbacks = [ParamCountCallback(), first_cb, loss_cb, SaveCheckpointCallback(model_kwargs)]
    unfreeze_at = int(mo.get("unfreeze_at_step", 0) or 0)
    if unfreeze_at > 0:
        callbacks.append(UnfreezeCallback(unfreeze_at))
    trainer = SeldACCDOAHFTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collate_fn,
        first_cb=first_cb, loss_cb=loss_cb,
        callbacks=callbacks,
    )

    trainer.train()
    final_pt = os.path.join(targs.output_dir, "seld_accdoa_final.pt")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.accdoa.state_dict().items()},
                "model_kwargs": model_kwargs}, final_pt)
    logger.info("training complete; final SeldMultiACCDOA -> %s", final_pt)


if __name__ == "__main__":
    main()
