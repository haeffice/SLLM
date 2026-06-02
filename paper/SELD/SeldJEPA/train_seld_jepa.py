"""Stage-1 LeJEPA self-supervised pre-training of the causal Conformer encoder.

Trains `SeldConformer` (causal Conformer + JEPA predictor + SIGReg) on **unlabeled**
2-channel audio: each clip yields DINO-style multi-crop temporal views of the
4-channel binaural feature stack, every view's embedding predicts the global
views' embeddings, and SIGReg drives the pooled embeddings toward an isotropic
Gaussian. No EMA teacher / stop-gradient / scheduler — one knob `sigreg_coeff`.

Stack (per `paper/CLAUDE.md`): `transformers.PreTrainedModel` wrapper +
`transformers.Trainer` + torch `Dataset`/`DataLoader` + accelerate/torchrun
(CPU-compatible). Logs per-module trainable params; the first batch's first
sample AUDIO PATH (+ feature shape) before the model is fed; step / train_loss /
pred_loss / reg_loss / lr every `logging_steps`; step-named checkpoints every
`save_steps`; aborts if an `init_from` warm-start matches nothing.

Run via `run_train_SeldJEPA.sh config_pretrain.yaml` (the shell enforces the
"no pre-existing checkpoints" guard). All arguments live in the yaml.
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

from features import AudioFeatureExtractor, FeatureConfig, C_FEAT, spec_augment  # noqa: E402
from SeldConformer import SeldConformer, SeldConformerConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_seld_jepa")


# =============================================================================
# Dataset — manifest-driven multi-crop audio views
# =============================================================================

class MultiCropAudioDataset(Dataset):
    """Manifest JSON ``{"data": [{"audio_id": "wav/clip.wav"}, ...], "meta": {...}}``.

    ``audio_path = dirname(manifest)/audio_id`` (2-channel wav). Each item returns
    ``num_global`` long temporal crops + ``num_local`` short crops of the 4-channel
    feature stack, each with channel-consistent SpecAugment applied.
    """

    def __init__(self, manifest: str, feat_cfg: FeatureConfig, *, num_global: int,
                 num_local: int, global_seconds: float, local_seconds: float,
                 specaug: dict):
        self.root = os.path.dirname(os.path.abspath(manifest))
        with open(manifest) as f:
            blob = json.load(f)
        self.samples = blob["data"]
        self.meta = blob.get("meta", {})
        if not self.samples:
            raise ValueError(f"Empty manifest: {manifest}")
        self.fx = AudioFeatureExtractor(feat_cfg)
        self.num_global = num_global
        self.num_local = num_local
        self.global_frames = max(1, round(global_seconds * feat_cfg.sample_rate / feat_cfg.hop_length))
        self.local_frames = max(1, round(local_seconds * feat_cfg.sample_rate / feat_cfg.hop_length))
        self.specaug = specaug

    def __len__(self) -> int:
        return len(self.samples)

    def _load_feat(self, path: str) -> torch.Tensor:
        wav, _ = sf.read(path, dtype="float32", always_2d=True)   # (T, C)
        wav = torch.from_numpy(wav).transpose(0, 1).contiguous()  # (C, T)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        with torch.no_grad():
            return self.fx(wav)                                   # (4, T_f, M)

    @staticmethod
    def _crop(feat: torch.Tensor, length: int) -> torch.Tensor:
        """Random temporal crop to `length` frames; wrap-pad if too short."""
        t = feat.shape[1]
        if t < length:
            reps = (length + t - 1) // t
            feat = feat.repeat(1, reps, 1)[:, :length]
            t = length
        start = random.randint(0, t - length)
        return feat[:, start:start + length]

    def _view(self, feat: torch.Tensor, length: int) -> torch.Tensor:
        return spec_augment(self._crop(feat, length), **self.specaug)

    def __getitem__(self, idx: int) -> dict:
        path = os.path.join(self.root, self.samples[idx]["audio_id"])
        feat = self._load_feat(path)
        globals_ = torch.stack([self._view(feat, self.global_frames)
                                for _ in range(self.num_global)])
        locals_ = (torch.stack([self._view(feat, self.local_frames)
                                for _ in range(self.num_local)])
                   if self.num_local > 0 else torch.empty(0))
        return {"globals": globals_, "locals": locals_, "audio_path": path}


def collate_fn(batch: list[dict]) -> dict:
    out = {
        "globals": torch.stack([b["globals"] for b in batch]),    # (B, ng, 4, Tg, M)
        "audio_path": [b["audio_path"] for b in batch],
    }
    locals_ = [b["locals"] for b in batch]
    out["locals"] = (torch.stack(locals_) if locals_[0].numel() > 0
                     else torch.empty(0))
    return out


# =============================================================================
# transformers wrapper
# =============================================================================

class SeldJEPAHFConfig(PretrainedConfig):
    model_type = "seld_jepa"

    def __init__(self, model_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.model_kwargs = model_kwargs or {}


class SeldJEPAModel(PreTrainedModel):
    config_class = SeldJEPAHFConfig

    def __init__(self, config: SeldJEPAHFConfig):
        super().__init__(config)
        self.le = SeldConformer(SeldConformerConfig(**config.model_kwargs))
        self.post_init()

    def _init_weights(self, module):                              # self-initialises
        pass

    def forward(self, globals=None, locals=None, **_) -> dict:
        loc = locals if (locals is not None and locals.numel() > 0) else None
        return self.le.compute_loss(globals, loc)


# =============================================================================
# Callbacks
# =============================================================================

def _unwrap(model) -> SeldJEPAModel:
    return model.module if hasattr(model, "module") else model


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class ParamCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        le = _unwrap(model).le
        groups = {
            "frontend (conv subsample)": le.encoder.frontend,
            "conformer blocks": le.encoder.blocks,
            "predictor (JEPA)": le.predictor,
        }
        logger.info("==== trainable parameters per module ====")
        for name, mod in groups.items():
            logger.info("  %-28s %15s", name, f"{_count(mod):,}")
        logger.info("  %-28s %15s", "TOTAL trainable", f"{_count(le):,}")
        logger.info("=========================================")


class FirstSampleLogCallback(TrainerCallback):
    def __init__(self):
        self.done = False

    def mark(self, audio_path, globals_: torch.Tensor):
        if self.done:
            return
        self.done = True
        logger.info(
            "==== first batch / first sample (pre-feed) ====\n"
            "  audio path : %s\n  globals    : shape=%s dtype=%s",
            audio_path[0] if audio_path else "<unknown>",
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
    """Dump a portable `.pt` (filename carries the step) at every HF checkpoint."""

    def __init__(self, model_kwargs: dict):
        self.model_kwargs = model_kwargs

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        le = _unwrap(model).le
        out = os.path.join(args.output_dir, f"seld_jepa_step{state.global_step}.pt")
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in le.state_dict().items()},
                    "model_kwargs": self.model_kwargs}, out)
        logger.info("saved SeldConformer checkpoint -> %s", out)


# =============================================================================
# Trainer subclass
# =============================================================================

class SeldJEPAHFTrainer(Trainer):
    def __init__(self, *a, first_cb=None, loss_cb=None, **kw):
        super().__init__(*a, **kw)
        self._first_cb = first_cb
        self._loss_cb = loss_cb

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_cb is not None:
            self._first_cb.mark(inputs.get("audio_path"), inputs["globals"])
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


def _build_model_kwargs(c: _Cfg) -> dict:
    f, e, ob, v = c.sec("features"), c.sec("encoder"), c.sec("objective"), c.sec("views")
    kw: dict = {"in_chans": C_FEAT, "n_mels": int(f.get("n_mels", 64))}
    for k in ("encoder_dim", "num_layers", "num_heads", "ffn_expansion",
              "conv_kernel", "chunk_frames", "frontend_hidden", "dropout", "pool"):
        if k in e:
            kw[k] = e[k]
    for k in ("sigreg_coeff", "num_slices", "num_points", "pred_hidden"):
        if k in ob:
            kw[k] = ob[k]
    for k in ("num_global", "num_local"):
        if k in v:
            kw[k] = v[k]
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config_pretrain.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = _Cfg(yaml.safe_load(f))
    d, v, mk, t, o = (cfg.sec("data"), cfg.sec("views"), cfg.sec("masking"),
                      cfg.sec("train"), cfg.sec("optim"))

    seed = int(t.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    feat_cfg = _feature_config(cfg.sec("features"))
    model_kwargs = _build_model_kwargs(cfg)
    logger.info("SeldConformer kwargs: %s", model_kwargs)
    model = SeldJEPAModel(SeldJEPAHFConfig(model_kwargs=model_kwargs))

    init_from = t.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location="cpu", weights_only=False)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.le.load_state_dict(sd, strict=False)
        applied = len(sd) - len(unexpected)
        if applied == 0:
            raise RuntimeError(f"init_from={init_from!r}: 0/{len(sd)} keys applied — aborting.")
        logger.info("init_from %s: applied %d/%d keys", init_from, applied, len(sd))

    specaug = dict(
        n_time_masks=int(mk.get("n_time_masks", 2)),
        time_width=int(mk.get("time_width", 16)),
        n_freq_masks=int(mk.get("n_freq_masks", 2)),
        freq_width=int(mk.get("freq_width", 8)),
        spectral_only=bool(mk.get("spectral_only", False)),
    )
    ds_kw = dict(
        num_global=int(v.get("num_global", 2)),
        num_local=int(v.get("num_local", 4)),
        global_seconds=float(v.get("global_seconds", 2.0)),
        local_seconds=float(v.get("local_seconds", 0.6)),
        specaug=specaug,
    )
    train_ds = MultiCropAudioDataset(d["train_manifest"], feat_cfg, **ds_kw)
    eval_ds = (MultiCropAudioDataset(d["valid_manifest"], feat_cfg, **ds_kw)
               if d.get("valid_manifest") else None)
    logger.info("train clips: %d%s", len(train_ds),
                f" | valid clips: {len(eval_ds)}" if eval_ds else "")

    use_cpu = not torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=t["output_dir"],
        max_steps=int(o.get("max_steps", 100_000)),
        per_device_train_batch_size=int(o.get("per_device_train_batch_size", 32)),
        per_device_eval_batch_size=int(o.get("per_device_eval_batch_size",
                                             o.get("per_device_train_batch_size", 32))),
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
    trainer = SeldJEPAHFTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collate_fn,
        first_cb=first_cb, loss_cb=loss_cb,
        callbacks=[ParamCountCallback(), first_cb, loss_cb, SaveModelCallback(model_kwargs)],
    )

    trainer.train()
    final_pt = os.path.join(targs.output_dir, "seld_jepa_final.pt")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.le.state_dict().items()},
                "model_kwargs": model_kwargs}, final_pt)
    logger.info("training complete; final SeldConformer -> %s", final_pt)


if __name__ == "__main__":
    main()
