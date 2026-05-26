"""PhaseCoder supervised training (PyTorch 2.8).

Trains the `PhaseCoder` encoder (multichannel audio + mic coords → spatial
token → azimuth/elevation/distance) on the dataset produced by
`make_phasecoder_dataset.py`.

Stack (per `paper/CLAUDE.md`):
    * `transformers.PreTrainedModel` wrapper around `PhaseCoder`
    * `transformers.Trainer` training loop, AdamW + cosine schedule
    * `torch.utils.data.Dataset` / `DataLoader` over the `.npz` manifest;
      a collate that **pads the variable mic count** and builds a channel mask
      (geometry-agnostic batching)
    * `accelerate` + `torchrun` for multi-GPU (CPU-compatible)
    * logs: per-module trainable params; first batch's first sample AUDIO PATH
      before the model is fed; step / train_loss / valid_loss / lr every
      `logging_steps`; checkpoints every `save_steps` with the step number in
      the filename; abort if an `init_from` warm-start matches nothing.

Run via `run_train_PhaseCoder.sh config.yaml` (the shell enforces the
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

from PhaseCoder import (  # noqa: E402
    PhaseCoder,
    N_AZIMUTH, N_ELEVATION, N_DISTANCE, DIST_MIN, DIST_MAX,
    N_FFT, HOP, EMBED_DIM, N_BLOCKS, N_HEADS, FFN_DIM,
    phasecoder_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_phasecoder")


# =============================================================================
# Dataset — reads the .npz samples written by make_phasecoder_dataset.py
# =============================================================================

class PhaseCoderDataset(Dataset):
    """Manifest-driven (audio, mic_coords) → (az/el/dist bin) dataset. Each
    `.npz` holds ``audio`` (C, T), ``mic_coords`` (C, 3) and integer labels;
    the mic count C varies per sample (geometry-agnostic)."""

    def __init__(self, manifest: str):
        self.root = os.path.dirname(os.path.abspath(manifest))
        with open(manifest, "r") as f:
            blob = json.load(f)
        self.samples = blob["data"]
        self.meta = blob.get("meta", {})
        if not self.samples:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path = os.path.join(self.root, self.samples[idx]["id"])
        with np.load(path) as z:
            audio = z["audio"].astype(np.float32)               # (C, T)
            coords = z["mic_coords"].astype(np.float32)         # (C, 3)
            az = int(z["az_bin"]); el = int(z["el_bin"]); di = int(z["dist_bin"])
            az_deg = float(z["azimuth"]); el_deg = float(z["elevation"]); di_m = float(z["distance"])
        return {
            "audio": torch.from_numpy(audio),
            "mic_coords": torch.from_numpy(coords),
            "azimuth": az, "elevation": el, "distance": di,
            "azimuth_deg": az_deg, "elevation_deg": el_deg, "distance_m": di_m,
            "audio_path": path,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad the variable mic count to the batch max; build a channel mask
    (True = real mic). Audio padded with zeros, coords with zeros."""
    b = len(batch)
    cmax = max(item["audio"].shape[0] for item in batch)
    t = batch[0]["audio"].shape[1]
    audio = torch.zeros(b, cmax, t)
    coords = torch.zeros(b, cmax, 3)
    mask = torch.zeros(b, cmax, dtype=torch.bool)
    for i, item in enumerate(batch):
        c = item["audio"].shape[0]
        audio[i, :c] = item["audio"]
        coords[i, :c] = item["mic_coords"]
        mask[i, :c] = True
    return {
        "audio": audio,
        "mic_coords": coords,
        "channel_mask": mask,
        "azimuth": torch.tensor([item["azimuth"] for item in batch], dtype=torch.long),
        "elevation": torch.tensor([item["elevation"] for item in batch], dtype=torch.long),
        "distance": torch.tensor([item["distance"] for item in batch], dtype=torch.long),
        "azimuth_deg": torch.tensor([item["azimuth_deg"] for item in batch]),
        "elevation_deg": torch.tensor([item["elevation_deg"] for item in batch]),
        "distance_m": torch.tensor([item["distance_m"] for item in batch]),
        "audio_path": [item["audio_path"] for item in batch],
    }


# =============================================================================
# transformers wrapper
# =============================================================================

class PhaseCoderHFConfig(PretrainedConfig):
    model_type = "phasecoder"

    def __init__(self, model_kwargs: Optional[dict] = None, lambda_az: float = 1.0,
                 lambda_el: float = 1.0, lambda_di: float = 0.5, **kw):
        super().__init__(**kw)
        self.model_kwargs = model_kwargs or {}
        self.lambda_az = lambda_az
        self.lambda_el = lambda_el
        self.lambda_di = lambda_di


class PhaseCoderModel(PreTrainedModel):
    """Thin HF wrapper: owns a `PhaseCoder`, computes the weighted multitask CE
    when labels are present, returns `{"loss": ...}` for `Trainer`."""

    config_class = PhaseCoderHFConfig

    def __init__(self, config: PhaseCoderHFConfig):
        super().__init__(config)
        self.phasecoder = PhaseCoder(**config.model_kwargs)
        self.lambda_az = config.lambda_az
        self.lambda_el = config.lambda_el
        self.lambda_di = config.lambda_di
        self.post_init()

    def _init_weights(self, module):                # PhaseCoder self-initialises.
        pass

    def forward(self, audio=None, mic_coords=None, channel_mask=None,
                azimuth=None, elevation=None, distance=None, **_) -> dict:
        out = self.phasecoder(audio, mic_coords, channel_mask)
        if azimuth is not None:
            loss, parts = phasecoder_loss(out, azimuth, elevation, distance,
                                          self.lambda_az, self.lambda_el, self.lambda_di)
            out["loss"] = loss
            out.update(parts)
        return out


# =============================================================================
# Callbacks
# =============================================================================

def _unwrap(model) -> PhaseCoderModel:
    return model.module if hasattr(model, "module") else model


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class ParamCountCallback(TrainerCallback):
    """Log per-module trainable parameter counts once at train start."""

    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        pc = _unwrap(model).phasecoder
        groups = {
            "patch extractor (STFT proj)": pc.patch,
            "transformer encoder": pc.encoder,
            "spatial-token MLP": pc.token_mlp,
            "heads (az+el+dist)": nn.ModuleList([pc.head_azimuth, pc.head_elevation, pc.head_distance]),
        }
        logger.info("==== trainable parameters per module ====")
        for name, mod in groups.items():
            logger.info("  %-28s %15s", name, f"{_count(mod):,}")
        logger.info("  %-28s %15s", "TOTAL trainable", f"{_count(pc):,}")
        logger.info("=========================================")


class FirstSampleLogCallback(TrainerCallback):
    """Log the first batch's first sample AUDIO PATH before the model is fed."""

    def __init__(self):
        self.done = False

    def mark(self, audio_path, audio, coords, az, el, di):
        if self.done:
            return
        self.done = True
        logger.info(
            "==== first batch / first sample (pre-feed) ====\n"
            "  audio path : %s\n  audio      : shape=%s dtype=%s\n"
            "  mic_coords : shape=%s (n_mics=%d)\n  labels     : az_bin=%d el_bin=%d dist_bin=%d",
            audio_path[0] if audio_path else "<unknown>",
            tuple(audio.shape), audio.dtype, tuple(coords.shape),
            int(coords.shape[1]), int(az[0]), int(el[0]), int(di[0]),
        )


class SaveCheckpointCallback(TrainerCallback):
    """On every HF checkpoint, also dump a portable `.pt` whose filename carries
    the step number (loadable via `PhaseCoder.from_checkpoint`)."""

    def __init__(self, model_kwargs: dict):
        self.model_kwargs = model_kwargs

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        pc = _unwrap(model).phasecoder
        out = os.path.join(args.output_dir, f"phasecoder_step{state.global_step}.pt")
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in pc.state_dict().items()},
                    "model_kwargs": self.model_kwargs}, out)
        logger.info("saved PhaseCoder checkpoint -> %s", out)


# =============================================================================
# Trainer subclass — paper optimiser/schedule + first-sample logging
# =============================================================================

class PhaseCoderHFTrainer(Trainer):
    def __init__(self, *a, first_sample_cb: FirstSampleLogCallback = None, **kw):
        super().__init__(*a, **kw)
        self._first_sample_cb = first_sample_cb

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_sample_cb is not None:
            self._first_sample_cb.mark(inputs.get("audio_path"), inputs["audio"],
                                       inputs["mic_coords"], inputs["azimuth"],
                                       inputs["elevation"], inputs["distance"])
        outputs = model(audio=inputs["audio"], mic_coords=inputs["mic_coords"],
                        channel_mask=inputs["channel_mask"], azimuth=inputs["azimuth"],
                        elevation=inputs["elevation"], distance=inputs["distance"])
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


def _build_model_kwargs(c: _Cfg) -> dict:
    m = c.sec("model")
    return dict(
        embed_dim=int(m.get("embed_dim", EMBED_DIM)),
        n_blocks=int(m.get("n_blocks", N_BLOCKS)),
        n_heads=int(m.get("n_heads", N_HEADS)),
        ffn_dim=int(m.get("ffn_dim", FFN_DIM)),
        n_fft=int(m.get("n_fft", N_FFT)),
        hop=int(m.get("hop", HOP)),
        dropout=float(m.get("dropout", 0.0)),
        n_azimuth=int(m.get("n_azimuth", N_AZIMUTH)),
        n_elevation=int(m.get("n_elevation", N_ELEVATION)),
        n_distance=int(m.get("n_distance", N_DISTANCE)),
        dist_min=float(m.get("dist_min", DIST_MIN)),
        dist_max=float(m.get("dist_max", DIST_MAX)),
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
    logger.info("PhaseCoder model kwargs: %s", model_kwargs)
    hf_cfg = PhaseCoderHFConfig(
        model_kwargs=model_kwargs,
        lambda_az=float(lo.get("lambda_az", 1.0)),
        lambda_el=float(lo.get("lambda_el", 1.0)),
        lambda_di=float(lo.get("lambda_di", 0.5)),
    )
    model = PhaseCoderModel(hf_cfg)

    # Optional warm-start — abort if nothing loads.
    init_from = t.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location="cpu", weights_only=False)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.phasecoder.load_state_dict(sd, strict=False)
        applied = len(sd) - len(unexpected)
        if applied == 0:
            raise RuntimeError(f"init_from={init_from!r}: 0/{len(sd)} keys applied.")
        logger.info("init_from %s: applied %d/%d keys (missing=%d, unexpected=%d)",
                    init_from, applied, len(sd), len(missing), len(unexpected))

    train_ds = PhaseCoderDataset(d["train_manifest"])
    eval_ds = PhaseCoderDataset(d["valid_manifest"]) if d.get("valid_manifest") else None
    logger.info("train samples: %d%s", len(train_ds),
                f" | valid samples: {len(eval_ds)}" if eval_ds else "")

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
        warmup_steps=int(o.get("warmup_steps", 1000)),
        logging_steps=int(t.get("logging_steps", 50)),
        save_steps=int(t.get("save_steps", 1000)),
        save_total_limit=t.get("save_total_limit"),
        eval_strategy=("steps" if eval_ds is not None else "no"),
        eval_steps=int(t.get("eval_steps", 1000)),
        dataloader_num_workers=int(t.get("num_workers", 4)),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,        # keep audio/mic_coords/labels/audio_path
        label_names=["azimuth", "elevation", "distance"],
        use_cpu=use_cpu,
        ddp_backend=("gloo" if use_cpu else None),
        max_grad_norm=float(o.get("max_grad_norm", 1.0)),
    )

    first_cb = FirstSampleLogCallback()
    trainer = PhaseCoderHFTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        first_sample_cb=first_cb,
        callbacks=[ParamCountCallback(), first_cb, SaveCheckpointCallback(model_kwargs)],
    )

    trainer.train()
    final_pt = os.path.join(targs.output_dir, "phasecoder_final.pt")
    torch.save({"state_dict": {k: v.detach().cpu()
                               for k, v in model.phasecoder.state_dict().items()},
                "model_kwargs": model_kwargs}, final_pt)
    logger.info("training complete; final PhaseCoder -> %s", final_pt)


if __name__ == "__main__":
    main()
