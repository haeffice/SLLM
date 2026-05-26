"""BatVision supervised training (PyTorch 2.8).

Trains the `BatVision` U-Net (binaural echoes → depth map) on the BatVision
dataset (V1 office / V2 campus) or the synthetic smoke set from
`make_batvision_dataset.py`.

Stack (per `paper/CLAUDE.md`):
    * `transformers.PreTrainedModel` wrapper around `BatVision`
    * `transformers.Trainer` training loop, AdamW + cosine schedule
    * `torch.utils.data.Dataset` / `DataLoader` reading the BatVision CSVs
    * `accelerate` + `torchrun` for multi-GPU (CPU-compatible)
    * logs: per-module trainable params; first batch's first sample AUDIO PATH
      before the model is fed; step / train_loss / valid_loss / lr every
      `logging_steps`; checkpoints every `save_steps` with the step number in
      the filename; abort if an `init_from` warm-start matches nothing.

Run via `run_train_BatVision.sh config.yaml` (the shell enforces the
"no pre-existing checkpoints" guard). All arguments live in `config.yaml`.
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

from BatVision import (  # noqa: E402
    BatVision,
    IMAGE_SIZE,
    MAX_DEPTH_V1,
    N_CHANNELS,
    N_FFT,
    WIN_LENGTH,
    HOP_LENGTH,
    SPEC_POWER,
    binaural_spectrogram,
    masked_l1_loss,
    resize_2d,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_batvision")

C_SOUND = 340.0          # m/s, matches the reference V2 waveform-cut constant


# =============================================================================
# Dataset — reads the BatVision V1 / V2 CSV layout
# =============================================================================

def _read_wav(path: str) -> np.ndarray:
    """Read a stereo wav as (2, T) float32. Uses soundfile if present (handles
    all PCM/float subtypes), falls back to scipy.io.wavfile."""
    try:
        import soundfile as sf
        wav, _ = sf.read(path, dtype="float32", always_2d=True)   # (T, C)
        return wav.T
    except ImportError:
        from scipy.io import wavfile
        sr, wav = wavfile.read(path)
        wav = wav.astype(np.float32)
        if np.issubdtype(wav.dtype, np.integer):
            wav = wav / np.iinfo(wav.dtype).max
        if wav.ndim == 1:
            wav = wav[:, None]
        return wav.T


class BatVisionDataset(Dataset):
    """BatVision (audio echoes → depth) dataset for both dataset versions.

    `variant='v1'`: a single CSV at `dataset_dir`; columns `depth path`,
    `audio path left`, `audio path right` (binaural `.npy` waveforms, depth
    `.npy` in mm). `variant='v2'`: `dataset_dir` holds per-location folders,
    each with the CSV; columns `depth path`+`depth file name`,
    `audio path`+`audio file name` (stereo `.wav`).

    Returns: spectrogram (2, S, S) float32, depth (1, S, S) float32 (MinMax
    normalised by `max_depth` when `depth_norm`), and the audio file path
    (for first-sample logging; not fed to the model).
    """

    def __init__(self, cfg: dict, csv_name: str):
        self.variant = cfg.get("variant", "v1").lower()
        self.root = cfg["dataset_dir"]
        self.image_size = int(cfg.get("images_size", IMAGE_SIZE))
        self.max_depth = float(cfg.get("max_depth", MAX_DEPTH_V1))
        self.depth_norm = bool(cfg.get("depth_norm", True))
        self.n_fft = int(cfg.get("n_fft", N_FFT))
        self.win_length = int(cfg.get("win_length", WIN_LENGTH))
        self.hop_length = int(cfg.get("hop_length", HOP_LENGTH))

        import pandas as pd
        if self.variant == "v1":
            self.instances = pd.read_csv(os.path.join(self.root, csv_name))
        elif self.variant == "v2":
            blacklist = set(cfg.get("location_blacklist") or [])
            locations = [d for d in sorted(os.listdir(self.root))
                         if os.path.isdir(os.path.join(self.root, d)) and d not in blacklist]
            frames = [pd.read_csv(os.path.join(self.root, loc, csv_name))
                      for loc in locations
                      if os.path.exists(os.path.join(self.root, loc, csv_name))]
            if not frames:
                raise ValueError(f"no '{csv_name}' found under any location in {self.root}")
            self.instances = pd.concat(frames, ignore_index=True)
        else:
            raise ValueError(f"variant must be 'v1' or 'v2', got {self.variant!r}")
        if len(self.instances) == 0:
            raise ValueError(f"empty dataset: {self.root}/{csv_name}")

    def __len__(self) -> int:
        return len(self.instances)

    def _load(self, idx: int):
        row = self.instances.iloc[idx]
        if self.variant == "v1":
            depth_path = os.path.join(self.root, row["depth path"])
            wav_l = np.load(os.path.join(self.root, row["audio path left"])).astype(np.float32)
            wav_r = np.load(os.path.join(self.root, row["audio path right"])).astype(np.float32)
            waveform = np.stack((wav_l, wav_r))                       # (2, T)
            audio_path = os.path.join(self.root, row["audio path left"])
        else:  # v2
            depth_path = os.path.join(self.root, row["depth path"], row["depth file name"])
            audio_path = os.path.join(self.root, row["audio path"], row["audio file name"])
            waveform = _read_wav(audio_path)                          # (2, T)
            # cut the waveform to the round-trip window for max_depth (reference)
            cut = int((2 * self.max_depth / C_SOUND) * 44100)
            waveform = waveform[:, :cut]
        return depth_path, waveform.astype(np.float32), audio_path

    def __getitem__(self, idx: int) -> dict:
        depth_path, waveform, audio_path = self._load(idx)

        # ---- depth: mm → m, clip, (1,H,W), resize, MinMax-normalise ----
        depth = np.load(depth_path).astype(np.float32)
        depth = np.nan_to_num(depth, posinf=0.0, neginf=0.0)
        depth = depth / 1000.0                                        # mm → m
        depth = np.clip(depth, 0.0, self.max_depth)
        depth_t = torch.from_numpy(depth)[None]                       # (1, H, W)
        depth_t = resize_2d(depth_t, self.image_size, mode="bilinear")
        if self.depth_norm:
            depth_t = depth_t / self.max_depth                        # → [0,1]

        # ---- audio: binaural magnitude spectrogram, resize to S×S ----
        wav = torch.from_numpy(waveform)                              # (2, T)
        spec = binaural_spectrogram(wav, n_fft=self.n_fft, win_length=self.win_length,
                                    hop_length=self.hop_length, power=SPEC_POWER)
        spec = resize_2d(spec, self.image_size, mode="bilinear")      # (2, S, S)

        return {"spectrogram": spec.float(), "depth": depth_t.float(), "audio_path": audio_path}


def collate_fn(batch: list[dict]) -> dict:
    return {
        "spectrogram": torch.stack([b["spectrogram"] for b in batch]),  # (B, 2, S, S)
        "depth": torch.stack([b["depth"] for b in batch]),              # (B, 1, S, S)
        "audio_path": [b["audio_path"] for b in batch],
    }


# =============================================================================
# transformers wrapper
# =============================================================================

class BatVisionHFConfig(PretrainedConfig):
    model_type = "batvision"

    def __init__(self, model_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.model_kwargs = model_kwargs or {}


class BatVisionModel(PreTrainedModel):
    """Thin HF wrapper: owns a `BatVision`, computes masked-L1 loss when the
    depth label is present, returns `{"loss": ...}` for `Trainer`."""

    config_class = BatVisionHFConfig

    def __init__(self, config: BatVisionHFConfig):
        super().__init__(config)
        self.batvision = BatVision(**config.model_kwargs)
        self.post_init()

    def _init_weights(self, module):                # BatVision self-initialises.
        pass

    def forward(self, spectrogram=None, depth=None, **_) -> dict:
        pred = self.batvision(spectrogram)
        result = {"depth_pred": pred}
        if depth is not None:
            result["loss"] = masked_l1_loss(pred, depth)
        return result


# =============================================================================
# Callbacks
# =============================================================================

def _unwrap(model) -> BatVisionModel:
    return model.module if hasattr(model, "module") else model


def _count(params) -> int:
    return sum(p.numel() for p in params if p.requires_grad)


class ParamCountCallback(TrainerCallback):
    """Log trainable parameter counts (encoder/decoder/norm split) at start."""

    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        net = _unwrap(model).batvision.net
        down, up, norm, other = [], [], [], []
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                down += list(m.parameters(recurse=False))
            elif isinstance(m, nn.ConvTranspose2d):
                up += list(m.parameters(recurse=False))
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                norm += list(m.parameters(recurse=False))
            else:
                other += list(m.parameters(recurse=False))
        logger.info("==== trainable parameters per module ====")
        logger.info("  %-28s %15s", "encoder (downsample Conv2d)", f"{_count(down):,}")
        logger.info("  %-28s %15s", "decoder (upsample ConvT2d)", f"{_count(up):,}")
        logger.info("  %-28s %15s", "normalization", f"{_count(norm):,}")
        if _count(other):
            logger.info("  %-28s %15s", "other", f"{_count(other):,}")
        logger.info("  %-28s %15s", "TOTAL trainable",
                    f"{_count(net.parameters()):,}")
        logger.info("=========================================")


class FirstSampleLogCallback(TrainerCallback):
    """Log the first batch's first sample AUDIO PATH before the model is fed."""

    def __init__(self):
        self.done = False

    def mark(self, audio_path, spectrogram, depth):
        if self.done:
            return
        self.done = True
        valid = int((depth[0] != 0).sum().item())
        logger.info(
            "==== first batch / first sample (pre-feed) ====\n"
            "  audio path  : %s\n  spectrogram : shape=%s dtype=%s\n"
            "  depth       : shape=%s valid_px=%d range=[%.3f, %.3f]",
            audio_path[0] if audio_path else "<unknown>",
            tuple(spectrogram.shape), spectrogram.dtype,
            tuple(depth.shape), valid,
            float(depth[0].min()), float(depth[0].max()),
        )


class SaveCheckpointCallback(TrainerCallback):
    """On every HF checkpoint, also dump a portable `.pt` whose filename
    carries the step number (loadable via `BatVision.from_checkpoint`)."""

    def __init__(self, model_kwargs: dict):
        self.model_kwargs = model_kwargs

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        bv = _unwrap(model).batvision
        out = os.path.join(args.output_dir, f"batvision_step{state.global_step}.pt")
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in bv.state_dict().items()},
                    "model_kwargs": self.model_kwargs}, out)
        logger.info("saved BatVision checkpoint -> %s", out)


# =============================================================================
# Trainer subclass — paper optimiser/schedule + first-sample logging
# =============================================================================

class BatVisionHFTrainer(Trainer):
    def __init__(self, *a, first_sample_cb: FirstSampleLogCallback = None, **kw):
        super().__init__(*a, **kw)
        self._first_sample_cb = first_sample_cb

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_sample_cb is not None:
            self._first_sample_cb.mark(inputs.get("audio_path"),
                                       inputs["spectrogram"], inputs["depth"])
        outputs = model(spectrogram=inputs["spectrogram"], depth=inputs["depth"])
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
    m, d = c.sec("model"), c.sec("data")
    return dict(
        generator=str(m.get("generator", "unet_256")),
        ngf=int(m.get("ngf", 64)),
        in_channels=int(m.get("in_channels", N_CHANNELS)),
        norm=str(m.get("norm", "batch")),
        depth_norm=bool(d.get("depth_norm", True)),
        max_depth=float(d.get("max_depth", MAX_DEPTH_V1)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = _Cfg(yaml.safe_load(f))
    d, t, o = cfg.sec("data"), cfg.sec("train"), cfg.sec("optim")

    seed = int(t.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_kwargs = _build_model_kwargs(cfg)
    logger.info("BatVision model kwargs: %s", model_kwargs)

    # image size must match the chosen generator (unet_256→256, unet_128→128).
    expected_size = {"unet_256": 256, "unet_128": 128}[model_kwargs["generator"]]
    if int(d.get("images_size", IMAGE_SIZE)) != expected_size:
        raise ValueError(
            f"data.images_size ({d.get('images_size')}) must equal "
            f"{expected_size} for generator {model_kwargs['generator']!r}")
    if str(d.get("audio_format", "spectrogram")) != "spectrogram":
        raise ValueError("the U-Net baseline takes a spectrogram; set data.audio_format: spectrogram")

    hf_cfg = BatVisionHFConfig(model_kwargs=model_kwargs)
    model = BatVisionModel(hf_cfg)

    # Optional warm-start — abort if nothing loads.
    init_from = t.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location="cpu", weights_only=False)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.batvision.load_state_dict(sd, strict=False)
        applied = len(sd) - len(unexpected)
        if applied == 0:
            raise RuntimeError(f"init_from={init_from!r}: 0/{len(sd)} keys applied.")
        logger.info("init_from %s: applied %d/%d keys (missing=%d, unexpected=%d)",
                    init_from, applied, len(sd), len(missing), len(unexpected))

    train_ds = BatVisionDataset(d, d["train_csv"])
    eval_ds = BatVisionDataset(d, d["valid_csv"]) if d.get("valid_csv") else None
    logger.info("train samples: %d%s", len(train_ds),
                f" | valid samples: {len(eval_ds)}" if eval_ds else "")

    use_cpu = not torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=float(o.get("num_train_epochs", 100)),
        max_steps=int(o.get("max_steps", -1)),
        per_device_train_batch_size=int(o.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(o.get("per_device_eval_batch_size",
                                             o.get("per_device_train_batch_size", 16))),
        gradient_accumulation_steps=int(o.get("gradient_accumulation_steps", 1)),
        learning_rate=float(o.get("learning_rate", 1.0e-3)),
        weight_decay=float(o.get("weight_decay", 0.0)),
        adam_beta1=float(o.get("adam_beta1", 0.9)),
        adam_beta2=float(o.get("adam_beta2", 0.999)),
        warmup_steps=int(o.get("warmup_steps", 0)),
        logging_steps=int(t.get("logging_steps", 50)),
        save_steps=int(t.get("save_steps", 1000)),
        save_total_limit=t.get("save_total_limit"),
        eval_strategy=("steps" if eval_ds is not None else "no"),
        eval_steps=int(t.get("eval_steps", 1000)),
        dataloader_num_workers=int(t.get("num_workers", 4)),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,        # keep spectrogram/depth/audio_path
        label_names=["depth"],
        use_cpu=use_cpu,
        ddp_backend=("gloo" if use_cpu else None),
        max_grad_norm=float(o.get("max_grad_norm", 1.0)),
    )

    first_cb = FirstSampleLogCallback()
    trainer = BatVisionHFTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        first_sample_cb=first_cb,
        callbacks=[ParamCountCallback(), first_cb, SaveCheckpointCallback(model_kwargs)],
    )

    trainer.train()
    final_pt = os.path.join(targs.output_dir, "batvision_final.pt")
    torch.save({"state_dict": {k: v.detach().cpu()
                               for k, v in model.batvision.state_dict().items()},
                "model_kwargs": model_kwargs}, final_pt)
    logger.info("training complete; final BatVision -> %s", final_pt)


if __name__ == "__main__":
    main()
