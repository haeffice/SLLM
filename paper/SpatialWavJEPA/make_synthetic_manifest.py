"""Generate a tiny synthetic dataset for CPU smoke-testing Spatial WavJEPA.

Writes random mono AudioSet-like wavs, random binaural BRIR `.npy` files,
and train/valid manifest JSONs into `--out`:

    <out>/wav/clip_*.wav            mono, 16 kHz, ~3 s
    <out>/brir/brir_*.npy           shape (2, R)
    <out>/train.json , <out>/valid.json
    <out>/config.yaml               points at the above; tiny smoke settings

Usage:
    python make_synthetic_manifest.py --out /tmp/swj_smoke
    bash run_train_SpatialWavJEPA.sh /tmp/swj_smoke/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import soundfile as sf
import yaml

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-valid", type=int, default=2)
    ap.add_argument("--n-brir", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--brir-len", type=int, default=2400)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    out = os.path.abspath(args.out)
    wav_dir = os.path.join(out, "wav")
    brir_dir = os.path.join(out, "brir")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(brir_dir, exist_ok=True)

    n_total = args.n_train + args.n_valid
    samples = int(args.seconds * SR)
    for i in range(n_total):
        sig = (0.1 * rng.standard_normal(samples)).astype(np.float32)
        sf.write(os.path.join(wav_dir, f"clip_{i:03d}.wav"), sig, SR)

    for j in range(args.n_brir):
        # Decaying-noise stereo BRIR (rough but valid shape (2, R)).
        decay = np.exp(-np.linspace(0, 6, args.brir_len))
        brir = (rng.standard_normal((2, args.brir_len)) * decay).astype(np.float32)
        np.save(os.path.join(brir_dir, f"brir_{j:03d}.npy"), brir)

    def manifest(idxs):
        return {"data": [
            {"audio_id": f"clip_{i:03d}.wav",
             "reverb_id": f"brir_{i % args.n_brir:03d}.npy"}
            for i in idxs
        ]}

    with open(os.path.join(out, "train.json"), "w") as f:
        json.dump(manifest(range(args.n_train)), f, indent=2)
    with open(os.path.join(out, "valid.json"), "w") as f:
        json.dump(manifest(range(args.n_train, n_total)), f, indent=2)

    cfg = {
        "data": {
            "train_manifest": os.path.join(out, "train.json"),
            "valid_manifest": os.path.join(out, "valid.json"),
            "audio_root": wav_dir,
            "reverb_root": brir_dir,
            "sample_rate": SR,
            "process_audio_seconds": 2.01,
        },
        "model": {
            "encoder_d_model": 768, "encoder_nhead": 12,
            "encoder_dim_feedforward": 3072, "encoder_num_layers": 12,
            "decoder_d_model": 384, "decoder_nhead": 12,
            "decoder_dim_feedforward": 1536, "decoder_num_layers": 12,
            "average_top_k_layers": 12,
        },
        "masking": {
            "context_mask_prob": 0.065, "context_mask_length": 10,
            "target_prob": 0.025, "target_length": 10,
            "target_masks_per_context": 8, "ratio_cutoff": 0.10,
            "masker_kind": "time-inverse",
        },
        "ema": {"ema_decay": 0.999, "ema_end_decay": 0.99999,
                "ema_anneal_end_step": 100000},
        "optim": {
            "learning_rate": 2.0e-4, "adam_beta1": 0.9, "adam_beta2": 0.98,
            "weight_decay": 0.04, "warmup_steps": 2, "max_steps": 4,
            "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
        },
        "train": {
            "output_dir": os.path.join(out, "ckpts"),
            "logging_steps": 1, "save_steps": 2, "eval_steps": 2,
            "num_workers": 0, "seed": 42,
        },
    }
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"synthetic dataset + config written to {out}")
    print(f"  smoke train: bash run_train_SpatialWavJEPA.sh {out}/config.yaml")


if __name__ == "__main__":
    main()
