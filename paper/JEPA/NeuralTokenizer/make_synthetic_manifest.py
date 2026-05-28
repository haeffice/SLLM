"""Generate a tiny synthetic 24 kHz speech-like dataset for CPU smoke-testing
the JEPA Neural Tokenizer (both stages).

Writes short `.wav` clips (sums of random sinusoids + a little noise — enough
signal for the encoder, FSQ and HiFi-GAN to run), train/valid/test manifests,
and TWO shrunken configs:

    <out>/wav/clip_*.wav
    <out>/train.json , <out>/valid.json , <out>/test.json
    <out>/config.yaml          stage 1 (SSL)         -> stage1_ckpts/
    <out>/config_stage2.yaml   stage 2 (recon, GAN)  -> stage2_ckpts/, frozen enc from stage1

Usage:
    python make_synthetic_manifest.py --out /tmp/nt_smoke
    bash run_train_NeuralTokenizer.sh /tmp/nt_smoke/config.yaml          # stage 1
    bash run_train_NeuralTokenizer.sh /tmp/nt_smoke/config_stage2.yaml   # stage 2
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import soundfile as sf
import yaml


def _make_wav(rng, seconds, sr):
    t = np.arange(int(seconds * sr)) / sr
    wave = np.zeros_like(t)
    for _ in range(rng.integers(2, 5)):                # a few partials
        f = rng.uniform(100, 3000)
        wave += rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * f * t)
    wave += 0.01 * rng.standard_normal(t.shape)
    wave /= np.max(np.abs(wave)) + 1e-6
    return wave.astype(np.float32)


# tiny CPU-smoke model: hop 512 (3 strides) so a 2 s clip -> ~93 frames.
_SMOKE_MODEL = {
    "conv_channels": [16, 32, 32, 32],
    "conv_strides": [8, 8, 8],
    "stem_kernel": 7,
    "use_daam": True, "daam_k": 4, "daam_alpha": 0.05,
    "num_layers": 1, "num_heads": 4, "ff_mult": 2, "conv_kernel": 7,
    "dropout": 0.0,
    "pred_layers": 1, "pred_heads": 4, "mask_ratio": 0.5, "mask_min_span": 2,
    "ema_decay": 0.996,
    "fsq_dim": 16, "fsq_level": 4, "pack_group": 4,
    "dec_channels": [], "dec_min_channels": 8,
    "lambda_stft": 2.0, "lambda_gan": 0.1, "disc_warmup": 0,   # exercise GAN in smoke
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-valid", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--sr", type=int, default=24000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    out = os.path.abspath(args.out)
    wav_dir = os.path.join(out, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    def write_split(n: int, tag: str) -> str:
        items = []
        for i in range(n):
            name = f"{tag}_clip_{i:04d}.wav"
            sf.write(os.path.join(wav_dir, name),
                     _make_wav(rng, args.seconds, args.sr), args.sr)
            items.append({"audio_id": os.path.join("wav", name)})
        path = os.path.join(out, f"{tag}.json")
        with open(path, "w") as f:
            json.dump({"data": items}, f, indent=2)
        return path

    train_json = write_split(args.n_train, "train")
    valid_json = write_split(args.n_valid, "valid")
    test_json = write_split(args.n_test, "test")

    data = {"train_manifest": train_json, "valid_manifest": valid_json,
            "audio_root": out, "sample_rate": args.sr,
            "clip_seconds": args.seconds}
    optim = {"learning_rate": 1.5e-4, "adam_beta1": 0.8, "adam_beta2": 0.99,
             "weight_decay": 1.0e-3, "warmup_steps": 2, "max_steps": 4,
             "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
             "max_grad_norm": 1.0}
    train_common = {"logging_steps": 1, "save_steps": 2, "eval_steps": 2,
                    "save_total_limit": None, "num_workers": 0, "seed": 42}

    stage1_dir = os.path.join(out, "stage1_ckpts")
    stage2_dir = os.path.join(out, "stage2_ckpts")

    # ---- stage 1 config ----
    cfg1 = {
        "stage": 1, "data": data, "model": dict(_SMOKE_MODEL), "optim": optim,
        "train": {**train_common, "output_dir": stage1_dir,
                  "init_from": None, "stage1_ckpt": None},
        "eval": {"test_manifest": test_json, "audio_root": out},
    }
    cfg1_path = os.path.join(out, "config.yaml")
    with open(cfg1_path, "w") as f:
        yaml.safe_dump(cfg1, f, sort_keys=False)

    # ---- stage 2 config (frozen encoder from the stage-1 step-4 ckpt) ----
    cfg2 = {
        "stage": 2, "data": data, "model": dict(_SMOKE_MODEL), "optim": optim,
        "train": {**train_common, "output_dir": stage2_dir, "init_from": None,
                  "stage1_ckpt": os.path.join(stage1_dir, "nt_stage1_step4.pt")},
        "eval": {"test_manifest": test_json, "audio_root": out},
    }
    cfg2_path = os.path.join(out, "config_stage2.yaml")
    with open(cfg2_path, "w") as f:
        yaml.safe_dump(cfg2, f, sort_keys=False)

    print(f"wrote {args.n_train}+{args.n_valid}+{args.n_test} wavs -> {wav_dir}")
    print(f"stage-1 config -> {cfg1_path}")
    print(f"stage-2 config -> {cfg2_path}")
    print(f"run: bash run_train_NeuralTokenizer.sh {cfg1_path}")
    print(f"then bash run_train_NeuralTokenizer.sh {cfg2_path}")


if __name__ == "__main__":
    main()
