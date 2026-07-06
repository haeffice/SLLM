"""Generate a tiny synthetic video dataset for CPU smoke-testing V-JEPA 2.

Writes random uint8 clips as `.npy` (so the smoke test needs no
ffmpeg/decord), train/valid manifest JSONs, and a tiny-settings
`config.yaml` into `--out`:

    <out>/vid/clip_*.npy            (T, H, W, 3) uint8
    <out>/train.json , <out>/valid.json
    <out>/config.yaml               points at the above; tiny smoke settings

Usage:
    python make_synthetic_manifest.py --out /tmp/vjepa2_smoke
    bash run_train_VJEPA2.sh /tmp/vjepa2_smoke/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-valid", type=int, default=2)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=64)       # tiny for CPU
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    out = os.path.abspath(args.out)
    vid_dir = os.path.join(out, "vid")
    os.makedirs(vid_dir, exist_ok=True)

    def write_split(n: int, tag: str) -> str:
        items = []
        for i in range(n):
            name = f"{tag}_clip_{i:04d}.npy"
            clip = rng.integers(0, 256, size=(args.frames, args.size,
                                              args.size, 3), dtype=np.uint8)
            np.save(os.path.join(vid_dir, name), clip)
            items.append({"video_id": os.path.join("vid", name)})
        path = os.path.join(out, f"{tag}.json")
        with open(path, "w") as f:
            json.dump({"data": items}, f, indent=2)
        return path

    train_json = write_split(args.n_train, "train")
    valid_json = write_split(args.n_valid, "valid")

    cfg = {
        "data": {
            "train_manifest": train_json,
            "valid_manifest": valid_json,
            "video_root": out,
            "num_frames": args.frames,
            "img_size": args.size,
        },
        "model": {
            "variant": "vit_large",
            "embed_dim": 96,                  # shrink encoder for CPU smoke
            "depth": 2,
            "num_heads": 6,
            "mlp_ratio": 4.0,
            "patch_size": 16,
            "tubelet_size": 2,
            "use_rope": True,
            "predictor_embed_dim": 96,
            "predictor_depth": 2,
            "predictor_num_heads": 6,
            "predictor_mlp_ratio": 4.0,
        },
        "masking": {
            "mask_ratios": [0.15, 0.7],
            "mask_blocks_per_ratio": [4, 1],
            "min_context_ratio": 0.1,
        },
        "ema": {"ema_decay": 0.998, "ema_end_decay": 1.0,
                "ema_anneal_end_step": 50},
        "optim": {
            "learning_rate": 1.0e-4, "adam_beta1": 0.9, "adam_beta2": 0.999,
            "weight_decay": 0.04, "warmup_steps": 2, "max_steps": 4,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1, "max_grad_norm": 1.0,
        },
        "train": {
            "output_dir": os.path.join(out, "ckpts"),
            "logging_steps": 1, "save_steps": 2, "eval_steps": 2,
            "save_total_limit": None, "num_workers": 0, "seed": 42,
            "init_from": None,
        },
    }
    cfg_path = os.path.join(out, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"wrote {args.n_train} train + {args.n_valid} valid clips -> {vid_dir}")
    print(f"smoke config -> {cfg_path}")
    print(f"run: bash run_train_VJEPA2.sh {cfg_path}")


if __name__ == "__main__":
    main()
