"""Generate a tiny synthetic moving-shapes dataset for CPU smoke-testing
Causal-JEPA.

Each clip has a few colored squares bouncing on a dark background — a
minimal multi-object scene so Slot Attention has objects to bind to. Clips
are saved as `.npy` (T, H, W, 3) uint8 (no codec dependency), with
train/valid/test manifests and a shrunken `config.yaml`.

    <out>/vid/clip_*.npy            (T, H, W, 3) uint8
    <out>/train.json , <out>/valid.json , <out>/test.json
    <out>/config.yaml               tiny smoke settings

Usage:
    python make_synthetic_manifest.py --out /tmp/cjepa_smoke
    bash run_train_CausalJEPA.sh /tmp/cjepa_smoke/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml


def _make_clip(rng, frames, size, n_objs, obj_size):
    """Bouncing colored squares -> (T, H, W, 3) uint8."""
    clip = np.zeros((frames, size, size, 3), dtype=np.uint8)
    pos = rng.integers(0, size - obj_size, size=(n_objs, 2)).astype(np.float32)
    vel = rng.uniform(-3, 3, size=(n_objs, 2)).astype(np.float32)
    col = rng.integers(80, 256, size=(n_objs, 3)).astype(np.uint8)
    for t in range(frames):
        for o in range(n_objs):
            pos[o] += vel[o]
            for ax in range(2):
                if pos[o, ax] < 0 or pos[o, ax] > size - obj_size:
                    vel[o, ax] *= -1
                    pos[o, ax] = np.clip(pos[o, ax], 0, size - obj_size)
            y, x = int(pos[o, 0]), int(pos[o, 1])
            clip[t, y:y + obj_size, x:x + obj_size] = col[o]
    return clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-valid", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--size", type=int, default=32)       # divisible by 8
    ap.add_argument("--objs", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    out = os.path.abspath(args.out)
    vid_dir = os.path.join(out, "vid")
    os.makedirs(vid_dir, exist_ok=True)

    def write_split(n: int, tag: str) -> str:
        items = []
        for i in range(n):
            name = f"{tag}_clip_{i:04d}.npy"
            clip = _make_clip(rng, args.frames, args.size, args.objs,
                              obj_size=max(2, args.size // 6))
            np.save(os.path.join(vid_dir, name), clip)
            items.append({"video_id": os.path.join("vid", name)})
        path = os.path.join(out, f"{tag}.json")
        with open(path, "w") as f:
            json.dump({"data": items}, f, indent=2)
        return path

    train_json = write_split(args.n_train, "train")
    valid_json = write_split(args.n_valid, "valid")
    test_json = write_split(args.n_test, "test")

    cfg = {
        "data": {
            "train_manifest": train_json,
            "valid_manifest": valid_json,
            "video_root": out,
            "num_frames": args.frames,
            "img_size": args.size,
        },
        "model": {
            "in_chans": 3,
            "enc_channels": [16, 32, 32, 32],   # shrink for CPU smoke
            "num_slots": 4,
            "slot_dim": 32,
            "slot_iters": 2,
            "slot_hidden": 32,
            "dec_hidden": 32,
            "history_len": 3,
            "pred_dim": 64,
            "pred_depth": 2,
            "pred_heads": 4,
            "pred_mlp_ratio": 4.0,
            "freeze_encoder": False,
        },
        "masking": {"max_masked_slots": 2},
        "objective": {"recon_weight": 1.0, "history_weight": 1.0,
                      "future_weight": 1.0},
        "optim": {
            "learning_rate": 4.0e-4, "adam_beta1": 0.9, "adam_beta2": 0.999,
            "weight_decay": 0.0, "warmup_steps": 2, "max_steps": 4,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1, "max_grad_norm": 1.0,
        },
        "train": {
            "output_dir": os.path.join(out, "ckpts"),
            "logging_steps": 1, "save_steps": 2, "eval_steps": 2,
            "save_total_limit": None, "num_workers": 0, "seed": 42,
            "init_from": None,
        },
        "eval": {"test_manifest": test_json, "video_root": out},
    }
    cfg_path = os.path.join(out, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"wrote {args.n_train} train + {args.n_valid} valid + {args.n_test} "
          f"test clips -> {vid_dir}")
    print(f"smoke config -> {cfg_path}")
    print(f"run: bash run_train_CausalJEPA.sh {cfg_path}")


if __name__ == "__main__":
    main()
