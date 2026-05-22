"""Generate a tiny synthetic point-cloud dataset for CPU smoke-testing.

Writes random `.npy` point clouds, train/valid manifest JSONs, and a tiny
`config.yaml` into `--out`:

    <out>/pc/obj_*.npy              (P, 3) float32
    <out>/train.json , <out>/valid.json
    <out>/config.yaml               points at the above; tiny smoke settings

Usage:
    python make_synthetic_manifest.py --out /tmp/pj_smoke
    bash run_train_PointJEPA.sh /tmp/pj_smoke/config.yaml
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
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-valid", type=int, default=2)
    ap.add_argument("--points", type=int, default=512)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    out = os.path.abspath(args.out)
    pc_dir = os.path.join(out, "pc")
    os.makedirs(pc_dir, exist_ok=True)

    def write_split(n: int, tag: str) -> str:
        items = []
        for i in range(n):
            name = f"{tag}_obj_{i:04d}.npy"
            # random points on a perturbed sphere (something to encode)
            v = rng.normal(size=(args.points, 3)).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
            v *= 1.0 + 0.1 * rng.normal(size=(args.points, 1)).astype(np.float32)
            np.save(os.path.join(pc_dir, name), v)
            items.append({"points_id": os.path.join("pc", name)})
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
            "points_root": out,
            "num_points": 256,                 # small for CPU smoke
        },
        "model": {
            "num_groups": 16, "group_size": 16,
            "embed_dim": 96, "depth": 2, "num_heads": 6, "mlp_ratio": 4.0,
            "predictor_dim": 48, "predictor_depth": 2,
        },
        "masking": {
            "num_target_blocks": 4,
            "target_ratio_min": 0.15, "target_ratio_max": 0.20,
            "context_ratio_min": 0.40, "context_ratio_max": 0.75,
            "smooth_l1_beta": 2.0,
        },
        "ema": {"ema_decay": 0.995, "ema_end_decay": 1.0,
                "ema_anneal_end_step": 50},
        "optim": {
            "learning_rate": 1.0e-3, "adam_beta1": 0.9, "adam_beta2": 0.999,
            "weight_decay": 0.05, "warmup_steps": 2, "max_steps": 4,
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

    print(f"wrote {args.n_train} train + {args.n_valid} valid clouds -> {pc_dir}")
    print(f"smoke config -> {cfg_path}")
    print(f"run: bash run_train_PointJEPA.sh {cfg_path}")


if __name__ == "__main__":
    main()
