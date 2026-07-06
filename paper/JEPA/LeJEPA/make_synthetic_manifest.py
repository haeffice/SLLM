"""Generate a tiny synthetic image dataset for CPU smoke-testing LeJEPA.

Writes random `.npy` images (with simple per-image structure so multi-crop
views share content), train/valid manifests, and a tiny `config.yaml` into
`--out`:

    <out>/img/img_*.npy            (H, W, 3) uint8
    <out>/train.json , <out>/valid.json
    <out>/config.yaml              tiny smoke settings

Usage:
    python make_synthetic_manifest.py --out /tmp/lejepa_smoke
    bash run_train_LeJEPA.sh /tmp/lejepa_smoke/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml


def _image(rng, S) -> np.ndarray:
    # smooth low-frequency colour blobs so crops of one image are related
    base = rng.uniform(0, 1, size=(4, 4, 3)).astype(np.float32)
    img = np.kron(base, np.ones((S // 4, S // 4, 1), dtype=np.float32))
    img = img[:S, :S]
    img = img + 0.1 * rng.standard_normal((S, S, 3)).astype(np.float32)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-valid", type=int, default=4)
    ap.add_argument("--size", type=int, default=64)        # source image side
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    out = os.path.abspath(args.out)
    img_dir = os.path.join(out, "img")
    os.makedirs(img_dir, exist_ok=True)

    def write_split(n: int, tag: str) -> str:
        items = []
        for i in range(n):
            name = f"{tag}_img_{i:04d}.npy"
            np.save(os.path.join(img_dir, name), _image(rng, args.size))
            items.append({"image_id": os.path.join("img", name)})
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
            "image_root": out,
        },
        "model": {                                 # tiny ViT for CPU smoke
            "in_chans": 3, "img_size": 32, "patch_size": 8,
            "embed_dim": 48, "depth": 2, "num_heads": 3,
            "mlp_ratio": 2.0, "pred_hidden": 64,
        },
        "views": {                                 # small crops for CPU smoke
            "num_global": 2, "num_local": 4,
            "global_size": 32, "local_size": 16,
            "global_scale": [0.3, 1.0], "local_scale": [0.05, 0.3],
        },
        "objective": {
            "sigreg_coeff": 1.0, "num_slices": 64,
            "num_points": 17, "probe_last_layers": 2,
        },
        "optim": {
            "learning_rate": 5.0e-4, "adam_beta1": 0.9, "adam_beta2": 0.999,
            "weight_decay": 5.0e-2, "warmup_steps": 2, "max_steps": 4,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 1, "max_grad_norm": 1.0,
            "bf16": False,
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

    print(f"wrote {args.n_train} train + {args.n_valid} valid images -> {img_dir}")
    print(f"smoke config -> {cfg_path}")
    print(f"run: bash run_train_LeJEPA.sh {cfg_path}")


if __name__ == "__main__":
    main()
