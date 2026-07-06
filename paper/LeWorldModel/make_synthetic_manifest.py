"""Generate a tiny synthetic episode dataset for CPU smoke-testing LeWM.

Writes random episode `.npz` files (obs frames + actions), train/valid
manifest JSONs, and a tiny `config.yaml` into `--out`:

    <out>/ep/ep_*.npz              obs (T, H, W, 3) uint8, actions (T, A)
    <out>/train.json , <out>/valid.json
    <out>/config.yaml              points at the above; tiny smoke settings

To make next-frame prediction non-trivial, each episode is a small bright
patch that translates by the (continuous) action each step.

Usage:
    python make_synthetic_manifest.py --out /tmp/lewm_smoke
    bash run_train_LeWorldModel.sh /tmp/lewm_smoke/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml


def _episode(rng, T, H, W) -> tuple[np.ndarray, np.ndarray]:
    obs = np.zeros((T, H, W, 3), dtype=np.uint8)
    actions = np.zeros((T, 2), dtype=np.float32)
    pos = np.array([H // 2, W // 2], dtype=np.float32)
    for t in range(T):
        a = rng.uniform(-3, 3, size=2).astype(np.float32)
        actions[t] = a / 3.0                                  # normalized action
        pos = np.clip(pos + a, 2, [H - 3, W - 3])
        y, x = int(pos[0]), int(pos[1])
        obs[t, y - 2:y + 3, x - 2:x + 3, :] = 255             # bright patch
    return obs, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-valid", type=int, default=2)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--size", type=int, default=32)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    out = os.path.abspath(args.out)
    ep_dir = os.path.join(out, "ep")
    os.makedirs(ep_dir, exist_ok=True)

    def write_split(n: int, tag: str) -> str:
        items = []
        for i in range(n):
            name = f"{tag}_ep_{i:04d}.npz"
            obs, actions = _episode(rng, args.T, args.size, args.size)
            np.savez(os.path.join(ep_dir, name), obs=obs, actions=actions)
            items.append({"episode_id": os.path.join("ep", name)})
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
            "data_root": out,
            "img_size": 32,
            "action_dim": 2,
            "seq_len": 8,
        },
        "model": {
            "patch_size": 8, "embed_dim": 96, "depth": 2, "num_heads": 6,
            "mlp_ratio": 4.0, "latent_dim": 96, "pred_depth": 2,
            "pred_num_heads": 6,
        },
        "objective": {"reg_weight": 1.0, "num_proj": 16},
        "optim": {
            "learning_rate": 5.0e-4, "adam_beta1": 0.9, "adam_beta2": 0.999,
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

    print(f"wrote {args.n_train} train + {args.n_valid} valid episodes -> {ep_dir}")
    print(f"smoke config -> {cfg_path}")
    print(f"run: bash run_train_LeWorldModel.sh {cfg_path}")


if __name__ == "__main__":
    main()
