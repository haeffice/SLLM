"""Generate a tiny synthetic view-pair dataset for CPU smoke-testing.

Writes a synthetic NL->code-ish JSONL train/valid split and a tiny
`config.yaml` (model_name "__tiny__" -> offline random LlamaForCausalLM +
byte tokenizer, no network / no HF download) into `--out`:

    <out>/train.jsonl , <out>/valid.jsonl   {"text": ..., "code": ...}
    <out>/config.yaml                        tiny smoke settings

Usage:
    python make_synthetic_manifest.py --out /tmp/llmjepa_smoke
    bash run_train_LLMJEPA.sh /tmp/llmjepa_smoke/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random

import yaml

_NOUNS = ["users", "orders", "books", "movies", "songs", "cities"]
_COLS = ["name", "id", "price", "year", "rating", "count"]


def _pair(rng: random.Random) -> dict:
    n = rng.choice(_NOUNS)
    c = rng.choice(_COLS)
    text = f"List the {c} of all {n} sorted by {c}."
    code = f"SELECT {c} FROM {n} ORDER BY {c};"
    return {"text": text, "code": code}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-valid", type=int, default=4)
    args = ap.parse_args()

    rng = random.Random(0)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    def write(path: str, n: int):
        with open(path, "w") as f:
            for _ in range(n):
                f.write(json.dumps(_pair(rng)) + "\n")

    train = os.path.join(out, "train.jsonl")
    valid = os.path.join(out, "valid.jsonl")
    write(train, args.n_train)
    write(valid, args.n_valid)

    cfg = {
        "data": {"train_file": train, "valid_file": valid},
        "model": {
            "model_name": "__tiny__",
            "num_predictors": 1,
            "front_pred": False,
            "jepa_objective": "cos",
            "lbd": 0.1, "gamma": 1.0, "infonce_temp": 0.07,
            "max_length": 64, "torch_dtype": "float32",
            "use_lora": False,
            "tiny_hidden": 64, "tiny_layers": 2,
            "tiny_heads": 4, "tiny_inter": 128,
        },
        "optim": {
            "learning_rate": 1.0e-3, "adam_beta1": 0.9, "adam_beta2": 0.999,
            "weight_decay": 0.0, "warmup_steps": 1, "max_steps": 4,
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

    print(f"wrote {args.n_train} train + {args.n_valid} valid pairs -> {out}")
    print(f"smoke config -> {cfg_path}")
    print(f"run: bash run_train_LLMJEPA.sh {cfg_path}")


if __name__ == "__main__":
    main()
