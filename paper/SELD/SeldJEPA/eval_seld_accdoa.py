"""Stage-2 Multi-ACCDOA SELD evaluation — ER_20 / F_20 / LE_CD / LR_CD / ε_SELD.

Uses the training model class but a *separate* frozen load/inference path:
`SeldMultiACCDOA.from_checkpoint` freezes the parameters and switches to eval mode,
`predict` re-asserts `eval()` + `no_grad`. Reads the same `config_seld.yaml` used
for training so the feature/label settings stay identical.

    python eval_seld_accdoa.py --config config_seld.yaml \
        --ckpt /path/to/ckpts/accdoa/seld_accdoa_final.pt \
        [--manifest data/test.json] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from SeldMultiACCDOA import SeldMultiACCDOA  # noqa: E402
from seld_metrics import compute_seld_metrics  # noqa: E402
from train_seld_accdoa import SeldACCDOADataset, collate_fn, _feature_config  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="training config_seld.yaml")
    ap.add_argument("--ckpt", required=True, help="SeldMultiACCDOA .pt checkpoint")
    ap.add_argument("--manifest", default=None, help="test manifest (default: eval.test_manifest)")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    feat_cfg = _feature_config(cfg.get("features", {}) or {})
    manifest = args.manifest or (cfg.get("eval", {}) or {}).get("test_manifest")
    if not manifest:
        raise ValueError("no test manifest (pass --manifest or set eval.test_manifest)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SeldMultiACCDOA.from_checkpoint(args.ckpt, map_location=device, device=device)
    assert not any(p.requires_grad for p in model.parameters()), \
        "inference model must have all params frozen"

    label_fps = int(round(1000.0 / float((cfg.get("model", {}).get("head", {}) or {})
                                         .get("label_hop_ms", 100))))
    ds = SeldACCDOADataset(manifest, feat_cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    preds, targets = [], []
    for batch in loader:
        pred = model.predict(batch["feat"].to(device))
        preds.append(pred.cpu())
        targets.append(batch["target"])

    # pad time dim to a common length so we can stack across batches
    t_max = max(p.shape[1] for p in preds)
    def _pad_t(x, t):
        if x.shape[1] == t:
            return x
        pad = list(x.shape); pad[1] = t - x.shape[1]
        return torch.cat([x, torch.zeros(*pad, dtype=x.dtype)], dim=1)
    pred = torch.cat([_pad_t(p, t_max) for p in preds], dim=0)
    tgt_tmax = max(t.shape[1] for t in targets)
    target = torch.cat([_pad_t(t, tgt_tmax) for t in targets], dim=0)

    m = compute_seld_metrics(
        pred, target,
        activity_threshold=model.activity_threshold,
        unify_angle_deg=model.unify_angle_deg,
        label_fps=label_fps,
    )

    print("==== SeldMultiACCDOA evaluation ====")
    print(f"  clips              : {len(ds)}  (ref events {m['n_ref']}, pred events {m['n_pred']})")
    print(f"  ER_20              : {m['ER_20']:.4f}   (lower better)")
    print(f"  F_20               : {m['F_20'] * 100:.2f}%   (higher better)")
    print(f"  LE_CD              : {m['LE_CD']:.2f}°   (lower better)")
    print(f"  LR_CD              : {m['LR_CD'] * 100:.2f}%   (higher better)")
    print(f"  SELD error (eps)   : {m['SELD']:.4f}   (lower better)")


if __name__ == "__main__":
    main()
