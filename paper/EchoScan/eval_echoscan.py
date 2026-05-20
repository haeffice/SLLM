"""EchoScan evaluation — floorplan / height IoU on a test manifest.

Uses the training model class but a *separate* load/inference path:
`EchoScan.from_checkpoint` freezes the parameters and switches to eval mode,
and `EchoScan.predict` re-asserts `eval()` + `no_grad`. Reads the same
`config.yaml` used for training so the map resolutions stay identical.

    python eval_echoscan.py --config config.yaml \
        --checkpoint /path/to/ckpts/echoscan/echoscan_final.pt \
        --manifest /path/to/data/test/manifest.json [--save-viz out_dir --max-viz 8]
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

from EchoScan import EchoScan, iou                       # noqa: E402
from train_echoscan import EchoScanDataset, collate_fn   # noqa: E402


def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _save_viz(out_dir: str, idx: int, pred_fp, gt_fp, pred_h, gt_h):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  [viz] matplotlib not installed — skipping visualisation")
        return
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(gt_fp, cmap="gray"); ax[0].set_title("GT floorplan")
    ax[1].imshow(pred_fp, cmap="gray"); ax[1].set_title("Pred floorplan")
    ax[2].plot(gt_h, label="GT"); ax[2].plot(pred_h, label="Pred")
    ax[2].set_title("height profile"); ax[2].legend()
    for a in ax[:2]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"sample_{idx:04d}.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="training config.yaml")
    ap.add_argument("--checkpoint", required=True, help="EchoScan .pt checkpoint")
    ap.add_argument("--manifest", required=True, help="test manifest.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save-viz", default=None, help="dir for prediction PNGs")
    ap.add_argument("--max-viz", type=int, default=8)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    d = cfg.get("data", {})
    b = int(d.get("floorplan_size", 1024))
    h = int(d.get("height_size", 512))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EchoScan.from_checkpoint(args.checkpoint, map_location=device).to(device)
    assert not any(p.requires_grad for p in model.parameters()), \
        "inference model must have all params frozen"

    ds = EchoScanDataset(args.manifest, b, h)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn)

    fp_ious, h_ious, n_viz = [], [], 0
    for batch in loader:
        rir = batch["rir"].to(device)
        gt_fp = batch["floorplan"].to(device)              # (B, 1, b, b)
        gt_h = batch["height"].to(device)                  # (B, h)
        pred = model.predict(rir, threshold=args.threshold)

        fp_ious.append(iou(pred["floorplan"], gt_fp.squeeze(1)).cpu())
        # height: account for the flip ambiguity (PIT-style), take the better
        h_pred = pred["height"]
        h_iou = torch.maximum(iou(h_pred, gt_h),
                              iou(h_pred, torch.flip(gt_h, dims=[1])))
        h_ious.append(h_iou.cpu())

        if args.save_viz and n_viz < args.max_viz:
            for i in range(rir.shape[0]):
                if n_viz >= args.max_viz:
                    break
                _save_viz(args.save_viz, n_viz,
                          pred["floorplan"][i].cpu().numpy(),
                          gt_fp[i, 0].cpu().numpy(),
                          pred["height"][i].cpu().numpy(),
                          gt_h[i].cpu().numpy())
                n_viz += 1

    fp_iou = torch.cat(fp_ious).mean().item()
    h_iou = torch.cat(h_ious).mean().item()
    print("==== EchoScan evaluation ====")
    print(f"  samples           : {len(ds)}")
    print(f"  floorplan mean IoU : {fp_iou:.4f}")
    print(f"  height    mean IoU : {h_iou:.4f}")
    print(f"  (b={b}, h={h}, threshold={args.threshold})")
    if args.save_viz:
        print(f"  visualisations    : {n_viz} PNGs -> {args.save_viz}")


if __name__ == "__main__":
    main()
