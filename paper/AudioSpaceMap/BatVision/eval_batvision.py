"""BatVision evaluation — depth-map error metrics on a test split.

Uses the training model class but a *separate* load/inference path:
`BatVision.from_checkpoint` freezes the parameters and switches to eval mode,
and `BatVision.predict` re-asserts `eval()` + `no_grad`. Reads the same
`config.yaml` used for training so the resolutions / max_depth stay identical.

    python eval_batvision.py --config config.yaml \
        --checkpoint /path/to/ckpts/batvision/batvision_final.pt \
        --csv test.csv [--save-viz out_dir --max-viz 8]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import yaml  # noqa: E402

from BatVision import BatVision, compute_errors          # noqa: E402
from train_batvision import BatVisionDataset, collate_fn  # noqa: E402


def _save_viz(out_dir: str, idx: int, spec, pred_m, gt_m, max_depth: float):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  [viz] matplotlib not installed — skipping visualisation")
        return
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(spec[0], aspect="auto", origin="lower"); ax[0].set_title("input spec (L)")
    im1 = ax[1].imshow(gt_m, cmap="viridis", vmin=0, vmax=max_depth); ax[1].set_title("GT depth (m)")
    im2 = ax[2].imshow(pred_m, cmap="viridis", vmin=0, vmax=max_depth); ax[2].set_title("Pred depth (m)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046); fig.colorbar(im2, ax=ax[2], fraction=0.046)
    ax[0].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"sample_{idx:04d}.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="training config.yaml")
    ap.add_argument("--checkpoint", required=True, help="BatVision .pt checkpoint")
    ap.add_argument("--csv", default=None, help="test CSV name (default: data.test_csv)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-viz", default=None, help="dir for prediction PNGs")
    ap.add_argument("--max-viz", type=int, default=8)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    d = cfg.get("data", {})
    max_depth = float(d.get("max_depth", 12.0))
    depth_norm = bool(d.get("depth_norm", True))
    csv_name = args.csv or d.get("test_csv")
    if not csv_name:
        raise ValueError("no test CSV given (--csv or data.test_csv)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BatVision.from_checkpoint(args.checkpoint, map_location=device).to(device)
    assert not any(p.requires_grad for p in model.parameters()), \
        "inference model must have all params frozen"

    ds = BatVisionDataset(d, csv_name)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    keys = ("abs_rel", "rmse", "delta1", "delta2", "delta3", "log10", "mae")
    sums = {k: 0.0 for k in keys}
    n, n_viz = 0, 0
    for batch in loader:
        spec = batch["spectrogram"].to(device)
        gt_norm = batch["depth"].to(device)                  # (B, 1, S, S), normalised
        pred = model.predict(spec)
        pred_m = pred["depth_m"]                             # metres
        gt_m = gt_norm * max_depth if depth_norm else gt_norm

        for i in range(spec.shape[0]):
            e = compute_errors(gt_m[i], pred_m[i])
            for k in keys:
                sums[k] += e[k]
            n += 1
            if args.save_viz and n_viz < args.max_viz:
                _save_viz(args.save_viz, n_viz, spec[i].cpu().numpy(),
                          pred_m[i, 0].cpu().numpy(), gt_m[i, 0].cpu().numpy(), max_depth)
                n_viz += 1

    means = {k: sums[k] / max(n, 1) for k in keys}
    print("==== BatVision evaluation ====")
    print(f"  samples   : {n}")
    print(f"  RMSE (m)  : {means['rmse']:.4f}")
    print(f"  MAE  (m)  : {means['mae']:.4f}")
    print(f"  ABS_REL   : {means['abs_rel']:.4f}")
    print(f"  LOG10     : {means['log10']:.4f}")
    print(f"  delta<1.25 : {means['delta1']:.4f}  "
          f"delta<1.25^2: {means['delta2']:.4f}  delta<1.25^3: {means['delta3']:.4f}")
    print(f"  (max_depth={max_depth} m, depth_norm={depth_norm})")
    if args.save_viz:
        print(f"  visualisations : {n_viz} PNGs -> {args.save_viz}")


if __name__ == "__main__":
    main()
