"""PhaseCoder evaluation — localization accuracy + angular/distance error.

Uses the training model class but a *separate* load/inference path:
`PhaseCoder.from_checkpoint` freezes the parameters and switches to eval mode,
and `PhaseCoder.predict` re-asserts `eval()` + `no_grad`. Reads the same
`config.yaml` used for training so the class counts / ranges stay identical.

    python eval_phasecoder.py --config config.yaml \
        --checkpoint /path/to/ckpts/phasecoder/phasecoder_final.pt \
        --manifest data/test/manifest.json
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

from PhaseCoder import PhaseCoder, angular_error_deg     # noqa: E402
from train_phasecoder import PhaseCoderDataset, collate_fn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="training config.yaml")
    ap.add_argument("--checkpoint", required=True, help="PhaseCoder .pt checkpoint")
    ap.add_argument("--manifest", required=True, help="test manifest.json")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    m = cfg.get("model", {})
    model_kwargs = {k: m[k] for k in (
        "embed_dim", "n_blocks", "n_heads", "ffn_dim", "n_fft", "hop",
        "n_azimuth", "n_elevation", "n_distance", "dist_min", "dist_max") if k in m}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PhaseCoder.from_checkpoint(args.checkpoint, map_location=device,
                                       **model_kwargs).to(device)
    assert not any(p.requires_grad for p in model.parameters()), \
        "inference model must have all params frozen"

    ds = PhaseCoderDataset(args.manifest)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    az_correct = el_correct = di_correct = n = 0
    az_errs, el_errs, di_errs = [], [], []
    for batch in loader:
        pred = model.predict(batch["audio"].to(device), batch["mic_coords"].to(device),
                             batch["channel_mask"].to(device))
        az_correct += int((pred["azimuth_bin"].cpu() == batch["azimuth"]).sum())
        el_correct += int((pred["elevation_bin"].cpu() == batch["elevation"]).sum())
        di_correct += int((pred["distance_bin"].cpu() == batch["distance"]).sum())
        n += batch["azimuth"].shape[0]

        az_errs.append(angular_error_deg(pred["azimuth_deg"].cpu(), batch["azimuth_deg"], circular=True))
        el_errs.append(angular_error_deg(pred["elevation_deg"].cpu(), batch["elevation_deg"], circular=False))
        # distance MAE over samples with a real source
        valid = (pred["distance_m"].cpu() >= 0) & (batch["distance_m"] >= 0)
        di_errs.append((pred["distance_m"].cpu()[valid] - batch["distance_m"][valid]).abs())

    az_mae = torch.cat(az_errs).mean().item() if sum(len(e) for e in az_errs) else float("nan")
    el_mae = torch.cat(el_errs).mean().item() if sum(len(e) for e in el_errs) else float("nan")
    di_mae = torch.cat(di_errs).mean().item() if sum(len(e) for e in di_errs) else float("nan")

    print("==== PhaseCoder evaluation ====")
    print(f"  samples            : {n}")
    print(f"  azimuth   top-1 acc : {az_correct / max(n,1):.4f}   MAE: {az_mae:.2f}°")
    print(f"  elevation top-1 acc : {el_correct / max(n,1):.4f}   MAE: {el_mae:.2f}°")
    print(f"  distance  top-1 acc : {di_correct / max(n,1):.4f}   MAE: {di_mae:.3f} m")


if __name__ == "__main__":
    main()
