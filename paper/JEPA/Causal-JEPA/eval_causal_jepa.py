"""Causal-JEPA latent-rollout / planning evaluation (PyTorch 2.8).

Loads the frozen model with `CausalJEPA.from_checkpoint` (which applies
`eval()` + `requires_grad=False`) and reports, on a held-out clip set, the
two quantities the paper's world model is judged on:

  * **Future slot-prediction MSE** — encode the full clip, roll the
    predictor out from the first `history_len` frames, and compare the
    predicted future slots to the encoder's actual future slots.
  * **MPC goal distance** — latent goal distance ‖Ŝ_T − S_goal‖² to the
    clip's last frame (the planning objective `a* = argmin ‖Ŝ_{t+H} − S_g‖²`).

Reuses the training model class + dataset and the SAME `config.yaml`
(model section) so the network is rebuilt identically.

Usage:
    bash run_eval_CausalJEPA.sh config.yaml causal_jepa_step2000.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from CausalJEPA import CausalJEPA, CausalJEPAConfig  # noqa: E402
from train_causal_jepa import VideoClipDataset, collate_fn  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_causal_jepa")


def load_model(config: dict, ckpt: str, device: torch.device) -> CausalJEPA:
    m, d = config.get("model", {}), config.get("data", {})
    msk, ob = config.get("masking", {}) or {}, config.get("objective", {}) or {}
    kw = dict(img_size=int(d.get("img_size", 64)),
              num_frames=int(d.get("num_frames", 16)))
    for src, keys in (
        (m, ("in_chans", "enc_channels", "num_slots", "slot_dim", "slot_iters",
             "slot_hidden", "dec_hidden", "pred_dim", "pred_depth", "pred_heads",
             "pred_mlp_ratio", "history_len")),
        (msk, ("max_masked_slots",)),
        (ob, ("recon_weight", "history_weight", "future_weight")),
    ):
        for k in keys:
            if k in src:
                v = src[k]
                kw[k] = tuple(v) if isinstance(v, list) else v
    cfg = CausalJEPAConfig(**kw)
    return CausalJEPA.from_checkpoint(ckpt, config=cfg, device=device)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="causal_jepa_step*.pt")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    e, d = config.get("eval", {}) or {}, config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.ckpt, device)
    H = model.config.history_len
    img_size = int(d.get("img_size", 64))
    num_frames = int(d.get("num_frames", 16))

    manifest = e.get("test_manifest", d.get("valid_manifest"))
    video_root = e.get("video_root", d.get("video_root"))
    ds = VideoClipDataset(manifest, video_root, num_frames, img_size, train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_fn)

    fut_mse_tot = goal_tot = n = 0.0
    for batch in dl:
        video = batch["video"].to(device)                 # (B, T, C, H, W)
        B = video.shape[0]
        slots_all = model.encode_slots(video)             # (B, T, N, d)
        pred_future = model.rollout(video[:, :H])         # (B, T-H, N, d)
        fut_mse = ((pred_future - slots_all[:, H:]) ** 2).mean().item()
        goal = model.goal_distance(video[:, :H], video[:, -1]).mean().item()
        fut_mse_tot += fut_mse * B
        goal_tot += goal * B
        n += B

    logger.info("==== Causal-JEPA rollout eval (%d clips, history_len=%d) ====",
                int(n), H)
    logger.info("  future slot-prediction MSE : %.6f", fut_mse_tot / max(1.0, n))
    logger.info("  MPC latent goal distance   : %.6f", goal_tot / max(1.0, n))


if __name__ == "__main__":
    main()
