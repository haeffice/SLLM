"""V-JEPA 2.1 frozen-encoder evaluation (PyTorch 2.8).

Two protocols on the frozen 2.1 encoder (loaded via
`VJEPA21.from_checkpoint`, which already applies `eval()` +
`requires_grad=False`):

  * `--mode probe` (default): the V-JEPA attentive-probe protocol — freeze
    the encoder, train one attentive-probe query + linear classifier on a
    labelled video set, report top-1 accuracy. Reuses the SAME `config.yaml`
    (model section) so the encoder is rebuilt identically to pre-training.
  * `--mode dense`: export the dense per-token feature grid `(B, N, D)` for
    the first eval batch to `--dense-out` — the feature quality 2.1 targets.

Manifest JSON: ``{"data": [{"video_id": str, "label": int}, ...]}`` (label
optional in `dense` mode).

Usage:
    bash run_eval_VJEPA21.sh config.yaml student.pt
    bash run_eval_VJEPA21.sh config.yaml student.pt --mode dense
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_VJEPA2_DIR = os.path.join(os.path.dirname(_HERE), "VJEPA2")
for _p in (_HERE, _VJEPA2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from VJEPA2 import AttentiveProbe, VJEPA2Config  # noqa: E402
from train_vjepa2 import VJEPA2VideoDataset, collate_fn  # noqa: E402
from VJEPA21 import VJEPA21  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_vjepa21")


class _LabeledVideoDataset(VJEPA2VideoDataset):
    """`VJEPA2VideoDataset` + integer `label` passthrough (0 if absent)."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        item["label"] = int(self.samples[idx].get("label", 0))
        return item


def _collate(batch: list[dict]) -> dict:
    out = collate_fn(batch)
    out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return out


def load_encoder(config: dict, ckpt: str, device: torch.device) -> VJEPA21:
    """Build the encoder exactly as in pre-training, load frozen weights."""
    m, d = config.get("model", {}), config.get("data", {})
    cfg = VJEPA2Config.from_variant(
        m.get("variant", "vit_large"),
        img_size=int(d.get("img_size", 256)),
        num_frames=int(d.get("num_frames", 16)),
        patch_size=int(m.get("patch_size", 16)),
        tubelet_size=int(m.get("tubelet_size", 2)),
        use_rope=bool(m.get("use_rope", True)),
    )
    for k in ("embed_dim", "depth", "num_heads", "mlp_ratio"):
        if k in m:
            setattr(cfg, k, m[k])
    return VJEPA21.from_checkpoint(ckpt, config=cfg, device=device)


@torch.inference_mode()
def _extract(encoder: VJEPA21, video: torch.Tensor) -> torch.Tensor:
    return encoder.forward_features(encoder.normalize_pixels(video))


def _run_probe(encoder, train_dl, test_dl, num_classes, device, args):
    probe = AttentiveProbe(encoder.config.embed_dim, num_classes,
                           encoder.config.probe_num_heads).to(device)
    logger.info("attentive probe trainable params: %s",
                f"{sum(p.numel() for p in probe.parameters()):,}")
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        probe.train()
        tot = 0.0
        for batch in train_dl:
            feats = _extract(encoder, batch["video"].to(device))   # frozen
            loss = ce(probe(feats), batch["label"].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        logger.info("epoch %d/%d  train_loss=%.4f",
                    ep + 1, args.epochs, tot / max(1, len(train_dl)))

    probe.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in test_dl:
            y = batch["label"].to(device)
            logits = probe(_extract(encoder, batch["video"].to(device)))
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.numel()
    acc = 100.0 * correct / max(1, total)
    logger.info("==== attentive-probe top-1 accuracy: %.2f%% (%d/%d) ====",
                acc, correct, total)


def _run_dense(encoder, test_dl, device, out_path):
    batch = next(iter(test_dl))
    feats = _extract(encoder, batch["video"].to(device))            # (B, N, D)
    logger.info("dense features: shape=%s dtype=%s  (B, N=tokens, D)",
                tuple(feats.shape), feats.dtype)
    np.save(out_path, feats.cpu().numpy())
    logger.info("==== saved dense per-token features -> %s ====", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="student .pt checkpoint")
    ap.add_argument("--mode", choices=["probe", "dense"], default="probe")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dense-out", default="dense_features.npy")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    e = config.get("eval", {}) or {}
    d = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(config, args.ckpt, device)
    img_size = int(d.get("img_size", 256))
    num_frames = int(d.get("num_frames", 16))

    test_ds = _LabeledVideoDataset(
        e["test_manifest"], e["video_root"], num_frames, img_size, train=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=_collate)

    if args.mode == "dense":
        _run_dense(encoder, test_dl, device, args.dense_out)
        return

    train_ds = _LabeledVideoDataset(
        e["train_manifest"], e["video_root"], num_frames, img_size, train=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=_collate)
    _run_probe(encoder, train_dl, test_dl, int(e["num_classes"]), device, args)


if __name__ == "__main__":
    main()
