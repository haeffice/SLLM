"""V-JEPA 2 frozen-encoder attentive-probe evaluation (PyTorch 2.8).

The V-JEPA 2 downstream protocol: freeze the pre-trained encoder and train
only a small attentive probe (one learnable query cross-attending over the
frozen token features + a linear classifier) on a labelled video dataset,
then report top-1 accuracy.

Reuses the training-time model class (`VJEPA2` / `AttentiveProbe`) and the
SAME `config.yaml` (model section) so the encoder is built identically to
pre-training. The encoder is loaded with `VJEPA2.from_checkpoint(...)`,
which already applies `model.eval()` and `requires_grad=False`.

Manifest JSON: ``{"data": [{"video_id": str, "label": int}, ...]}``.

Usage:
    bash run_eval_VJEPA2.sh config.yaml student.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from VJEPA2 import AttentiveProbe, VJEPA2, VJEPA2Config  # noqa: E402
from train_vjepa2 import VJEPA2VideoDataset, collate_fn  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_vjepa2")


class _LabeledVideoDataset(VJEPA2VideoDataset):
    """`VJEPA2VideoDataset` + integer `label` passthrough."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        item["label"] = int(self.samples[idx]["label"])
        return item


def _collate(batch: list[dict]) -> dict:
    out = collate_fn(batch)
    out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return out


def load_encoder(config: dict, ckpt: str,
                 device: torch.device) -> VJEPA2:
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
    model = VJEPA2.from_checkpoint(ckpt, config=cfg, device=device)
    return model


@torch.inference_mode()
def _extract(encoder: VJEPA2, video: torch.Tensor) -> torch.Tensor:
    return encoder.forward_features(encoder.normalize_pixels(video))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="student .pt checkpoint")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    e = config.get("eval", {}) or {}
    d = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(config, args.ckpt, device)
    num_classes = int(e["num_classes"])
    img_size = int(d.get("img_size", 256))
    num_frames = int(d.get("num_frames", 16))

    train_ds = _LabeledVideoDataset(
        e["train_manifest"], e["video_root"], num_frames, img_size, train=True)
    test_ds = _LabeledVideoDataset(
        e["test_manifest"], e["video_root"], num_frames, img_size, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=_collate)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=_collate)

    probe = AttentiveProbe(encoder.config.embed_dim, num_classes,
                           encoder.config.probe_num_heads).to(device)
    n_probe = sum(p.numel() for p in probe.parameters())
    logger.info("attentive probe trainable params: %s", f"{n_probe:,}")

    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        probe.train()
        tot = 0.0
        for batch in train_dl:
            video = batch["video"].to(device)
            y = batch["label"].to(device)
            feats = _extract(encoder, video)              # frozen, no grad
            logits = probe(feats)
            loss = ce(logits, y)
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
            video = batch["video"].to(device)
            y = batch["label"].to(device)
            logits = probe(_extract(encoder, video))
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.numel()
    acc = 100.0 * correct / max(1, total)
    logger.info("==== attentive-probe top-1 accuracy: %.2f%% (%d/%d) ====",
                acc, correct, total)


if __name__ == "__main__":
    main()
