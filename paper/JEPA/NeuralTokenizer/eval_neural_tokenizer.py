"""JEPA-Neural-Tokenizer evaluation (PyTorch 2.8).

Loads the full tokenizer with `NeuralTokenizer.from_checkpoint` (which
applies `eval()` + `requires_grad=False`) and reports, on a held-out wav set:

  * **token rate** (tokens/sec) and frame rate (Hz),
  * **mixed-radix reversibility** — unpack(pack(indices)) == indices (the
    packing is lossless by construction), and
  * **reconstruction quality** — round-trip L1 + multi-resolution STFT
    distance between the input waveform and `detokenize(tokenize(wav))`.

Needs a Stage-2 checkpoint (encoder + FSQ proj + decoder). Reuses the same
`config.yaml` (model section) so the network is rebuilt identically.

Usage:
    bash run_eval_NeuralTokenizer.sh config.yaml nt_stage2_step29000.pt
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

from NeuralTokenizer import NeuralTokenizer, NeuralTokenizerConfig  # noqa: E402
from NeuralTokenizer_Trainer import MultiResolutionSTFTLoss  # noqa: E402
from train_neural_tokenizer import (  # noqa: E402
    WaveformDataset,
    _build_model_kwargs,
    _Cfg,
    collate_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_neural_tokenizer")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="nt_stage2_step*.pt")
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    cfg = _Cfg(raw)
    d, e = cfg.sec("data"), cfg.sec("eval")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nt_cfg = NeuralTokenizerConfig(**_build_model_kwargs(cfg))
    model = NeuralTokenizer.from_checkpoint(args.ckpt, config=nt_cfg, device=device)

    logger.info("frame rate = %.3f Hz  |  token rate = %.1f tok/s  "
                "(%d groups, vocab %d/group)",
                nt_cfg.sample_rate / model.encoder.hop, model.token_rate,
                model.codec.n_groups, model.codec.vocab_size)

    manifest = e.get("test_manifest", d.get("valid_manifest", d.get("train_manifest")))
    audio_root = e.get("audio_root", d.get("audio_root"))
    ds = WaveformDataset(manifest, audio_root, nt_cfg.sample_rate,
                         nt_cfg.clip_samples, train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_fn)
    stft = MultiResolutionSTFTLoss().to(device)

    l1_tot = stft_tot = n = 0.0
    rev_ok = True
    for batch in dl:
        wav = batch["audio"].to(device)
        tokens = model.tokenize(wav)                         # (B, T, n_groups)
        # reversibility: unpack(pack(idx)) must recover the per-dim indices.
        _, idx = model.fsq(model.proj_in(model.encoder(wav)))
        rev_ok = rev_ok and torch.equal(model.codec.unpack(model.codec.pack(idx)), idx)
        x_hat = model.detokenize(tokens)                     # (B, 1, L')
        L = min(x_hat.shape[-1], wav.shape[-1])
        B = wav.shape[0]
        l1_tot += torch.nn.functional.l1_loss(
            x_hat[..., :L], wav[..., :L]).item() * B
        stft_tot += stft(x_hat[..., :L], wav[..., :L]).item() * B
        n += B

    logger.info("==== NeuralTokenizer round-trip eval (%d clips) ====", int(n))
    logger.info("  mixed-radix reversibility : %s", "OK" if rev_ok else "FAILED")
    logger.info("  reconstruction L1         : %.6f", l1_tot / max(1.0, n))
    logger.info("  reconstruction MR-STFT    : %.6f", stft_tot / max(1.0, n))


if __name__ == "__main__":
    main()
