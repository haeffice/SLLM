"""Synthetic BatVision-V1-format dataset generator (CPU smoke test).

The *real* BatVision dataset is a multi-GB public download (see README) — this
script does **not** replace it. It writes a tiny synthetic dataset in the exact
BatVision **V1** on-disk layout so the full train/eval pipeline can be smoke
tested without the download. One root holds every split's CSV and shared media
(exactly like the real ``BatvisionV1/{train,val,test}.csv``):

    <out>/
      audio/{idx}_left.npy   binaural left  waveform, float32 (T,)
      audio/{idx}_right.npy  binaural right waveform, float32 (T,)
      depth/{idx}.npy        depth map in MILLIMETRES, float32 (H, W)
      train.csv | val.csv | test.csv   columns:
          'depth path', 'audio path left', 'audio path right'   (paths relative to <out>)

The synthetic signal is loosely physical: a random fronto-parallel depth blob
sets an echo delay (round-trip time = 2·depth/c), so the network has a real
audio→depth cue to fit. It is a plumbing check, not a benchmark — train on the
real dataset for meaningful numbers.

    python make_batvision_dataset.py --out data/smoke --splits train:8,val:4,test:4 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

C_SOUND = 343.0          # m/s
SR = 44100               # Hz (BatVision sample rate)


def _make_depth(h: int, w: int, rng: np.random.Generator, max_depth: float) -> np.ndarray:
    """A smooth random depth field in metres: a couple of Gaussian blobs over a
    sloped background, clipped to (0.3, max_depth]. Returns (H, W) float32."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ys /= max(h - 1, 1)
    xs /= max(w - 1, 1)
    depth = 0.4 * max_depth * (0.5 + 0.5 * xs)            # gentle horizontal ramp
    for _ in range(rng.integers(1, 4)):
        cy, cx = rng.uniform(0, 1), rng.uniform(0, 1)
        sig = rng.uniform(0.1, 0.35)
        amp = rng.uniform(-0.3, 0.3) * max_depth
        depth += amp * np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sig ** 2)))
    depth = np.clip(depth, 0.3, max_depth)
    return depth.astype(np.float32)


def _make_echo(depth_m: np.ndarray, t_samples: int, rng: np.random.Generator) -> tuple:
    """Binaural chirp echoes whose dominant delay tracks the scene's mean depth.

    Round-trip delay = 2·depth/c; left/right get a tiny azimuth-dependent offset
    so the two channels differ. Returns (left, right) float32 (t_samples,)."""
    mean_depth = float(depth_m.mean())
    delay = 2.0 * mean_depth / C_SOUND                    # seconds, round trip
    t = np.arange(t_samples) / SR
    # emitted chirp: 1 kHz→8 kHz over 3 ms
    chirp_len = int(0.003 * SR)
    tc = np.arange(chirp_len) / SR
    chirp = np.sin(2 * np.pi * (1000 + (7000 / 0.003) * tc / 2) * tc).astype(np.float32)

    def render(extra_delay: float) -> np.ndarray:
        sig = rng.normal(0, 0.01, t_samples).astype(np.float32)   # background noise
        onset = int((delay + extra_delay) * SR)
        if 0 <= onset < t_samples - chirp_len:
            atten = 1.0 / (1.0 + mean_depth)              # farther → quieter
            sig[onset:onset + chirp_len] += atten * chirp
        return sig

    azimuth_off = rng.uniform(-1e-4, 1e-4)                # ~±0.1 ms inter-aural
    return render(0.0), render(azimuth_off)


def _parse_splits(spec: str) -> list:
    out = []
    for part in spec.split(","):
        name, _, n = part.partition(":")
        out.append((name.strip(), int(n)))
    return out


def main():
    ap = argparse.ArgumentParser(description="Synthetic BatVision-V1-format smoke dataset")
    ap.add_argument("--out", required=True, help="dataset root (holds all split CSVs + media)")
    ap.add_argument("--splits", default="train:8,val:4,test:4",
                    help="comma list of split:count, e.g. 'train:8,val:4,test:4'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=64, help="depth map H")
    ap.add_argument("--width", type=int, default=64, help="depth map W")
    ap.add_argument("--audio-ms", type=float, default=72.5, help="audio length in ms (V1≈72.5)")
    ap.add_argument("--max-depth", type=float, default=12.0, help="max depth in m (V1=12)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.out, "audio"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "depth"), exist_ok=True)
    t_samples = int(args.audio_ms * 1e-3 * SR)

    idx = 0
    total = 0
    for split, count in _parse_splits(args.splits):
        rows = []
        for _ in range(count):
            depth_m = _make_depth(args.height, args.width, rng, args.max_depth)
            left, right = _make_echo(depth_m, t_samples, rng)

            d_rel = os.path.join("depth", f"{idx}.npy")
            l_rel = os.path.join("audio", f"{idx}_left.npy")
            r_rel = os.path.join("audio", f"{idx}_right.npy")
            np.save(os.path.join(args.out, d_rel), (depth_m * 1000.0).astype(np.float32))  # mm
            np.save(os.path.join(args.out, l_rel), left)
            np.save(os.path.join(args.out, r_rel), right)
            rows.append({"depth path": d_rel, "audio path left": l_rel, "audio path right": r_rel})
            idx += 1

        csv_path = os.path.join(args.out, f"{split}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["depth path", "audio path left", "audio path right"])
            w.writeheader()
            w.writerows(rows)
        print(f"[make_batvision_dataset] {split}: {count} samples -> {csv_path}")
        total += count

    print(f"  root         : {args.out}  (shared audio/ + depth/)")
    print(f"  total        : {total} samples")
    print(f"  audio length : {t_samples} samples ({args.audio_ms} ms @ {SR} Hz)")
    print(f"  depth map    : {args.height}x{args.width}, max_depth={args.max_depth} m (stored in mm)")


if __name__ == "__main__":
    main()
