"""Synthetic 2-channel SELD dataset generator.

Renders short 2-microphone clips by placing 1-`max_overlap` sound sources at random
azimuths in a free field. Each source is class-specific band-limited noise with a
random active span; the 2-channel signal is produced by a far-field mic-pair model
(fractional inter-channel delay = ITD + mild level difference = ILD). Writes:

    <out>/wav/clip_XXXX.wav        2-channel float32 wav (fs = 24 kHz)
    <out>/label/clip_XXXX.npy      Multi-ACCDOA target (T_label, C, 3, 2) azimuth (x,y)
    <out>/train.json / valid.json / test.json   manifests ({"meta":..., "data":[...]})
    <out>/config_pretrain.yaml     shrunken Stage-1 CPU smoke config
    <out>/config_seld.yaml         shrunken Stage-2 CPU smoke config

The same manifests serve both stages: Stage-1 reads only `audio_id`, Stage-2 reads
`audio_id` + `label_id`. `pyroomacoustics` is imported lazily and is NOT required
(the default free-field renderer needs only numpy/scipy/soundfile).

    python make_seld_dataset.py --out /tmp/seld_smoke --n 16 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import soundfile as sf
import yaml

# --- fixed generation constants (match features.py defaults) ---
FS = 24_000
LABEL_FPS = 10                  # 10 Hz label rate (100 ms hop)
MIC_SPACING = 0.15             # m, inter-mic distance of the pair
SPEED_OF_SOUND = 343.0         # m/s
N_TRACKS = 3                   # max simultaneous overlap


def _bandlimited_noise(n: int, lo_hz: float, hi_hz: float, rng: np.random.Generator) -> np.ndarray:
    """White noise band-limited to [lo, hi] Hz via an rFFT mask (no filter design)."""
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    X[(freqs < lo_hz) | (freqs > hi_hz)] = 0.0
    y = np.fft.irfft(X, n=n)
    return y / (np.abs(y).max() + 1e-9)


def _class_band(c: int, n_classes: int) -> tuple[float, float]:
    """Distinct band per class so the encoder has spectral cues to localize."""
    edges = np.linspace(200.0, 10_000.0, n_classes + 1)
    return float(edges[c]), float(edges[c + 1])


def _fractional_delay(sig: np.ndarray, delay_samples: float) -> np.ndarray:
    """Delay a signal by a (possibly fractional) number of samples (linear interp)."""
    n = len(sig)
    idx = np.arange(n) - delay_samples
    return np.interp(idx, np.arange(n), sig, left=0.0, right=0.0)


def _render_source(sig: np.ndarray, azimuth_rad: float) -> np.ndarray:
    """Far-field 2-mic render → (2, T). ITD via opposite half-delays, mild ILD."""
    itd = MIC_SPACING * np.sin(azimuth_rad) / SPEED_OF_SOUND        # seconds
    d = 0.5 * itd * FS                                             # half-delay (samples)
    ild = 0.2 * np.sin(azimuth_rad)                                # mild level difference
    ch0 = _fractional_delay(sig, +d) * (1.0 - ild)                # "left"
    ch1 = _fractional_delay(sig, -d) * (1.0 + ild)                # "right"
    return np.stack([ch0, ch1], axis=0)


def _make_clip(rng: np.random.Generator, clip_seconds: float, n_classes: int,
               max_overlap: int) -> tuple[np.ndarray, np.ndarray]:
    """One clip → (wav (2, T), label (T_label, C, 3, 2))."""
    n = int(round(clip_seconds * FS))
    t_label = int(round(clip_seconds * LABEL_FPS))
    mix = np.zeros((2, n), dtype=np.float32)
    # per (frame, class) list of azimuth (x, y) unit vectors
    buckets = [[[] for _ in range(n_classes)] for _ in range(t_label)]

    n_src = int(rng.integers(1, max_overlap + 1))
    for _ in range(n_src):
        c = int(rng.integers(0, n_classes))
        az = float(rng.uniform(-np.pi, np.pi))
        lo, hi = _class_band(c, n_classes)
        sig = _bandlimited_noise(n, lo, hi, rng).astype(np.float32)
        # random active span
        f0 = int(rng.integers(0, max(1, t_label - 1)))
        f1 = int(rng.integers(f0 + 1, t_label + 1))
        env = np.zeros(n, dtype=np.float32)
        s0, s1 = int(f0 / LABEL_FPS * FS), int(f1 / LABEL_FPS * FS)
        env[s0:s1] = 1.0
        mix += _render_source(sig * env, az).astype(np.float32)
        xy = (float(np.cos(az)), float(np.sin(az)))
        for f in range(f0, f1):
            buckets[f][c].append(xy)

    mix /= (np.abs(mix).max() + 1e-9)
    mix *= 0.9

    label = np.zeros((t_label, n_classes, N_TRACKS, 2), dtype=np.float32)
    for f in range(t_label):
        for c in range(n_classes):
            for s, xy in enumerate(buckets[f][c][:N_TRACKS]):       # active-first, <=3
                label[f, c, s] = xy
    return mix, label


def _write_yaml(path: str, obj: dict):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, default_flow_style=False)


def _smoke_configs(out: str, n_classes: int):
    """Tiny CPU-smoke configs for both stages, pointing at the generated data."""
    features = dict(sample_rate=FS, n_fft=512, win_length=512, hop_length=240, n_mels=64)
    pretrain = {
        "data": {"train_manifest": os.path.join(out, "train.json"),
                 "valid_manifest": os.path.join(out, "valid.json")},
        "features": features,
        "encoder": {"encoder_dim": 64, "num_layers": 2, "num_heads": 4, "ffn_expansion": 2,
                    "conv_kernel": 7, "chunk_frames": 5, "frontend_hidden": 16,
                    "dropout": 0.1, "pool": "last"},
        "views": {"num_global": 2, "num_local": 2, "global_seconds": 1.2, "local_seconds": 0.4},
        "masking": {"n_time_masks": 2, "time_width": 8, "n_freq_masks": 2,
                    "freq_width": 6, "spectral_only": False},
        "objective": {"sigreg_coeff": 1.0, "num_slices": 64, "num_points": 17, "pred_hidden": 128},
        "optim": {"learning_rate": 5.0e-4, "weight_decay": 5.0e-2, "warmup_steps": 1,
                  "max_steps": 4, "per_device_train_batch_size": 2, "max_grad_norm": 1.0,
                  "bf16": False},
        "train": {"output_dir": os.path.join(out, "ckpts", "jepa"), "logging_steps": 1,
                  "save_steps": 2, "eval_steps": 2, "num_workers": 0, "seed": 42},
    }
    seld = {
        "data": {"train_manifest": os.path.join(out, "train.json"),
                 "valid_manifest": os.path.join(out, "valid.json")},
        "model": {"encoder_ckpt": os.path.join(out, "ckpts", "jepa", "seld_jepa_final.pt"),
                  "freeze_encoder": True, "unfreeze_at_step": 0,
                  "head": {"num_tracks": N_TRACKS, "num_classes": n_classes, "pool_factor": 5,
                           "activity_threshold": 0.5, "unify_angle_deg": 30.0,
                           "label_hop_ms": 100}},
        "features": features,
        "loss": {"swap_prob": 0.0},
        "optim": {"num_train_epochs": 2, "max_steps": 4, "learning_rate": 1.0e-3,
                  "weight_decay": 1.0e-4, "warmup_steps": 1, "per_device_train_batch_size": 2,
                  "max_grad_norm": 1.0},
        "train": {"output_dir": os.path.join(out, "ckpts", "accdoa"), "logging_steps": 1,
                  "save_steps": 2, "eval_steps": 2, "num_workers": 0, "seed": 42},
        "eval": {"test_manifest": os.path.join(out, "test.json")},
    }
    _write_yaml(os.path.join(out, "config_pretrain.yaml"), pretrain)
    _write_yaml(os.path.join(out, "config_seld.yaml"), seld)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, default=64, help="number of clips")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip-seconds", type=float, default=1.6)
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--max-overlap", type=int, default=N_TRACKS)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.out, "wav"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "label"), exist_ok=True)

    records = []
    for i in range(args.n):
        wav, label = _make_clip(rng, args.clip_seconds, args.n_classes, args.max_overlap)
        wav_id = os.path.join("wav", f"clip_{i:04d}.wav")
        lab_id = os.path.join("label", f"clip_{i:04d}.npy")
        sf.write(os.path.join(args.out, wav_id), wav.T, FS, subtype="FLOAT")
        np.save(os.path.join(args.out, lab_id), label)
        records.append({"audio_id": wav_id, "label_id": lab_id})

    n_test = max(1, args.n // 5)
    n_valid = max(1, args.n // 5)
    splits = {"test": records[:n_test],
              "valid": records[n_test:n_test + n_valid],
              "train": records[n_test + n_valid:] or records}
    meta = {"fs": FS, "n_channels": 2, "n_classes": args.n_classes, "n_tracks": N_TRACKS,
            "label_hop_ms": int(1000 / LABEL_FPS), "clip_seconds": args.clip_seconds}
    for name, data in splits.items():
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump({"meta": meta, "data": data}, f, indent=2)

    _smoke_configs(args.out, args.n_classes)

    print(f"[make_seld_dataset] wrote {args.n} clips to {args.out}")
    print(f"  splits: train={len(splits['train'])} valid={len(splits['valid'])} test={len(splits['test'])}")
    print(f"  meta: {meta}")
    print(f"  smoke configs: {args.out}/config_pretrain.yaml , {args.out}/config_seld.yaml")
    print("  smoke run:")
    print(f"    bash run_train_SeldJEPA.sh   {args.out}/config_pretrain.yaml")
    print(f"    bash run_train_SeldACCDOA.sh {args.out}/config_seld.yaml")
    print(f"    bash run_eval_SeldACCDOA.sh  {args.out}/config_seld.yaml {args.out}/ckpts/accdoa/seld_accdoa_final.pt")


if __name__ == "__main__":
    main()
