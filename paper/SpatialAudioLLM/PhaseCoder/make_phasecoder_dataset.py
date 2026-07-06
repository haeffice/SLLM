"""Synthetic spatial-audio dataset generator for PhaseCoder (CPU smoke test).

The paper trains on ~1.5 M simulated room impulse responses (image-source
method) convolving LibriSpeech speech + Freesound distractors — a large
pipeline (see README for how to obtain the real corpora). This script does
**not** replace it; it produces a small, self-contained dataset with the cues
PhaseCoder actually consumes — **inter-microphone phase/time differences** (for
azimuth/elevation) and **received level** (for distance) — so the whole
train/eval pipeline runs without any download.

Crucially it is **geometry-agnostic**: every sample draws a *random* array
(3–8 mics, 7–18 cm aperture, random 3-D positions), so the saved mic
coordinates differ per sample — exactly the setting PhaseCoder targets.

Two simulators:
  * ``free-field`` (default, numpy-only): point source, per-mic fractional
    delay via FFT phase shift + 1/r amplitude. Fast, portable. Distance cue is
    only the 1/r level (no reverberation), so distance is the weakest task.
  * ``rir`` (``--sim rir``): pyroomacoustics image-source RIRs in a random
    shoebox room — adds reverberation, giving a realistic direct-to-reverberant
    distance cue. Closer to the paper; slower.

Each sample → ``{idx}.npz`` (audio (C,T) float32, mic_coords (C,3) float32,
azimuth/elevation/distance floats + binned int labels, no_source bool) indexed
by ``manifest.json``.

    python make_phasecoder_dataset.py --out data/train --n 2000 --seed 0
    python make_phasecoder_dataset.py --out data/val   --n 200  --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from PhaseCoder import (  # noqa: E402
    CLIP_SAMPLES, SAMPLE_RATE, N_AZIMUTH, N_ELEVATION, N_DISTANCE,
    DIST_MIN, DIST_MAX, azimuth_bin_centers, elevation_bin_centers,
    distance_bin_centers, value_to_bin,
)

C_SOUND = 343.0          # m/s


# -----------------------------------------------------------------------------
# Source signals
# -----------------------------------------------------------------------------

def _speech_like(n: int, rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A crude voiced-speech-like signal: harmonic stack at a random F0 with a
    couple of formant resonances and a syllabic amplitude envelope."""
    t = np.arange(n) / sr
    f0 = rng.uniform(90, 220)
    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, 26):
        if k * f0 > sr / 2:
            break
        sig += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
    # two simple resonances (formants) via 2nd-order IIR colouring of noise mix
    sig += 0.3 * rng.standard_normal(n)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 6) * t + rng.uniform(0, 6))
    sig *= np.clip(env, 0, None)
    rms = np.sqrt(np.mean(sig ** 2)) + 1e-9
    return (sig / rms).astype(np.float64)


def _load_speech(path: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Load a random n-sample window from a real audio file (needs soundfile)."""
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if len(wav) < n:
        wav = np.pad(wav, (0, n - len(wav)))
    start = rng.integers(0, max(1, len(wav) - n))
    seg = wav[start:start + n].astype(np.float64)
    rms = np.sqrt(np.mean(seg ** 2)) + 1e-9
    return seg / rms


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def _random_array(rng: np.random.Generator, min_mics: int, max_mics: int) -> np.ndarray:
    """Random mic array centred at the origin: C∈[min,max] mics within a ball of
    radius = aperture/2 (aperture 7–18 cm). Returns (C, 3) metres."""
    c = int(rng.integers(min_mics, max_mics + 1))
    radius = rng.uniform(0.07, 0.18) / 2.0
    pts = rng.standard_normal((c, 3))
    pts /= (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-9)
    pts *= rng.uniform(0.3, 1.0, size=(c, 1)) * radius      # fill the ball
    return (pts - pts.mean(0, keepdims=True)).astype(np.float64)


def _source_xyz(az_deg: float, el_deg: float, dist_m: float) -> np.ndarray:
    az, el = np.radians(az_deg), np.radians(el_deg)
    return dist_m * np.array([np.cos(el) * np.cos(az),
                              np.cos(el) * np.sin(az),
                              np.sin(el)])


# -----------------------------------------------------------------------------
# Simulators
# -----------------------------------------------------------------------------

def _fractional_delay(sig: np.ndarray, delay_samples: float) -> np.ndarray:
    """Shift `sig` by a (possibly fractional) number of samples via FFT phase."""
    n = len(sig)
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n)
    return np.fft.irfft(spec * np.exp(-2j * np.pi * freqs * delay_samples), n=n)


def _render_freefield(source: np.ndarray, mics: np.ndarray, src: np.ndarray,
                      clip: int) -> np.ndarray:
    """Free-field point source: per-mic relative delay + 1/r gain. Returns
    (C, clip). `source` is padded around the crop window to avoid FFT wrap."""
    pad = 256
    dists = np.linalg.norm(mics - src[None, :], axis=1)     # (C,)
    delays = dists / C_SOUND * SAMPLE_RATE                  # absolute, samples
    delays = delays - delays.min()                          # relative (DoA cue)
    long = np.zeros(clip + 2 * pad)
    long[pad:pad + clip] = source[:clip]
    out = np.empty((len(mics), clip))
    for i, (d, r) in enumerate(zip(delays, dists)):
        shifted = _fractional_delay(long, d)
        out[i] = (1.0 / max(r, 1e-3)) * shifted[pad:pad + clip]   # 1/r distance cue
    return out


def _render_rir(source: np.ndarray, mics: np.ndarray, src: np.ndarray,
                clip: int, rng: np.random.Generator) -> np.ndarray:
    """Image-source RIR in a random shoebox (pyroomacoustics). Adds reverb →
    realistic distance cue. mics/src are centred at the origin; shifted into a
    random room here."""
    import pyroomacoustics as pra
    room_dim = rng.uniform([4, 4, 2.5], [10.5, 10.5, 5.0])
    centre = room_dim / 2.0
    rt60 = float(rng.uniform(0.2, 0.7))
    e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=SAMPLE_RATE, materials=pra.Material(e_abs),
                       max_order=min(max_order, 12))
    mic_pos = (mics + centre).T                             # (3, C)
    mic_pos = np.clip(mic_pos, 0.05, (room_dim - 0.05)[:, None])
    room.add_microphone_array(pra.MicrophoneArray(mic_pos, SAMPLE_RATE))
    spos = np.clip(src + centre, 0.05, room_dim - 0.05)
    room.add_source(spos, signal=source)
    room.simulate()
    sig = room.mic_array.signals                            # (C, T')
    if sig.shape[1] < clip:
        sig = np.pad(sig, ((0, 0), (0, clip - sig.shape[1])))
    return sig[:, :clip]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Synthetic geometry-agnostic spatial dataset")
    ap.add_argument("--out", required=True, help="output dir (one split)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim", choices=["free-field", "rir"], default="free-field")
    ap.add_argument("--min-mics", type=int, default=3)
    ap.add_argument("--max-mics", type=int, default=8)
    ap.add_argument("--el-range", type=float, nargs=2, default=[-60.0, 60.0],
                    help="elevation sampling range (deg)")
    ap.add_argument("--no-source-prob", type=float, default=0.0,
                    help="fraction of silent (no-source) samples")
    ap.add_argument("--speech-dir", default=None,
                    help="dir of audio files to use as sources (needs soundfile); "
                         "default = synthetic speech-like signal")
    ap.add_argument("--clip", type=int, default=CLIP_SAMPLES)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    speech_files = []
    if args.speech_dir:
        for root, _, files in os.walk(args.speech_dir):
            for fn in files:
                if fn.lower().endswith((".wav", ".flac", ".ogg", ".mp3")):
                    speech_files.append(os.path.join(root, fn))
        if not speech_files:
            raise ValueError(f"no audio files under {args.speech_dir}")
        print(f"[make] using {len(speech_files)} real source files from {args.speech_dir}")

    az_c = azimuth_bin_centers(N_AZIMUTH)
    el_c = elevation_bin_centers(N_ELEVATION)
    di_c = distance_bin_centers(N_DISTANCE, DIST_MIN, DIST_MAX)
    no_az, no_el, no_di = N_AZIMUTH - 1, N_ELEVATION - 1, N_DISTANCE - 1

    samples = []
    for idx in range(args.n):
        mics = _random_array(rng, args.min_mics, args.max_mics)
        no_source = rng.random() < args.no_source_prob

        if no_source:
            audio = 1e-3 * rng.standard_normal((len(mics), args.clip))
            az = el = dist = -1.0
            az_b, el_b, dist_b = no_az, no_el, no_di
        else:
            az = float(rng.uniform(0, 360))
            el = float(rng.uniform(args.el_range[0], args.el_range[1]))
            dist = float(rng.uniform(DIST_MIN, DIST_MAX))
            src = _source_xyz(az, el, dist)
            if speech_files:
                source = _load_speech(speech_files[rng.integers(len(speech_files))],
                                      args.clip + 512, rng)
            else:
                source = _speech_like(args.clip + 512, rng)
            if args.sim == "rir":
                audio = _render_rir(source, mics, src, args.clip, rng)
            else:
                audio = _render_freefield(source, mics, src, args.clip)
            az_b = value_to_bin(az, az_c, circular=True)
            el_b = value_to_bin(el, el_c)
            dist_b = value_to_bin(dist, di_c)

        np.savez(os.path.join(args.out, f"{idx}.npz"),
                 audio=audio.astype(np.float32), mic_coords=mics.astype(np.float32),
                 azimuth=np.float32(az), elevation=np.float32(el), distance=np.float32(dist),
                 az_bin=np.int64(az_b), el_bin=np.int64(el_b), dist_bin=np.int64(dist_b),
                 no_source=np.bool_(no_source))
        samples.append({"id": f"{idx}.npz", "n_mics": int(len(mics)), "no_source": bool(no_source)})

    manifest = {
        "meta": {
            "sample_rate": SAMPLE_RATE, "clip_samples": args.clip, "simulator": args.sim,
            "n_azimuth": N_AZIMUTH, "n_elevation": N_ELEVATION, "n_distance": N_DISTANCE,
            "dist_min": DIST_MIN, "dist_max": DIST_MAX,
            "min_mics": args.min_mics, "max_mics": args.max_mics,
        },
        "data": samples,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[make_phasecoder_dataset] wrote {args.n} samples to {args.out} ({args.sim})")
    print(f"  clip   : {args.clip} samples ({args.clip / SAMPLE_RATE * 1000:.0f} ms @ {SAMPLE_RATE} Hz)")
    print(f"  mics   : {args.min_mics}-{args.max_mics} (random geometry per sample)")
    print(f"  labels : az {N_AZIMUTH} / el {N_ELEVATION} / dist {N_DISTANCE} classes (+ no-source)")


if __name__ == "__main__":
    main()
