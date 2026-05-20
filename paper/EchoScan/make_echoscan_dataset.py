"""Simulate the EchoScan training/eval dataset with Pyroomacoustics.

Why this script *is* the dataset
---------------------------------
EchoScan trains on synthesised RIRs — there is no single file to download.
The paper builds its "Basic Room" set by drawing random room polygons of five
types (quadrilateral / pentagonal / hexagonal / L-type / T-type), placing a
commercial-style audio device (a 6-mic, 5 cm-radius circular array with a
loudspeaker at the centre) inside, and simulating the multi-channel RIRs with
Pyroomacoustics' ray-tracing engine. This script reproduces that pipeline, so
running it *obtains* a usable dataset locally (see README "데이터셋 준비").

For each sample we save one ``.npz`` holding
    rir          float32 (M, N)         — direct sound removed, Gaussian noise added
    floor_packed uint8                  — np.packbits of the (b×b) floorplan label
    height       uint8 (h,)             — interior=1 height profile
    b, h, fs, ...                       — scalar attributes
and append ``{rir_id, polygon, floor_z, ceil_z, room_type}`` to ``manifest.json``
(``polygon`` is the device-local room outline, kept for visualisation/eval).

The grid convention matches `EchoScan.py`: local coordinates centred at the
device, 2 cm/pixel, floorplan ±(b/2)·2 cm, height ±(h/2)·2 cm.

Usage
-----
    python make_echoscan_dataset.py --out data/train --n 50000
    python make_echoscan_dataset.py --out data/test  --n 1000 --seed 1
    # fast CPU smoke set (tiny maps, ISM only):
    python make_echoscan_dataset.py --out /tmp/echoscan_smoke --n 8 \
        --floorplan-size 128 --height-size 64 --rir-length 256 \
        --no-ray-tracing --max-order 3
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Pyroomacoustics is heavy (scipy/C++); import lazily so --help works without it.
try:
    import pyroomacoustics as pra
except Exception:                                    # pragma: no cover
    pra = None

SPEED_OF_SOUND = 343.0
ARRAY_RADIUS = 0.05          # m — 5 cm circular mic array (source at centre)
ROOM_TYPES = ("quad", "pentagon", "hexagon", "L", "T")


# -----------------------------------------------------------------------------
# Room polygon generators (world frame, CCW corners, metres)
# -----------------------------------------------------------------------------

def _regular_polygon(n: int, radius: float, jitter: float, rng) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ang = ang + rng.uniform(-0.15, 0.15, size=n)             # break the symmetry
    r = radius * (1.0 + rng.uniform(-jitter, jitter, size=n))
    return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)


def _quad(rng) -> np.ndarray:
    lx, ly = rng.uniform(2.0, 5.0), rng.uniform(2.0, 5.0)
    base = np.array([[0, 0], [lx, 0], [lx, ly], [0, ly]], dtype=float)
    base += rng.uniform(-0.5, 0.5, size=base.shape)          # ±0.5 m vertex perturb
    return base


def _l_shape(rng) -> np.ndarray:
    a, b = rng.uniform(3.0, 5.0), rng.uniform(3.0, 5.0)      # outer box
    c, d = rng.uniform(1.0, a - 1.0), rng.uniform(1.0, b - 1.0)   # notch
    poly = np.array([[0, 0], [a, 0], [a, d], [c, d], [c, b], [0, b]], dtype=float)
    return poly + rng.uniform(-0.3, 0.3, size=poly.shape)


def _t_shape(rng) -> np.ndarray:
    w, h = rng.uniform(3.0, 5.0), rng.uniform(3.0, 5.0)      # full width / height
    sw = rng.uniform(1.0, w - 1.0)                            # stem width
    sh = rng.uniform(1.0, h - 1.0)                            # arm (top) height
    x0 = (w - sw) / 2.0
    poly = np.array([
        [x0, 0], [x0 + sw, 0], [x0 + sw, h - sh],
        [w, h - sh], [w, h], [0, h], [0, h - sh], [x0, h - sh],
    ], dtype=float)
    return poly + rng.uniform(-0.2, 0.2, size=poly.shape)


def make_polygon(room_type: str, rng) -> np.ndarray:
    if room_type == "quad":
        return _quad(rng)
    if room_type == "pentagon":
        return _regular_polygon(5, rng.uniform(1.6, 3.2), 0.25, rng)
    if room_type == "hexagon":
        return _regular_polygon(6, rng.uniform(1.6, 3.2), 0.25, rng)
    if room_type == "L":
        return _l_shape(rng)
    if room_type == "T":
        return _t_shape(rng)
    raise ValueError(f"unknown room_type {room_type!r}")


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _polygon_centroid(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return poly.mean(axis=0)
    cx = ((x + x1) * cross).sum() / (6.0 * area)
    cy = ((y + y1) * cross).sum() / (6.0 * area)
    return np.array([cx, cy])


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised ray-casting point-in-polygon. pts (P,2), poly (V,2) → (P,) bool."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    xj, yj = poly[-1]
    for xi, yi in poly:
        cond = ((yi > y) != (yj > y)) & \
               (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        xj, yj = xi, yi
    return inside


def sample_device_pose(poly: np.ndarray, rng, shrink: float = 0.7):
    """Device (x, y) inside the 0.7-scaled polygon (avoid wall proximity),
    z in [1, 1.5] m; orientation θ uniform in [0, 2π)."""
    centroid = _polygon_centroid(poly)
    shrunk = centroid + shrink * (poly - centroid)
    lo, hi = shrunk.min(axis=0), shrunk.max(axis=0)
    for _ in range(200):
        xy = rng.uniform(lo, hi)
        if points_in_polygon(xy[None, :], shrunk)[0]:
            return xy, rng.uniform(1.0, 1.5), rng.uniform(0.0, 2 * np.pi)
    return centroid, rng.uniform(1.0, 1.5), rng.uniform(0.0, 2 * np.pi)


def circular_array_xy(radius: float, n: int) -> np.ndarray:
    """n mic offsets on a circle of `radius` (source sits at the centre)."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)


# -----------------------------------------------------------------------------
# Label rasterisation (device-local frame)
# -----------------------------------------------------------------------------

def rasterize_floorplan(poly_local: np.ndarray, size: int, pixel_m: float) -> np.ndarray:
    """Device-local polygon → (size, size) uint8 interior mask. Row 0 is +y
    (north-up); column 0 is -x. Centre pixel ≈ device origin."""
    half = size * pixel_m / 2.0
    axis = (np.arange(size) - size / 2.0 + 0.5) * pixel_m
    xx, yy = np.meshgrid(axis, axis[::-1])               # y flipped → row 0 = +y
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    mask = points_in_polygon(pts, poly_local).reshape(size, size)
    return mask.astype(np.uint8)


def rasterize_height(floor_z: float, ceil_z: float, size: int, pixel_m: float) -> np.ndarray:
    """1-D interior mask over z ∈ ±(size/2)·pixel_m, relative to the device."""
    axis = (np.arange(size) - size / 2.0 + 0.5) * pixel_m
    return ((axis >= floor_z) & (axis <= ceil_z)).astype(np.uint8)


# -----------------------------------------------------------------------------
# RIR simulation
# -----------------------------------------------------------------------------

def simulate_rir(poly: np.ndarray, room_height: float, device_xy: np.ndarray,
                 device_z: float, fs: int, rir_length: int, absorption: float,
                 ray_tracing: bool, max_order: int, n_mics: int, rng) -> np.ndarray:
    """Return (M, rir_length) RIR with the direct sound removed and Gaussian
    noise (SNR 10–20 dB) added. Mics on a 5 cm circle, source at the centre."""
    corners = poly.T                                    # pra wants (2, V)
    room = pra.Room.from_corners(
        corners, fs=fs, max_order=max_order,
        materials=pra.Material(absorption), ray_tracing=ray_tracing,
        air_absorption=True,
    )
    room.extrude(room_height, materials=pra.Material(absorption))
    if ray_tracing:
        room.set_ray_tracing(n_rays=2000, receiver_radius=0.5)

    mic_xy = device_xy[None, :] + circular_array_xy(ARRAY_RADIUS, n_mics)
    mic_locs = np.column_stack([mic_xy, np.full(n_mics, device_z)]).T  # (3, M)
    room.add_source([device_xy[0], device_xy[1], device_z])
    room.add_microphone_array(pra.MicrophoneArray(mic_locs, fs))
    room.compute_rir()

    rirs = room.rir                                     # rirs[mic][src]
    maxlen = max(len(rirs[m][0]) for m in range(n_mics))
    full = np.zeros((n_mics, maxlen), dtype=np.float32)
    for m in range(n_mics):
        r = np.asarray(rirs[m][0], dtype=np.float32)
        full[m, :len(r)] = r

    # Remove the direct sound: it is the earliest, strongest peak. Find the
    # onset from the channel-summed energy, window N samples from there, then
    # zero the first few samples (the direct impulse, mics ≈5 cm from source).
    energy = np.abs(full).sum(axis=0)
    thresh = 0.02 * energy.max() if energy.max() > 0 else 0.0
    onset = int(np.argmax(energy > thresh)) if thresh > 0 else 0
    seg = full[:, onset:onset + rir_length]
    if seg.shape[1] < rir_length:
        seg = np.pad(seg, ((0, 0), (0, rir_length - seg.shape[1])))
    direct_guard = max(1, int(round(2 * ARRAY_RADIUS / SPEED_OF_SOUND * fs)) + 2)
    seg = seg.copy()
    seg[:, :direct_guard] = 0.0

    snr_db = rng.uniform(10.0, 20.0)
    sig_pow = float(np.mean(seg ** 2))
    if sig_pow > 0:
        noise_std = np.sqrt(sig_pow / (10.0 ** (snr_db / 10.0)))
        seg = seg + rng.normal(0.0, noise_std, size=seg.shape).astype(np.float32)
    return seg.astype(np.float32)


def world_to_local(poly: np.ndarray, device_xy: np.ndarray, theta: float) -> np.ndarray:
    """Translate so the device is at the origin and rotate by -θ."""
    c, s = np.cos(-theta), np.sin(-theta)
    rot = np.array([[c, -s], [s, c]])
    return (poly - device_xy) @ rot.T


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--n", type=int, default=1000, help="number of rooms/RIRs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fs", type=int, default=8000)
    ap.add_argument("--rir-length", type=int, default=1024)
    ap.add_argument("--n-mics", type=int, default=6)
    ap.add_argument("--floorplan-size", type=int, default=1024)
    ap.add_argument("--height-size", type=int, default=512)
    ap.add_argument("--pixel-m", type=float, default=0.02)
    ap.add_argument("--room-types", nargs="+", default=list(ROOM_TYPES),
                    choices=ROOM_TYPES)
    ap.add_argument("--abs-min", type=float, default=0.05)
    ap.add_argument("--abs-max", type=float, default=0.45)
    ap.add_argument("--max-order", type=int, default=3,
                    help="image-source order (ray tracing handles higher orders)")
    ap.add_argument("--no-ray-tracing", dest="ray_tracing", action="store_false",
                    help="ISM only — much faster, for smoke tests")
    ap.set_defaults(ray_tracing=True)
    args = ap.parse_args()

    if pra is None:
        raise SystemExit("pyroomacoustics is required: pip install -r requirements.txt")

    N_MICS_DEFAULT = args.n_mics
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    manifest = {
        "meta": {
            "fs": args.fs, "rir_length": args.rir_length, "n_mics": args.n_mics,
            "floorplan_size": args.floorplan_size, "height_size": args.height_size,
            "pixel_m": args.pixel_m, "array_radius": ARRAY_RADIUS,
        },
        "data": [],
    }

    made = 0
    attempts = 0
    while made < args.n:
        attempts += 1
        if attempts > 20 * args.n + 100:
            raise RuntimeError(f"too many failed rooms ({attempts}); made={made}")
        room_type = rng.choice(args.room_types)
        poly = make_polygon(room_type, rng)
        room_height = float(rng.uniform(3.0, 5.0))
        absorption = float(rng.uniform(args.abs_min, args.abs_max))
        device_xy, device_z, theta = sample_device_pose(poly, rng)
        try:
            rir = simulate_rir(poly, room_height, device_xy, device_z, args.fs,
                               args.rir_length, absorption, args.ray_tracing,
                               args.max_order, args.n_mics, rng)
        except Exception as e:                          # skip degenerate rooms
            if attempts <= 5 or made == 0:
                print(f"  [skip] {room_type} room failed: {e}")
            continue

        poly_local = world_to_local(poly, device_xy, theta)
        floor = rasterize_floorplan(poly_local, args.floorplan_size, args.pixel_m)
        if floor.sum() == 0:                            # device fell outside the raster
            continue
        floor_z, ceil_z = -device_z, room_height - device_z
        height = rasterize_height(floor_z, ceil_z, args.height_size, args.pixel_m)

        rir_id = f"{made:06d}.npz"
        np.savez_compressed(
            os.path.join(args.out, rir_id),
            rir=rir,
            floor_packed=np.packbits(floor.reshape(-1)),
            height=height,
            b=np.int32(args.floorplan_size), h=np.int32(args.height_size),
        )
        manifest["data"].append({
            "rir_id": rir_id,
            "room_type": str(room_type),
            "polygon": poly_local.round(4).tolist(),
            "floor_z": round(floor_z, 4), "ceil_z": round(ceil_z, 4),
        })
        made += 1
        if made % max(1, args.n // 20) == 0 or made == args.n:
            print(f"  [{made}/{args.n}] rooms simulated ({attempts} attempts)")

    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"done: {made} samples → {args.out}  (manifest: {manifest_path})")


if __name__ == "__main__":
    main()
