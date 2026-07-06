"""드롭테스트 rollout 배치 생성 → HDF5 데이터셋 (physicsFM 최초 직렬화 경로).

요약 흐름:
  1) physics/be 의 free_fall_trajectory 를 importlib 로 로드(원본 = 단일 진실,
     `models` 패키지명 충돌 회피 — sys.path 삽입 없음).
  2) 메쉬(meshes.py 레지스트리) × 무작위 action 으로 frames (T,N,3) 생성.
  3) 물리 시간 복원(D1): 생성기의 정규화 타임라인은 비율적으로 물리적
     (낙하 t_c1, 바운스 지속 2·t_c1·e^k) → 균일 사상 하나로 강체 운동이 정확히
     뉴턴역학이 된다. g=9.81 mesh-units/s² 로 두고
       T_phys = sqrt(2h/g) / t_c1,   frames = clamp(round(T_phys/dt_target)+1, 2, 240)
       dt_s   = T_phys / (frames−1)
     즉 dt 는 고정 목표(12ms) 근방, 시퀀스 길이 T 가 (h,e)에 따라 연속적으로 변한다
     (eos 라벨 퇴화 방지). 교과서식 무한 바운스 공식은 금지 — 생성기는 2회 절단.
  4) 타임라인 메타(t_c1/events/z_rigid)는 로드된 모듈의 상수로 재유도.
     scale=0 패리티(변형 0 → frames = vrot + z_rigid)로 검증(tests/test_datagen.py).
  5) hold-extension(D5): 정착 프레임을 U{0..hold_frac·T}개 복제 추가(frame_kind=1,
     물리적으로 참인 평형 상태) — eos 양성 라벨과 settled 고정점 학습용.
  6) ke/pe 는 f32 양자화 전 f64 로 계산해 저장. 질량 = 면적 가중 lumped(Σ=1).
     eos 라벨 자체는 저장하지 않는다 — dataset.py 가 ke 에서 파생(ε 는 설정값).

스키마는 파일 attrs 의 schema_version 으로 버전 관리. --inspect 로 불변식 검사.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import h5py
import numpy as np

from common import git_commit, load_config, setup_logging
from meshes import EVAL_MESHES, Mesh, load_mesh

SCHEMA_VERSION = 1
PHYSICS_REPO = Path(__file__).resolve().parent.parent / "physics"
TRAJECTORY_PY = PHYSICS_REPO / "be" / "models" / "free_fall" / "trajectory.py"
SPLIT_FRACS = {"train": 0.8, "valid": 0.1, "test": 0.1}


def load_trajectory_module():
    """physics/be 원본 생성기를 파일 경로로 로드 (미러/복사 금지 — 상수 드리프트 방지)."""
    spec = importlib.util.spec_from_file_location("free_fall_trajectory_mod", TRAJECTORY_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_timeline(traj_mod, height: float, rest_e: float, frames: int):
    """생성기와 동일한 타임라인 재유도 → (ts, t_c1, events, z_rigid).

    trajectory.py:148-173 의 수식을 로드된 모듈의 상수로 재계산한다.
    scale=0 패리티 테스트가 이 함수의 정확성을 바이트 수준으로 보증한다.
    """
    bounce_ks = [
        k for k in range(1, traj_mod._MAX_BOUNCES + 1)
        if rest_e ** (2 * k) >= traj_mod._MIN_BOUNCE_H_FRAC
    ]
    denom = 1.0 + 2.0 * sum(rest_e ** k for k in bounce_ks)
    t_c1 = min(traj_mod._FALL_FRAC_MAX, traj_mod._CONTACT_END / denom)
    events = [t_c1]
    durations = []
    for k in bounce_ks:
        durations.append(2.0 * t_c1 * rest_e ** k)
        events.append(events[-1] + durations[-1])

    ts = np.linspace(0.0, 1.0, frames)
    z_rigid = np.zeros(frames)
    falling = ts < events[0]
    z_rigid[falling] = height * (1.0 - (ts[falling] / t_c1) ** 2)
    for idx, k in enumerate(bounce_ks):
        start, dur = events[idx], durations[idx]
        m = (ts >= start) & (ts < start + dur)
        tau = (ts[m] - start) / dur
        z_rigid[m] = 4.0 * height * rest_e ** (2 * k) * tau * (1.0 - tau)
    return ts, t_c1, np.asarray(events), z_rigid


def physical_frames(t_c1: float, height: float, g: float, dt_target: float, traj_mod):
    """고정 dt 목표에 맞춘 (frames, dt_s, T_phys). frames 는 생성기 캡 [2, MAX] 준수."""
    t_fall = float(np.sqrt(2.0 * height / g))
    t_phys = t_fall / t_c1  # 정규화 [0,1] 전체의 물리 길이 (초)
    frames = int(np.clip(round(t_phys / dt_target) + 1, 2, traj_mod.MAX_FRAMES))
    dt_s = t_phys / (frames - 1)
    return frames, dt_s, t_phys


def sample_action(rng: np.random.Generator, cfg_data: dict) -> dict:
    """설정 범위에서 action 무작위 샘플 (frames 는 dt 규칙이 결정 — 별도 샘플 금지)."""
    lo, hi = cfg_data["drop_height"]
    drop_height = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    e_lo, e_hi = cfg_data["restitution"]
    s_lo, s_hi = cfg_data["scale"]
    return {
        "drop_height": drop_height,
        "restitution": float(rng.uniform(e_lo, e_hi)),
        "orientation": [float(a) for a in rng.uniform(-180.0, 180.0, size=3)],
        "scale": float(rng.uniform(s_lo, s_hi)),
    }


def compute_vrot(traj_mod, vertices: np.ndarray, action: dict) -> tuple[np.ndarray, float]:
    """생성기와 동일한 낙하 자세 정렬 (trajectory.py:133-137 재유도) → (vrot, diag)."""
    rot = traj_mod._rotation_matrix(action.get("orientation", (0.0, 0.0, 0.0)))
    center = vertices.mean(0)
    vrot = (vertices - center) @ rot.T + center
    vrot = vrot - np.array([0.0, 0.0, float(vrot[:, 2].min())])
    diag = float(np.linalg.norm(vrot.max(0) - vrot.min(0))) or 1.0
    return vrot, diag


def derive_energies(positions: np.ndarray, node_mass: np.ndarray, dt_s: float, g: float):
    """f64 로 ke/pe 계산 (f32 저장 전). v 는 후방 차분, v[0]=0 (생성기 초기 정지)."""
    vel = np.zeros_like(positions)
    vel[1:] = (positions[1:] - positions[:-1]) / dt_s
    ke = 0.5 * np.einsum("n,tn->t", node_mass, np.einsum("tnc,tnc->tn", vel, vel))
    pe = g * np.einsum("n,tn->t", node_mass, positions[:, :, 2])
    return ke, pe


def generate_one(traj_mod, mesh: Mesh, action: dict, cfg_data: dict,
                 rng: np.random.Generator) -> dict:
    """rollout 하나 생성 → H5 기록용 dict."""
    g = float(cfg_data["g"])
    # 타임라인/프레임 수 결정 (t_c1 은 e 만의 함수 — frames 계산에 먼저 필요)
    _, t_c1, _, _ = compute_timeline(traj_mod, action["drop_height"], action["restitution"], 2)
    frames_n, dt_s, t_phys = physical_frames(
        t_c1, action["drop_height"], g, float(cfg_data["dt_target_s"]), traj_mod
    )
    action = dict(action, frames=frames_n)

    frames_arr = traj_mod.free_fall_trajectory(mesh.vertices, action)  # (T,N,3) f64
    vrot, diag = compute_vrot(traj_mod, mesh.vertices, action)
    _, _, _, z_rigid = compute_timeline(
        traj_mod, action["drop_height"], action["restitution"], frames_n
    )
    # 생성기와의 정렬 검증 — frames[0] = vrot + (0,0,h) (변형은 t=0 에서 0)
    assert np.allclose(
        frames_arr[0], vrot + np.array([0.0, 0.0, action["drop_height"]]), atol=1e-9
    ), "vrot 재유도가 생성기와 불일치"

    centers, shares = traj_mod._contact_sites(vrot, diag)
    z_ext = float(vrot[:, 2].max())  # 정렬 후 min z = 0
    tol = max(traj_mod._BAND_FRACTION * z_ext, 1e-9)
    contact_mask = (vrot[:, 2] <= tol).astype(np.uint8)

    # hold-extension: 정착 프레임 복제 (물리적으로 참인 평형 — eos 양성/고정점 학습)
    n_hold = int(rng.integers(0, int(cfg_data["hold_frac"] * frames_n) + 1))
    if n_hold:
        hold = np.repeat(frames_arr[-1:], n_hold, axis=0)
        frames_arr = np.concatenate([frames_arr, hold], axis=0)
        z_rigid = np.concatenate([z_rigid, np.full(n_hold, z_rigid[-1])])
    frame_kind = np.zeros(len(frames_arr), dtype=np.uint8)
    frame_kind[frames_n:] = 1

    ke, pe = derive_energies(frames_arr, mesh.node_mass, dt_s, g)
    return {
        "positions": frames_arr.astype(np.float32),
        "rest_positions": vrot.astype(np.float32),
        "z_rigid": z_rigid.astype(np.float32),
        "ke": ke.astype(np.float32),
        "pe": pe.astype(np.float32),
        "frame_kind": frame_kind,
        "contact_site_centers": centers.astype(np.float32),
        "contact_site_shares": shares.astype(np.float32),
        "contact_mask": contact_mask,
        "attrs": {
            "mesh": mesh.mesh_id,
            "action": json.dumps(action),
            "dt_s": dt_s,
            "T_phys_s": t_phys,
            "t_c1_norm": t_c1,
            "frames_generated": frames_n,
        },
    }


def write_mesh(h5: h5py.File, mesh: Mesh) -> None:
    grp = h5.require_group(f"meshes/{mesh.mesh_id}")
    grp.create_dataset("vertices", data=mesh.vertices)
    grp.create_dataset("faces", data=mesh.faces.astype(np.int32))
    grp.create_dataset("edges", data=mesh.edges.astype(np.int32))
    grp.create_dataset("node_mass", data=mesh.node_mass.astype(np.float32))
    grp.create_dataset("component_id", data=mesh.component_id.astype(np.int16))
    grp.create_dataset("node_type", data=mesh.node_type.astype(np.int8))
    grp.attrs["real_scale_m_per_unit"] = mesh.unit_scale_m
    grp.attrs["component_table"] = json.dumps(mesh.component_table, ensure_ascii=False)
    grp.attrs["source"] = mesh.source


def write_rollout(h5: h5py.File, rid: str, rec: dict, split: str, seed: int) -> None:
    grp = h5.create_group(f"rollouts/{rid}")
    pos = rec["positions"]
    grp.create_dataset(
        "positions", data=pos,
        chunks=(min(4, pos.shape[0]), pos.shape[1], 3), compression="gzip",
        compression_opts=4, shuffle=True,
    )
    for key in ("rest_positions", "z_rigid", "ke", "pe", "frame_kind",
                "contact_site_centers", "contact_site_shares", "contact_mask"):
        grp.create_dataset(key, data=rec[key])
    for k, v in rec["attrs"].items():
        grp.attrs[k] = v
    grp.attrs["split"] = split
    grp.attrs["seed"] = seed


def assign_splits(mesh_ids: list[str], counts: dict[str, int],
                  rng: np.random.Generator) -> list[tuple[str, str]]:
    """(mesh_id, split) 목록 — 학습 메쉬는 8:1:1 층화, 평가 메쉬는 전부 test."""
    plan = []
    for mid in mesh_ids:
        n = counts[mid]
        if mid in EVAL_MESHES:
            plan.extend((mid, "test") for _ in range(n))
            continue
        # n 이 아주 작아도 (train ≥ 소수) 총합이 n 을 넘지 않게 클램프
        n_te = min(max(1, round(n * SPLIT_FRACS["test"])), max(n - 1, 0))
        n_va = min(max(1, round(n * SPLIT_FRACS["valid"])), max(n - 1 - n_te, 0))
        splits = ["train"] * (n - n_va - n_te) + ["valid"] * n_va + ["test"] * n_te
        rng.shuffle(splits)
        plan.extend((mid, s) for s in splits)
    return plan


def generate(cfg: dict, out_path: str) -> None:
    cfg_data = cfg["data"]
    setup_logging()
    traj_mod = load_trajectory_module()
    base_seed = int(cfg_data["seed"])
    rng_split = np.random.default_rng(base_seed)

    mesh_ids = list(cfg_data["train_meshes"]) + list(cfg_data["eval_meshes"])
    counts = {
        mid: (cfg_data["eval_rollouts_per_mesh"] if mid in EVAL_MESHES
              else cfg_data["rollouts_per_mesh"])
        for mid in mesh_ids
    }
    plan = assign_splits(mesh_ids, counts, rng_split)
    meshes = {mid: load_mesh(mid) for mid in mesh_ids}

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    split_lists: dict[str, list[str]] = {s: [] for s in SPLIT_FRACS}
    with h5py.File(out, "w") as h5:
        h5.attrs.update({
            "schema_version": SCHEMA_VERSION,
            "g": float(cfg_data["g"]),
            "dt_target_s": float(cfg_data["dt_target_s"]),
            "coordinate_frame": "z-up, floor z=0",
            "mass_convention": "area_lumped_unit_total",
            "physics_repo_commit": git_commit(PHYSICS_REPO),
            "seed": base_seed,
        })
        for mesh in meshes.values():
            write_mesh(h5, mesh)
        for i, (mid, split) in enumerate(plan):
            seed_r = base_seed + 1000 + i
            rng = np.random.default_rng(seed_r)
            action = sample_action(rng, cfg_data)
            rec = generate_one(traj_mod, meshes[mid], action, cfg_data, rng)
            rid = f"r{i:05d}"
            write_rollout(h5, rid, rec, split, seed_r)
            split_lists[split].append(rid)
            if (i + 1) % 50 == 0 or i + 1 == len(plan):
                logging.info("%d/%d rollouts (%s)", i + 1, len(plan), mid)
        str_dt = h5py.string_dtype()
        for split, rids in split_lists.items():
            h5.create_dataset(f"splits/{split}", data=np.array(rids, dtype=str_dt))
        h5.attrs["num_rollouts"] = len(plan)
    logging.info("완료: %s (%.1f MB)", out, out.stat().st_size / 1e6)


def inspect(h5_path: str) -> None:
    """스키마 요약 + 불변식 검사. 실패 시 AssertionError."""
    with h5py.File(h5_path, "r") as h5:
        g = float(h5.attrs["g"])
        print(f"schema v{h5.attrs['schema_version']}  g={g}  "
              f"dt_target={h5.attrs['dt_target_s']}s  commit={h5.attrs['physics_repo_commit'][:8]}")
        print(f"meshes: {list(h5['meshes'])}")
        for mid, grp in h5["meshes"].items():
            mass_sum = float(np.sum(grp["node_mass"]))
            assert abs(mass_sum - 1.0) < 1e-5, f"{mid} 질량 합 {mass_sum} != 1"
            n = grp["vertices"].shape[0]
            assert int(np.max(grp["faces"])) < n and int(np.max(grp["edges"])) < n
            print(f"  {mid}: N={n} M={grp['faces'].shape[0]} E={grp['edges'].shape[0]}")

        counts, t_lens = {}, []
        for rid, grp in h5["rollouts"].items():
            a = grp.attrs
            dt_s, t_c1 = float(a["dt_s"]), float(a["t_c1_norm"])
            action = json.loads(a["action"])
            h, T_gen = action["drop_height"], int(a["frames_generated"])
            pos = grp["positions"][:]
            assert dt_s > 0 and np.isfinite(pos).all()
            # 낙하 포물선 항등식 (D1 증명): 물리 시간 τ 에서 z_rigid = h − ½gτ²
            z_rigid = grp["z_rigid"][:]
            ts_norm = np.linspace(0.0, 1.0, T_gen)
            fall = ts_norm < t_c1
            tau = ts_norm[fall] * float(a["T_phys_s"])
            assert np.allclose(z_rigid[:T_gen][fall], h - 0.5 * g * tau**2, atol=1e-5), \
                f"{rid}: 낙하 포물선 불일치"
            shares = grp["contact_site_shares"][:]
            assert abs(float(shares.sum()) - 1.0) < 1e-5
            ke = grp["ke"][:]
            assert ke[0] == 0.0 and (ke >= 0).all()
            fk = grp["frame_kind"][:]
            assert fk[0] == 0 and (np.diff(fk.astype(int)) >= 0).all()  # hold 는 꼬리에만
            counts[(a["mesh"], a["split"])] = counts.get((a["mesh"], a["split"]), 0) + 1
            t_lens.append(len(pos))
        for s in ("train", "valid", "test"):
            rids = [x.decode() if isinstance(x, bytes) else x for x in h5[f"splits/{s}"][:]]
            print(f"split {s}: {len(rids)}")
        print(f"rollouts: {len(t_lens)}, T 범위 [{min(t_lens)}, {max(t_lens)}], "
              f"평균 {np.mean(t_lens):.0f}")
        for (mid, split), c in sorted(counts.items()):
            print(f"  {mid:14s} {split:5s} {c}")
    print("모든 불변식 OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None, help="출력 h5 (기본: config data.h5)")
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    metavar="KEY=VAL", help="설정 오버라이드 (예: data.rollouts_per_mesh=4)")
    ap.add_argument("--inspect", metavar="H5", help="생성 대신 기존 파일 검사")
    args = ap.parse_args()
    if args.inspect:
        inspect(args.inspect)
        return
    cfg = load_config(args.config, args.overrides)
    generate(cfg, args.out or cfg["data"]["h5"])


if __name__ == "__main__":
    main()
