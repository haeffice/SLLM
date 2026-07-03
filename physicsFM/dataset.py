"""HDF5 rollout → 1-step 학습 샘플 Dataset + rollout 평가 로더.

요약 흐름:
  1) 라벨/피처는 전부 로더 파생(D5) — H5 에는 물리 원료만 있다.
     eos: settled(t) = 1 iff max_{t'≥t} KE(t') ≤ ε·max KE (역방향 cummax —
     바운스 정점의 순간 KE≈0 에 면역).
  2) 1-step 샘플 = (rollout, t), t ∈ [1, T−2]. 타깃은 가속도형(D2)
     y = x_{t+1} − 2x_t + x_{t−1}  (자유낙하 중 참값이 상수 −g·dt² → 높이 외삽 유리).
  3) 학습 노이즈(D9): x_{t−1}, x_t 에 σ=3e-3 i.i.d. 를 더하고 타깃은
     노이즈 입력 기준·클린 x_{t+1} 대상으로 재계산 → 롤아웃 오차 교정 학습.
  4) 정규화 통계는 train split 전체를 f64 스트리밍으로 계산해
     data/stats_<hash>.json 에 캐시(피처 설정 종속 — H5 에 넣지 않음).
  5) h5py 파일은 worker 별 lazy open (DataLoader 멀티프로세스 안전).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

import graph as G

FLOOR_DELTA_FACTOR = 3.0  # δ = 이 배수 × p99.9 스텝 변위 (절대 높이 포화 지점)
STATS_SAMPLE_EDGES = 200  # 엣지 통계에 쓰는 train 샘플 수


def derive_eos(ke: np.ndarray, eps: float) -> np.ndarray:
    """settled 라벨 (T,) f32 — KE 역방향 누적최대가 ε·max KE 이하인 구간."""
    tail_max = np.maximum.accumulate(ke[::-1])[::-1]
    return (tail_max <= eps * float(ke.max())).astype(np.float32)


def _stats_key(h5_path: str, cfg: dict) -> str:
    """캐시 키 — 통계에 영향 주는 모든 것을 포함 (h5 내용 지문 = mtime+size+커밋)."""
    st = Path(h5_path).stat()
    with h5py.File(h5_path, "r") as h5:
        ident = {
            "h5": str(Path(h5_path).resolve()),
            "mtime_ns": st.st_mtime_ns,
            "size": st.st_size,
            "schema": int(h5.attrs["schema_version"]),
            "seed": int(h5.attrs["seed"]),
            "num": int(h5.attrs["num_rollouts"]),
            "dt_target_s": float(h5.attrs["dt_target_s"]),
            "commit": str(h5.attrs["physics_repo_commit"]),
            "history": int(cfg["graph"]["history"]),
            "noise_std": float(cfg["train"]["noise_std"]),
            "eos_eps": float(cfg["labels"]["eos_eps"]),
        }
    return hashlib.md5(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:10]


def compute_stats(h5_path: str, cfg: dict) -> dict:
    """train split 전체의 vel/target/엣지 통계 + floor_delta + eos_pos_weight.

    캐시: data/stats_<hash>.json (피처 설정 종속). 전부 f64 로 누적.
    """
    cache = Path(h5_path).parent / f"stats_{_stats_key(h5_path, cfg)}.json"
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            stats = json.load(f)
        return {k: np.asarray(v) if isinstance(v, list) else v for k, v in stats.items()}

    logging.info("stats 계산 중 (train split 전체)...")
    eps = float(cfg["labels"]["eos_eps"])
    sum_v = np.zeros(3); sumsq_v = np.zeros(3); n_v = 0
    sum_y = np.zeros(3); sumsq_y = np.zeros(3); n_y = 0
    step_disp_samples, dt_list = [], []
    n_pos = n_neg = 0
    edge_rows_sum = np.zeros(G.EDGE_FEAT_DIM); edge_rows_sumsq = np.zeros(G.EDGE_FEAT_DIM)
    n_e = 0

    with h5py.File(h5_path, "r") as h5:
        train_ids = [x.decode() if isinstance(x, bytes) else x for x in h5["splits/train"][:]]
        rng = np.random.default_rng(0)
        edge_sample_ids = set(rng.choice(len(train_ids), min(STATS_SAMPLE_EDGES, len(train_ids)),
                                         replace=False).tolist())
        for i, rid in enumerate(train_ids):
            grp = h5[f"rollouts/{rid}"]
            pos = grp["positions"][:].astype(np.float64)
            dt = float(grp.attrs["dt_s"])
            dt_list.append(dt)
            vel = (pos[1:] - pos[:-1]) / dt                      # (T-1,N,3)
            y = pos[2:] - 2.0 * pos[1:-1] + pos[:-2]             # (T-2,N,3) 가속도형
            sum_v += vel.sum((0, 1)); sumsq_v += (vel**2).sum((0, 1)); n_v += vel.shape[0] * vel.shape[1]
            sum_y += y.sum((0, 1)); sumsq_y += (y**2).sum((0, 1)); n_y += y.shape[0] * y.shape[1]
            step = np.linalg.norm(pos[1:] - pos[:-1], axis=2)    # (T-1,N)
            step_disp_samples.append(np.percentile(step, 99.9))
            lab = derive_eos(grp["ke"][:], eps)[1:-1]            # 샘플 인덱스 범위와 일치
            n_pos += int(lab.sum()); n_neg += int(len(lab) - lab.sum())
            if i in edge_sample_ids:
                mesh_grp = h5[f"meshes/{grp.attrs['mesh']}"]
                edge_index = G.build_edge_index(mesh_grp["edges"][:])
                t = rng.integers(1, pos.shape[0] - 1)
                _, ef = G.build_sample(pos[t - 1], pos[t], grp["rest_positions"][:],
                                       edge_index, mesh_grp["node_type"][:], dt, 1.0)
                edge_rows_sum += ef.astype(np.float64).sum(0)
                edge_rows_sumsq += (ef.astype(np.float64)**2).sum(0)
                n_e += ef.shape[0]

    def _mean_std(s, ss, n):
        mean = s / n
        std = np.sqrt(np.maximum(ss / n - mean**2, 1e-16))
        return mean, np.maximum(std, 1e-12)

    vel_mean, vel_std = _mean_std(sum_v, sumsq_v, n_v)
    y_mean, y_std = _mean_std(sum_y, sumsq_y, n_y)
    e_mean, e_std = _mean_std(edge_rows_sum, edge_rows_sumsq, n_e)

    # 학습 노이즈가 더하는 분산을 해석적으로 반영(D9) — 이 생성기는 x/y 가 시간에
    # 대해 상수라 클린 std_x/y ≈ 0 이고, 반영하지 않으면 노이즈 주입 시 표준화
    # 타깃이 폭발한다. y − 2ε_t + ε_p → Var+=5σ²; v=(…+ε_t−ε_p)/dt → Var+=2σ²/dt²;
    # 엣지 Δx → Var+=2σ². (valid/test 도 같은 stats 를 써야 지표가 비교 가능)
    sigma = float(cfg["train"]["noise_std"])
    if sigma > 0:
        dt_ref = float(np.mean(dt_list))  # 실제 train dt 평균 (frames 캡 클램프에도 안전)
        vel_std = np.sqrt(vel_std**2 + 2.0 * sigma**2 / dt_ref**2)
        y_std = np.sqrt(y_std**2 + 5.0 * sigma**2)
        e_std = e_std.copy()
        e_std[4:8] = np.sqrt(e_std[4:8] ** 2 + 2.0 * sigma**2)
    max_step = float(np.max(step_disp_samples))
    stats = {
        "vel_mean": vel_mean, "vel_std": vel_std,
        "target_mean": y_mean, "target_std": y_std,
        "edge_mean": e_mean, "edge_std": e_std,
        "floor_delta": FLOOR_DELTA_FACTOR * max_step,
        "rms_step_disp": max_step,  # p99.9 스텝 변위 (노이즈 σ 검증용 척도)
        "eos_pos_weight": (n_neg / max(n_pos, 1)),
    }
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats.items()},
                  f, indent=1)
    logging.info("stats 저장: %s (floor_delta=%.4f, eos_pos_weight=%.2f)",
                 cache, stats["floor_delta"], stats["eos_pos_weight"])
    return stats


class RolloutDataset(Dataset):
    """1-step 학습 샘플. __getitem__ → dict[str, torch.Tensor]."""

    def __init__(self, h5_path: str, split: str, cfg: dict, stats: dict):
        assert int(cfg["graph"]["history"]) == G.HISTORY, "v0 은 history=1 만 지원"
        self.h5_path = h5_path
        self.split = split
        self.stats = stats
        self.noise_std = float(cfg["train"]["noise_std"]) if split == "train" else 0.0
        self.eos_eps = float(cfg["labels"]["eos_eps"])
        self.floor_delta = float(cfg["graph"]["floor_delta"] or stats["floor_delta"])
        self._h5: h5py.File | None = None  # worker 별 lazy open

        # 인덱스 구축 (한 번 열고 메타만 읽음 — positions 는 lazy)
        self.samples: list[tuple[str, int]] = []
        self.meta: dict[str, dict] = {}
        self.mesh_cache: dict[str, dict] = {}
        with h5py.File(h5_path, "r") as h5:
            rids = [x.decode() if isinstance(x, bytes) else x for x in h5[f"splits/{split}"][:]]
            for rid in rids:
                grp = h5[f"rollouts/{rid}"]
                T = grp["positions"].shape[0]
                self.meta[rid] = {
                    "dt_s": float(grp.attrs["dt_s"]),
                    "mesh": str(grp.attrs["mesh"]),
                    "eos": derive_eos(grp["ke"][:], self.eos_eps),
                }
                self.samples.extend((rid, t) for t in range(1, T - 1))
            for mid in {m["mesh"] for m in self.meta.values()}:
                mg = h5[f"meshes/{mid}"]
                self.mesh_cache[mid] = {
                    "edge_index": G.build_edge_index(mg["edges"][:]),
                    "node_type": mg["node_type"][:].astype(np.int64),
                }
        self._rng: np.random.Generator | None = None  # worker 별 lazy 시드 (아래 참고)

    def __len__(self) -> int:
        return len(self.samples)

    def _noise_rng(self) -> np.random.Generator:
        """DataLoader fork 시 부모 RNG 상태가 복제돼 워커들이 동일 노이즈를 뽑는
        문제 방지 — 첫 접근 시 torch 의 worker seed 로 워커별 독립 시드."""
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            seed = (info.seed if info is not None else torch.initial_seed()) % 2**32
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        rid, t = self.samples[i]
        meta = self.meta[rid]
        mesh = self.mesh_cache[meta["mesh"]]
        grp = self._h5[f"rollouts/{rid}"]
        window = grp["positions"][t - 1:t + 2].astype(np.float64)  # (3,N,3)
        rest = grp["rest_positions"][:].astype(np.float64)

        x_prev, x_curr, x_next = window
        if self.noise_std > 0:
            rng = self._noise_rng()
            x_prev = x_prev + rng.normal(0.0, self.noise_std, x_prev.shape)
            x_curr = x_curr + rng.normal(0.0, self.noise_std, x_curr.shape)
        y = x_next - 2.0 * x_curr + x_prev  # 노이즈 입력 기준, 클린 x_{t+1} 대상 (D9)

        node_feat, edge_feat = G.build_sample(
            x_prev, x_curr, rest, mesh["edge_index"], mesh["node_type"],
            meta["dt_s"], self.floor_delta,
        )
        node_feat, edge_feat = G.normalize_features(node_feat, edge_feat, self.stats)
        return {
            "node_feat": torch.from_numpy(node_feat),
            "edge_index": torch.from_numpy(mesh["edge_index"]),
            "edge_feat": torch.from_numpy(edge_feat),
            "target": torch.from_numpy(G.standardize_target(y, self.stats)),
            "eos_label": torch.tensor(meta["eos"][t], dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """노드 오프셋 그래프 배칭 — edge_index 에 노드 수 누적 오프셋."""
    node_feat = torch.cat([b["node_feat"] for b in batch])
    edge_feat = torch.cat([b["edge_feat"] for b in batch])
    target = torch.cat([b["target"] for b in batch])
    eos_label = torch.stack([b["eos_label"] for b in batch])
    edge_parts, batch_idx, offset = [], [], 0
    for gi, b in enumerate(batch):
        n = b["node_feat"].shape[0]
        edge_parts.append(b["edge_index"] + offset)
        batch_idx.append(torch.full((n,), gi, dtype=torch.int64))
        offset += n
    return {
        "node_feat": node_feat, "edge_feat": edge_feat,
        "edge_index": torch.cat(edge_parts, dim=1),
        "batch_idx": torch.cat(batch_idx),
        "target": target, "eos_label": eos_label,
    }


def iter_rollouts(h5_path: str, split: str, cfg: dict):
    """rollout.py 용 — rollout 전체 레코드를 하나씩 낸다."""
    eps = float(cfg["labels"]["eos_eps"])
    with h5py.File(h5_path, "r") as h5:
        rids = [x.decode() if isinstance(x, bytes) else x for x in h5[f"splits/{split}"][:]]
        for rid in rids:
            grp = h5[f"rollouts/{rid}"]
            mg = h5[f"meshes/{grp.attrs['mesh']}"]
            yield {
                "rid": rid,
                "mesh": str(grp.attrs["mesh"]),
                "positions": grp["positions"][:].astype(np.float64),
                "rest_positions": grp["rest_positions"][:].astype(np.float64),
                "edge_index": G.build_edge_index(mg["edges"][:]),
                "node_type": mg["node_type"][:].astype(np.int64),
                "dt_s": float(grp.attrs["dt_s"]),
                "eos": derive_eos(grp["ke"][:], eps),
                "action": json.loads(grp.attrs["action"]),
            }


def main():
    """자가 테스트: 샘플 셰이프/타이밍 출력."""
    from common import load_config, setup_logging

    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    stats = compute_stats(args.h5, cfg)
    ds = RolloutDataset(args.h5, "train", cfg, stats)
    t0 = time.perf_counter()
    s = ds[0]
    ms = (time.perf_counter() - t0) * 1e3
    for k, v in s.items():
        print(f"{k}: {tuple(v.shape)} {v.dtype}")
    b = collate([ds[0], ds[1]])
    print(f"collate: node_feat {tuple(b['node_feat'].shape)}, "
          f"edge_index {tuple(b['edge_index'].shape)}, eos {tuple(b['eos_label'].shape)}")
    print(f"샘플 {len(ds)}개, {ms:.1f} ms/sample")


if __name__ == "__main__":
    main()
