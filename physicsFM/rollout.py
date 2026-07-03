"""N-step autoregressive 롤아웃 평가 CLI — "초기 상태 + action → 상태열 → eos".

요약 흐름: GT 의 첫 두 프레임(x_0, x_1)을 시드로, 매 스텝 graph.build_sample 로
피처를 재조립해 가속도형 변위를 예측·적분한다:
  x̂_{t+1} = 2·x̂_t − x̂_{t−1} + destandardize(ŷ)
eos 헤드의 sigmoid 가 threshold 를 patience 스텝 연속 넘기면 그 시점을 예측
정지 스텝으로 기록한다(지표 비교를 위해 롤아웃 자체는 GT 길이까지 계속).

지표: pos RMSE@{1,10,50,full}, 최종 프레임 RMSE, eos 스텝 오차(예측 − GT 첫 eos).
--save-frames 로 예측 (T,N,3) 을 npz(key=frames) 저장 — physics fe 뷰어 규약과 동일.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import graph as G
from common import load_config, setup_logging
from dataset import iter_rollouts
from models import get_model
from models.eos_head import EosHead
from train import load_stats_from_ckpt

RMSE_STEPS = (1, 10, 50)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


@torch.no_grad()
def rollout_one(model, eos_head, rec: dict, cfg: dict, stats: dict, device: str) -> dict:
    pos_gt = rec["positions"]  # (T,N,3)
    T = pos_gt.shape[0]
    dt_s, rest = rec["dt_s"], rec["rest_positions"]
    edge_index_np = rec["edge_index"]
    edge_index = torch.from_numpy(edge_index_np).to(device)
    floor_delta = float(cfg["graph"]["floor_delta"] or stats["floor_delta"])
    thr = float(cfg["rollout"]["eos_threshold"])
    patience = int(cfg["rollout"]["eos_patience"])

    preds = np.empty_like(pos_gt)
    preds[0], preds[1] = pos_gt[0], pos_gt[1]
    x_prev, x_curr = pos_gt[0].copy(), pos_gt[1].copy()
    eos_run, eos_step_pred = 0, None

    for t in range(1, T - 1):
        node_feat, edge_feat = G.build_sample(
            x_prev, x_curr, rest, edge_index_np, rec["node_type"], dt_s, floor_delta)
        node_feat, edge_feat = G.normalize_features(node_feat, edge_feat, stats)
        nf = torch.from_numpy(node_feat).to(device)
        ef = torch.from_numpy(edge_feat).to(device)
        pred, latent = model(nf, edge_index, ef, return_latent=True)
        y = G.destandardize_target(pred.float().cpu().numpy(), stats)
        x_next = 2.0 * x_curr - x_prev + y
        preds[t + 1] = x_next
        x_prev, x_curr = x_curr, x_next

        p_eos = torch.sigmoid(eos_head(latent)).item()
        eos_run = eos_run + 1 if p_eos > thr else 0
        if eos_run >= patience and eos_step_pred is None:
            # 루프 t 에서 헤드는 프레임 t 를 분류(학습 라벨 eos[t] 와 동일 규약).
            # patience 연속 런의 시작 = 정지 온셋 → +patience 편향 제거.
            eos_step_pred = t - patience + 1

    gt_eos_idx = np.flatnonzero(rec["eos"] > 0.5)
    gt_eos = int(gt_eos_idx[0]) if len(gt_eos_idx) else T
    out = {
        "rid": rec["rid"], "mesh": rec["mesh"], "T": T,
        "final_rmse": _rmse(preds[-1], pos_gt[-1]),
        "full_rmse": _rmse(preds, pos_gt),
        "eos_pred": eos_step_pred if eos_step_pred is not None else -1,
        "eos_gt": gt_eos,
        "eos_err": (eos_step_pred - gt_eos) if eos_step_pred is not None else np.nan,
    }
    for k in RMSE_STEPS:
        # 시드 이후 k 스텝 시점 (t = 1+k)
        out[f"rmse@{k}"] = _rmse(preds[1 + k], pos_gt[1 + k]) if 1 + k < T else np.nan
    return out, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5", default=None, help="기본: ckpt 의 config data.h5")
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default=None, help="기본: ckpt 에 저장된 config")
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="rollout 수 제한 (0=전체)")
    args = ap.parse_args()
    setup_logging()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = load_config(args.config) if args.config else ckpt["cfg"]
    stats = load_stats_from_ckpt(ckpt)
    h5_path = args.h5 or cfg["data"]["h5"]

    model, eos_in_dim = get_model(cfg["model"])
    saved_backbone = ckpt.get("backbone_class", type(model).__name__)
    if saved_backbone != type(model).__name__:
        raise RuntimeError(
            f"체크포인트 백본({saved_backbone}) ≠ 현재 환경 백본({type(model).__name__}) — "
            "physicsnemo 설치 여부가 학습 시점과 다릅니다.")
    model = model.to(device).eval()
    model.load_state_dict(ckpt["model"])
    eos_head = EosHead(eos_in_dim).to(device).eval()
    eos_head.load_state_dict(ckpt["eos_head"])

    run_dir = Path(args.ckpt).parent
    frames_dir = run_dir / "rollout_frames"
    if args.save_frames:
        frames_dir.mkdir(exist_ok=True)

    rows, agg = [], defaultdict(list)
    for i, rec in enumerate(iter_rollouts(h5_path, args.split, cfg)):
        if args.limit and i >= args.limit:
            break
        out, preds = rollout_one(model, eos_head, rec, cfg, stats, device)
        rows.append(out)
        agg[out["mesh"]].append(out)
        if args.save_frames:
            np.savez_compressed(frames_dir / f"{out['rid']}.npz",
                                frames=preds.astype(np.float32))
        logging.info("%s (%s T=%d): rmse@1 %.2e @10 %.2e @50 %.2e full %.2e "
                     "eos %s/%d", out["rid"], out["mesh"], out["T"], out["rmse@1"],
                     out["rmse@10"], out.get("rmse@50", np.nan), out["full_rmse"],
                     out["eos_pred"], out["eos_gt"])

    csv_path = run_dir / f"rollout_{args.split}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\n== {args.split} 롤아웃 요약 (per mesh) ==")
    print(f"{'mesh':14s} {'n':>3s} {'rmse@1':>9s} {'rmse@10':>9s} {'full':>9s} "
          f"{'final':>9s} {'|eos_err|':>9s}")
    for mesh, outs in sorted(agg.items()):
        eos_errs = [abs(o["eos_err"]) for o in outs if np.isfinite(o["eos_err"])]
        print(f"{mesh:14s} {len(outs):3d} "
              f"{np.nanmean([o['rmse@1'] for o in outs]):9.2e} "
              f"{np.nanmean([o['rmse@10'] for o in outs]):9.2e} "
              f"{np.mean([o['full_rmse'] for o in outs]):9.2e} "
              f"{np.mean([o['final_rmse'] for o in outs]):9.2e} "
              f"{np.mean(eos_errs) if eos_errs else float('nan'):9.1f}")
    print(f"→ {csv_path}")


if __name__ == "__main__":
    main()
