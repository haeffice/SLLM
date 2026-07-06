"""MGN 1-step 학습 CLI — bf16 autocast, CSV 로깅, 체크포인트/재개.

요약 흐름: RolloutDataset(1-step, 노이즈 주입) → MeshGraphNetLite(기본) 로
가속도형 타깃 MSE + 보조 eos BCE(pos_weight 자동) 를 step 기반으로 최적화.
lr 은 ExponentialLR 로 steps 에 걸쳐 lr→lr_min. 로그는 runs/<run>/metrics.csv.

사용 예:
  .venv/bin/python train.py --run smoke --set train.steps=50 --set data.h5=data/smoke.h5
  .venv/bin/python train.py --run mgn_h128_l15            # 본 학습
  .venv/bin/python train.py --run mgn_h128_l15 --resume   # 재개
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import load_config, setup_logging
from dataset import RolloutDataset, collate, compute_stats
from models import get_model
from models.eos_head import EosHead


def save_checkpoint(path: Path, model, eos_head, opt, sched, stats: dict,
                    cfg: dict, step: int) -> None:
    payload = {
        "model": model.state_dict(),
        "eos_head": eos_head.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "stats": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats.items()},
        "cfg": cfg,
        "step": step,
        "backbone_class": type(model).__name__,  # 폴백 백본 불일치 검출용
    }
    tmp = path.with_suffix(".tmp")  # 원자적 교체 — 저장 중 크래시에도 latest.pt 보존
    torch.save(payload, tmp)
    tmp.replace(path)


def load_stats_from_ckpt(ckpt: dict) -> dict:
    return {k: np.asarray(v) if isinstance(v, list) else v for k, v in ckpt["stats"].items()}


@torch.no_grad()
def evaluate_1step(model, eos_head, loader, device, amp: bool, max_batches: int = 50) -> dict:
    """valid split 1-step MSE / eos BCE (정규화 단위)."""
    model.eval(); eos_head.eval()
    mse_sum = bce_sum = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred, latent = model(batch["node_feat"], batch["edge_index"],
                                 batch["edge_feat"], return_latent=True)
            logit = eos_head(latent, batch["batch_idx"])
        mse_sum += F.mse_loss(pred.float(), batch["target"]).item()
        bce_sum += F.binary_cross_entropy_with_logits(
            logit.float(), batch["eos_label"]).item()
        n += 1
    model.train(); eos_head.train()
    return {"val_mse": mse_sum / max(n, 1), "val_eos_bce": bce_sum / max(n, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run", required=True, help="runs/<run>/ 이름")
    ap.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VAL")
    ap.add_argument("--resume", action="store_true", help="runs/<run>/latest.pt 에서 재개")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.overrides)
    tr = cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(tr["amp"]) and device == "cuda"

    run_dir = Path("runs") / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    h5_path = cfg["data"]["h5"]
    resume_ckpt = None
    if args.resume:
        # 재개 시 stats 는 반드시 체크포인트 것 — 재계산하면 가중치와 정규화 불일치
        resume_ckpt = torch.load(run_dir / "latest.pt", map_location="cpu",
                                 weights_only=False)
        stats = load_stats_from_ckpt(resume_ckpt)
    else:
        stats = compute_stats(h5_path, cfg)

    train_ds = RolloutDataset(h5_path, "train", cfg, stats)
    valid_ds = RolloutDataset(h5_path, "valid", cfg, stats)
    # 검증은 max_batches 로 앞부분만 보므로, 고정 시드 셔플로 메쉬/시점 대표성 확보
    np.random.default_rng(0).shuffle(valid_ds.samples)
    workers = int(tr["num_workers"])
    train_loader = DataLoader(
        train_ds, batch_size=int(tr["batch_size"]), shuffle=True, collate_fn=collate,
        num_workers=workers, pin_memory=(device == "cuda"),
        persistent_workers=workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=int(tr["batch_size"]), shuffle=False, collate_fn=collate,
        num_workers=0,
    )

    model, eos_in_dim = get_model(cfg["model"])
    model = model.to(device)
    eos_head = EosHead(eos_in_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("backbone=%s params=%.2fM device=%s amp=%s 샘플 %d/%d(train/valid)",
                 type(model).__name__, n_params / 1e6, device, amp,
                 len(train_ds), len(valid_ds))

    steps = int(tr["steps"])
    opt = torch.optim.Adam(
        list(model.parameters()) + list(eos_head.parameters()), lr=float(tr["lr"])
    )
    gamma = (float(tr["lr_min"]) / float(tr["lr"])) ** (1.0 / max(steps, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    pos_weight = torch.tensor(float(stats["eos_pos_weight"]), device=device)
    eos_w = float(tr["eos_weight"])

    step = 0
    if resume_ckpt is not None:
        saved_backbone = resume_ckpt.get("backbone_class", type(model).__name__)
        if saved_backbone != type(model).__name__:
            raise RuntimeError(
                f"체크포인트 백본({saved_backbone}) ≠ 현재 백본({type(model).__name__}) — "
                "환경 폴백으로 아키텍처가 바뀌었습니다. model.backbone 설정을 확인하세요.")
        model.load_state_dict(resume_ckpt["model"])
        eos_head.load_state_dict(resume_ckpt["eos_head"])
        opt.load_state_dict(resume_ckpt["opt"])
        sched.load_state_dict(resume_ckpt["sched"])
        step = int(resume_ckpt["step"])
        logging.info("재개: step %d 부터", step)

    csv_path = run_dir / "metrics.csv"
    if resume_ckpt is not None and csv_path.exists():
        # 마지막 체크포인트 이후(크래시 전) 행 제거 — 재개 시 중복/모순 로그 방지
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        kept = [rows[0]] + [r for r in rows[1:] if r and int(r[0]) <= step]
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerows(kept)
    new_csv = not csv_path.exists()
    csv_f = open(csv_path, "a", newline="")
    writer = csv.writer(csv_f)
    if new_csv:
        writer.writerow(["step", "loss", "mse", "eos_bce", "val_mse", "val_eos_bce",
                         "lr", "sec_per_step", "vram_mb"])

    model.train(); eos_head.train()
    t_last = time.perf_counter()
    step_last_logged = step
    done = False
    while not done:
        for batch in train_loader:
            if step >= steps:
                done = True
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pred, latent = model(batch["node_feat"], batch["edge_index"],
                                     batch["edge_feat"], return_latent=True)
                logit = eos_head(latent, batch["batch_idx"])
                mse = F.mse_loss(pred.float(), batch["target"])
                eos_bce = F.binary_cross_entropy_with_logits(
                    logit.float(), batch["eos_label"], pos_weight=pos_weight)
                loss = mse + eos_w * eos_bce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1

            if step % int(tr["log_every"]) == 0 or step == 1:
                dt_step = (time.perf_counter() - t_last) / max(step - step_last_logged, 1)
                t_last = time.perf_counter()
                step_last_logged = step
                vram = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0
                row = [step, f"{loss.item():.6f}", f"{mse.item():.6f}",
                       f"{eos_bce.item():.6f}", "", "", f"{sched.get_last_lr()[0]:.2e}",
                       f"{dt_step:.3f}", f"{vram:.0f}"]
                writer.writerow(row); csv_f.flush()
                logging.info("step %d loss %.5f mse %.5f eos %.4f (%.3fs/step, %.0fMB)",
                             step, loss.item(), mse.item(), eos_bce.item(), dt_step, vram)
            if step % int(tr["val_every"]) == 0:
                val = evaluate_1step(model, eos_head, valid_loader, device, amp)
                writer.writerow([step, "", "", "", f"{val['val_mse']:.6f}",
                                 f"{val['val_eos_bce']:.6f}", "", "", ""])
                csv_f.flush()
                logging.info("step %d VAL mse %.6f eos %.4f",
                             step, val["val_mse"], val["val_eos_bce"])
                t_last = time.perf_counter(); step_last_logged = step  # 평가 시간 제외
            if step % int(tr["ckpt_every"]) == 0:
                save_checkpoint(run_dir / f"ckpt_{step}.pt", model, eos_head, opt,
                                sched, stats, cfg, step)
                save_checkpoint(run_dir / "latest.pt", model, eos_head, opt,
                                sched, stats, cfg, step)
                t_last = time.perf_counter(); step_last_logged = step  # 저장 시간 제외

    save_checkpoint(run_dir / "latest.pt", model, eos_head, opt, sched, stats, cfg, step)
    val = evaluate_1step(model, eos_head, valid_loader, device, amp)
    logging.info("완료: step %d, 최종 VAL mse %.6f eos %.4f → %s",
                 step, val["val_mse"], val["val_eos_bce"], run_dir / "latest.pt")
    csv_f.close()


if __name__ == "__main__":
    main()
