"""Azimuth-only SELD metrics (DCASE Task-3 joint metrics, 2-D variant).

Implements the four standard SELD metrics + the aggregated SELD error, adapted to
**azimuth-only** DOA (a 2-channel pair cannot observe elevation):

  * ER_20°  — location-dependent error rate (segment-based, 1-s segments).
  * F_20°   — location-dependent F-score; a TP needs class match AND angular
              distance < 20°.
  * LE_CD   — class-dependent localization error (mean azimuth error over
              class-matched pred/ref pairs).
  * LR_CD   — class-dependent localization recall.
  * ε_SELD  = (ER_20° + (1 − F_20°) + LE_CD/180 + (1 − LR_CD)) / 4.

Predictions are decoded from a Multi-ACCDOA tensor (activity = vector L2-norm >
threshold, azimuth = atan2(y, x)) with same-class track unification; matching of
predicted vs reference azimuths per (frame, class) uses the Hungarian algorithm
on circular angular distance. Reference: Mesaros et al. SELD metrics; DCASE
baseline `SELD_evaluation_metrics.py`.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

DOA_THRESHOLD_DEG = 20.0       # TP angular threshold (location-dependent F/ER)


def _circ_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute circular distance between azimuths (degrees), in [0, 180]."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _circ_mean(degs: list[float]) -> float:
    r = np.deg2rad(np.asarray(degs))
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def _unify(azimuths: list[float], angle_thr: float) -> list[float]:
    """Greedy same-class track unification: cluster azimuths within `angle_thr`
    and replace each cluster by its circular mean."""
    remaining = list(azimuths)
    clusters: list[float] = []
    while remaining:
        head = remaining.pop(0)
        group, rest = [head], []
        for other in remaining:
            (group if _circ_dist(np.array(head), np.array(other)) < angle_thr else rest).append(other)
        remaining = rest
        clusters.append(_circ_mean(group))
    return clusters


def compute_seld_metrics(pred: torch.Tensor, target: torch.Tensor, *,
                         activity_threshold: float = 0.5, unify_angle_deg: float = 30.0,
                         label_fps: int = 10, doa_threshold: float = DOA_THRESHOLD_DEG,
                         ) -> dict:
    """pred (B, T, N, C, 2) Multi-ACCDOA output, target (B, T, C, S=3, 2) reference.

    Returns {ER_20, F_20, LE_CD, LR_CD, SELD, n_ref, n_pred}.
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    tt = min(pred.shape[1], target.shape[1])
    pred, target = pred[:, :tt], target[:, :tt]
    b, t, n, c, _ = pred.shape
    seg_len = max(1, label_fps)

    err_sum, match_count, ref_total_cd = 0.0, 0, 0          # LE_CD / LR_CD
    seg = {}                                                # (batch, seg) -> [TP, FP, FN]
    n_ref_total = n_pred_total = 0

    pred_norm = np.linalg.norm(pred, axis=-1)               # (B, T, N, C)
    pred_az = np.rad2deg(np.arctan2(pred[..., 1], pred[..., 0]))
    ref_norm = np.linalg.norm(target, axis=-1)              # (B, T, C, S)
    ref_az = np.rad2deg(np.arctan2(target[..., 1], target[..., 0]))

    for bi in range(b):
        for ti in range(t):
            seg_key = (bi, ti // seg_len)
            cell = seg.setdefault(seg_key, [0, 0, 0])
            for ci in range(c):
                p_az = [pred_az[bi, ti, ni, ci] for ni in range(n)
                        if pred_norm[bi, ti, ni, ci] > activity_threshold]
                p_az = _unify(p_az, unify_angle_deg)
                r_az = [ref_az[bi, ti, ci, si] for si in range(target.shape[3])
                        if ref_norm[bi, ti, ci, si] > 0]
                n_pred_total += len(p_az)
                n_ref_total += len(r_az)
                ref_total_cd += len(r_az)
                if not p_az and not r_az:
                    continue
                tp = 0
                if p_az and r_az:
                    cost = _circ_dist(np.array(p_az)[:, None], np.array(r_az)[None, :])
                    ri, cj = linear_sum_assignment(cost)
                    for r_, c_ in zip(ri, cj):
                        d = cost[r_, c_]
                        err_sum += d
                        match_count += 1                    # class-dependent match (LR/LE)
                        if d < doa_threshold:
                            tp += 1                          # location-dependent TP
                fp, fn = len(p_az) - tp, len(r_az) - tp
                cell[0] += tp
                cell[1] += fp
                cell[2] += fn

    tp_all = sum(v[0] for v in seg.values())
    fp_all = sum(v[1] for v in seg.values())
    fn_all = sum(v[2] for v in seg.values())

    f_score = tp_all / (tp_all + 0.5 * (fp_all + fn_all) + 1e-9)
    sub = sum(min(v[1], v[2]) for v in seg.values())
    dele = sum(max(0, v[2] - v[1]) for v in seg.values())
    ins = sum(max(0, v[1] - v[2]) for v in seg.values())
    er = (sub + dele + ins) / max(n_ref_total, 1)
    le_cd = (err_sum / match_count) if match_count else 180.0
    lr_cd = match_count / max(ref_total_cd, 1)
    seld = (er + (1.0 - f_score) + le_cd / 180.0 + (1.0 - lr_cd)) / 4.0

    return {"ER_20": float(er), "F_20": float(f_score), "LE_CD": float(le_cd),
            "LR_CD": float(lr_cd), "SELD": float(seld),
            "n_ref": int(n_ref_total), "n_pred": int(n_pred_total)}
