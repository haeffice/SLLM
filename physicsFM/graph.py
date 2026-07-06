"""그래프 피처 조립 — dataset.py(학습)와 rollout.py(평가)가 공유하는 단일 계약.

요약 흐름:
  1) 메쉬 무방향 엣지(E,2) → 양방향 edge_index (2,2E)
  2) 노드 피처 (N,8): 속도 v_t(3) + floor_prox(1) + fragility one-hot(4)
     - v_t = (x_t − x_{t−1}) / dt  (후방 차분, mesh-units/s)
     - floor_prox = clip(z, 0, δ)/δ — 바닥(z=0 평면) 근접도. δ를 넘는 높이는
       포화시켜 절대 높이에 의존하지 못하게 한다(낙하 높이 일반화).
  3) 엣지 피처 (2E,8): Δu_rest(3) + ‖Δu_rest‖ + Δx(3) + ‖Δx‖
     (DeepMind MGN cloth 레시피 — mesh-space + world-space 상대 기하)
  4) 정규화: 속도 슬라이스/엣지 8차원/타깃(가속도형)을 stats(사이드카 json)로 표준화.
     one-hot·floor_prox는 이미 [0,1]이라 그대로 둔다.

world edges(다물체 접촉용)는 v0에서 불필요(바닥 = 해석적 평면 → floor_prox로 충분,
자기접촉 없음). 확장 시 build_sample 내 표시 지점에서 edge_index/edge_feat 뒤에
concat 하고 is_world one-hot 차원을 추가한다.
"""

from __future__ import annotations

import numpy as np

HISTORY = 1  # 속도 히스토리 창 (v0: v_t 하나 — DeepMind cloth 방식)
NUM_NODE_TYPES = 4  # meshes.FRAGILITY_CLASSES 크기와 일치해야 함
NODE_FEAT_DIM = 3 * HISTORY + 1 + NUM_NODE_TYPES  # 8
EDGE_FEAT_DIM = 8  # Δu_rest(3)+‖Δu_rest‖+Δx(3)+‖Δx‖


def build_edge_index(edges: np.ndarray) -> np.ndarray:
    """무방향 유니크 엣지 (E,2) → 양방향 edge_index (2,2E) i64.

    edge_index[0]=src, edge_index[1]=dst. 메시지는 dst로 집계된다.
    """
    edges = np.asarray(edges, dtype=np.int64)
    return np.concatenate([edges.T, edges.T[::-1]], axis=1)


def build_sample(
    pos_prev: np.ndarray,
    pos_curr: np.ndarray,
    rest_positions: np.ndarray,
    edge_index: np.ndarray,
    node_type: np.ndarray,
    dt_s: float,
    floor_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """한 시점의 (node_feat (N,8), edge_feat (2E,8)) 원시(비정규화) 피처.

    pos_prev/pos_curr: (N,3) — x_{t−1}, x_t. rest_positions: (N,3) vrot(변형 0 기준).
    """
    vel = (pos_curr - pos_prev) / dt_s  # (N,3) mesh-units/s
    floor_prox = np.clip(pos_curr[:, 2], 0.0, floor_delta) / floor_delta  # (N,)
    onehot = np.zeros((pos_curr.shape[0], NUM_NODE_TYPES), dtype=np.float64)
    onehot[np.arange(pos_curr.shape[0]), np.asarray(node_type, dtype=np.int64)] = 1.0
    node_feat = np.concatenate([vel, floor_prox[:, None], onehot], axis=1)

    src, dst = edge_index[0], edge_index[1]
    dx = pos_curr[src] - pos_curr[dst]  # (2E,3) world-space 상대 위치
    du = rest_positions[src] - rest_positions[dst]  # (2E,3) mesh-space 상대 위치
    edge_feat = np.concatenate(
        [du, np.linalg.norm(du, axis=1, keepdims=True),
         dx, np.linalg.norm(dx, axis=1, keepdims=True)],
        axis=1,
    )
    # (world edges 확장 지점: 여기서 edge_index/edge_feat 에 concat)
    return node_feat.astype(np.float32), edge_feat.astype(np.float32)


# --- 정규화 (stats = dataset.compute_stats 결과 dict) ---------------------------

def normalize_features(node_feat: np.ndarray, edge_feat: np.ndarray, stats: dict):
    """속도 슬라이스(per-axis)와 엣지 8차원을 표준화한 사본을 돌려준다."""
    node = node_feat.copy()
    node[:, 0:3] = (node[:, 0:3] - stats["vel_mean"]) / stats["vel_std"]
    edge = (edge_feat - stats["edge_mean"]) / stats["edge_std"]
    return node, edge.astype(np.float32)


def standardize_target(y: np.ndarray, stats: dict) -> np.ndarray:
    """가속도형 타깃 y = x_{t+1} − 2x_t + x_{t−1} 를 per-axis 표준화."""
    return ((y - stats["target_mean"]) / stats["target_std"]).astype(np.float32)


def destandardize_target(y_norm: np.ndarray, stats: dict) -> np.ndarray:
    """표준화 타깃 → 원 단위 가속도형 변위 (rollout 적분용)."""
    return y_norm * stats["target_std"] + stats["target_mean"]
