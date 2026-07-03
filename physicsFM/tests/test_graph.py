"""graph.py 피처 조립 계약 단위 테스트 — plate41 메쉬 기준.

요약 흐름:
  1) plate41 엣지 수 = 닫힌형 공식 2n(n−1) + (n−1)² (n=41 → 4880,
     수평+수직 2·41·40=3280 + 대각 40²=1600) — faces_to_edges 출력과 대조
  2) build_edge_index: (2,2E), 앞 절반 = edges.T, 뒤 절반 = 역방향, 인덱스 < N
  3) build_sample: 피처 차원 (N,8)/(2E,8), 속도 슬라이스 = (x_t−x_{t−1})/dt,
     floor_prox ∈ [0,1] 구간별 값, one-hot 행 합 = 1, 엣지 피처 열 순서
  4) 정규화/타깃 표준화 왕복: destandardize(standardize(y)) ≈ y
pytest와 `python tests/test_graph.py` 양쪽에서 실행 가능.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import graph as G
from meshes import PLATE_N, faces_to_edges, load_mesh

SEED = 0


def test_plate41_edge_count():
    """plate41: E = 2n(n−1) + (n−1)² = 4880 — 그리드 닫힌형과 레지스트리 일치."""
    mesh = load_mesh("plate41")
    n = PLATE_N
    expected_e = 2 * n * (n - 1) + (n - 1) ** 2  # 수평+수직 3280 + 대각 1600
    assert expected_e == 4880
    edges = faces_to_edges(mesh.faces)
    assert edges.shape == (expected_e, 2)
    assert np.array_equal(mesh.edges, edges), "레지스트리 edges != faces_to_edges 출력"
    # 그리드 인덱스 차분으로 방향별 개수 검증: +1 수평, +n 수직, +n+1 대각
    diff = edges[:, 1] - edges[:, 0]  # 행별 정렬돼 있어 항상 양수
    assert int((diff == 1).sum()) == n * (n - 1)          # 1640
    assert int((diff == n).sum()) == n * (n - 1)          # 1640
    assert int((diff == n + 1).sum()) == (n - 1) ** 2     # 1600


def test_build_edge_index():
    """양방향 edge_index: (2,2E), 앞 절반 = edges.T, 뒤 절반 = src/dst 반전."""
    mesh = load_mesh("plate41")
    e = len(mesh.edges)
    ei = G.build_edge_index(mesh.edges)
    assert ei.shape == (2, 2 * e)
    assert ei.dtype == np.int64
    assert np.array_equal(ei[:, :e], mesh.edges.T)
    assert np.array_equal(ei[:, e:], mesh.edges.T[::-1])
    n = len(mesh.vertices)
    assert ei.min() >= 0 and ei.max() < n


def test_build_sample():
    """피처 차원 + 속도/floor_prox/one-hot/엣지 열 순서 불변식."""
    mesh = load_mesh("plate41")
    rng = np.random.default_rng(SEED)
    n = len(mesh.vertices)
    ei = G.build_edge_index(mesh.edges)
    dt_s, delta = 0.012, 0.05
    rest = mesh.vertices.copy()
    pos_prev = rest + rng.normal(0.0, 0.01, (n, 3))
    pos_curr = rest + rng.normal(0.0, 0.01, (n, 3))
    pos_curr[:, 2] = rng.uniform(-0.02, 0.2, n)  # 바닥 아래/근접/포화 세 구간 포함
    node_type = rng.integers(0, G.NUM_NODE_TYPES, n)

    node_feat, edge_feat = G.build_sample(
        pos_prev, pos_curr, rest, ei, node_type, dt_s, delta
    )
    assert G.NODE_FEAT_DIM == 8 and G.EDGE_FEAT_DIM == 8
    assert node_feat.shape == (n, G.NODE_FEAT_DIM)
    assert edge_feat.shape == (2 * len(mesh.edges), G.EDGE_FEAT_DIM)
    assert node_feat.dtype == np.float32 and edge_feat.dtype == np.float32

    # 속도 슬라이스 == (x_t − x_{t−1})/dt (f64 계산 후 f32 캐스팅까지 동일)
    vel = ((pos_curr - pos_prev) / dt_s).astype(np.float32)
    assert np.array_equal(node_feat[:, 0:3], vel)

    # floor_prox: [0,1] 범위, z ≥ δ → 1, 0 ≤ z < δ → z/δ, z < 0 → 0
    fp = node_feat[:, 3]
    z = pos_curr[:, 2]
    assert fp.min() >= 0.0 and fp.max() <= 1.0
    assert np.all(fp[z >= delta] == 1.0)
    below = (z >= 0.0) & (z < delta)
    assert below.any() and (z >= delta).any() and (z < 0.0).any()  # 세 구간 모두 표집
    assert np.array_equal(fp[below], (z[below] / delta).astype(np.float32))
    assert np.all(fp[z < 0.0] == 0.0)

    # fragility one-hot: 행 합 1, 지정 클래스 위치만 1
    onehot = node_feat[:, 4:8]
    assert np.all(onehot.sum(axis=1) == 1.0)
    assert np.array_equal(onehot.argmax(axis=1), node_type)

    # 엣지 피처 열 순서: Δu_rest(3)+‖Δu_rest‖ + Δx(3)+‖Δx‖
    src, dst = ei[0], ei[1]
    du = rest[src] - rest[dst]
    dx = pos_curr[src] - pos_curr[dst]
    assert np.array_equal(edge_feat[:, 0:3], du.astype(np.float32))
    assert np.array_equal(edge_feat[:, 4:7], dx.astype(np.float32))
    assert np.allclose(edge_feat[:, 3], np.linalg.norm(du, axis=1), atol=1e-6)
    assert np.allclose(edge_feat[:, 7], np.linalg.norm(dx, axis=1), atol=1e-6)


def test_normalize_roundtrip():
    """정규화/표준화 왕복 — destandardize(standardize(y)) ≈ y, 수동 역변환 원복."""
    rng = np.random.default_rng(SEED)
    stats = {
        "vel_mean": rng.normal(0.0, 1.0, 3), "vel_std": rng.uniform(0.5, 2.0, 3),
        "edge_mean": rng.normal(0.0, 1.0, G.EDGE_FEAT_DIM),
        "edge_std": rng.uniform(0.5, 2.0, G.EDGE_FEAT_DIM),
        "target_mean": rng.normal(0.0, 1e-3, 3), "target_std": rng.uniform(1e-4, 1e-3, 3),
    }
    node = rng.normal(0.0, 1.0, (50, G.NODE_FEAT_DIM)).astype(np.float32)
    edge = rng.normal(0.0, 1.0, (120, G.EDGE_FEAT_DIM)).astype(np.float32)

    node_n, edge_n = G.normalize_features(node, edge, stats)
    # 사본 반환(원본 불변) + 속도 슬라이스 외 노드 피처는 그대로
    assert node_n is not node
    assert np.array_equal(node_n[:, 3:], node[:, 3:])
    assert np.allclose(node_n[:, 0:3] * stats["vel_std"] + stats["vel_mean"],
                       node[:, 0:3], atol=1e-6)
    assert np.allclose(edge_n * stats["edge_std"] + stats["edge_mean"], edge, atol=1e-5)

    y = rng.normal(0.0, 1e-3, (50, 3))  # 가속도형 타깃 스케일
    y_round = G.destandardize_target(G.standardize_target(y, stats), stats)
    assert np.allclose(y_round, y, atol=1e-8)


def main():
    """pytest 없이 직접 실행할 때 전체 테스트 수행."""
    tests = [
        test_plate41_edge_count,
        test_build_edge_index,
        test_build_sample,
        test_normalize_roundtrip,
    ]
    for fn in tests:
        fn()
        print(f"{fn.__name__}: OK")
    print("OK")


if __name__ == "__main__":
    main()
