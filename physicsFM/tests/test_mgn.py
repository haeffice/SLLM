"""MeshGraphNetLite 단위 테스트 — shape/grad/checkpoint/국소성.

요약 흐름:
  1) 무작위 그래프(N=2000, E=6000 무방향 → edge_index (2,12000))에서
     forward (N,3), return_latent (N,hidden) shape 검증
  2) backward: 모든 파라미터에 grad 존재 + 전체적으로 0 아님
  3) grad_checkpoint=True: 같은 가중치에서 출력/grad 가 비체크포인트와 일치
  4) cuda 가용 시 forward+backward 후 peak VRAM 출력 (정보용)
  5) 메시지 전파 국소성: 고립 성분 {0,1}/{2,3} — 노드 0 변화가 노드 2 출력에
     영향 없음(누설 금지), 이웃 노드 1 출력은 반드시 변함
pytest와 `python tests/test_mgn.py` 양쪽에서 실행 가능.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import graph as G
from models.mgn import MeshGraphNetLite

SEED = 0
N_NODES = 2000
N_EDGES = 6000  # 무방향 — build_edge_index 후 2E=12000
HIDDEN = 64
LAYERS = 4


def _random_graph(seed: int = SEED):
    """무작위 그래프 텐서 (node_feat, edge_index, edge_feat) — 시드 고정 결정적."""
    rng = np.random.default_rng(seed)
    src = rng.integers(0, N_NODES, N_EDGES)
    dst = (src + rng.integers(1, N_NODES, N_EDGES)) % N_NODES  # self-loop 회피
    edges = np.stack([src, dst], axis=1).astype(np.int64)
    edge_index = torch.from_numpy(G.build_edge_index(edges))
    torch.manual_seed(seed)
    node_feat = torch.randn(N_NODES, G.NODE_FEAT_DIM)
    edge_feat = torch.randn(2 * N_EDGES, G.EDGE_FEAT_DIM)
    return node_feat, edge_index, edge_feat


def _make_model(grad_checkpoint: bool = False, seed: int = SEED) -> MeshGraphNetLite:
    torch.manual_seed(seed)
    return MeshGraphNetLite(node_in=8, edge_in=8, hidden=HIDDEN,
                            num_layers=LAYERS, grad_checkpoint=grad_checkpoint)


def test_forward_shapes():
    """forward (N,3), return_latent (N,hidden) — 두 경로 pred 동일."""
    node_feat, edge_index, edge_feat = _random_graph()
    assert edge_index.shape == (2, 2 * N_EDGES)
    assert int(edge_index.max()) < N_NODES
    model = _make_model().eval()
    with torch.no_grad():
        pred = model(node_feat, edge_index, edge_feat)
        assert pred.shape == (N_NODES, 3)
        pred2, latent = model(node_feat, edge_index, edge_feat, return_latent=True)
        assert latent.shape == (N_NODES, HIDDEN)
        assert torch.equal(pred, pred2)


def test_backward_all_grads():
    """loss.backward() 후 모든 파라미터 grad 존재 + 전체적으로 0 아님."""
    node_feat, edge_index, edge_feat = _random_graph()
    model = _make_model().train()
    pred = model(node_feat, edge_index, edge_feat)
    loss = (pred ** 2).mean()
    loss.backward()
    params = list(model.parameters())
    assert len(params) > 0
    assert all(p.grad is not None for p in params), "grad 없는 파라미터 존재"
    assert any(p.grad.abs().max().item() > 0.0 for p in params), "모든 grad 가 0"


def test_grad_checkpoint():
    """grad_checkpoint=True: 같은 가중치에서 출력/grad 가 비체크포인트와 일치."""
    node_feat, edge_index, edge_feat = _random_graph()
    base = _make_model(grad_checkpoint=False).train()
    ckpt = _make_model(grad_checkpoint=True).train()
    ckpt.load_state_dict(base.state_dict())  # 가중치 명시 복사

    pred_base = base(node_feat, edge_index, edge_feat)
    pred_ckpt = ckpt(node_feat, edge_index, edge_feat)
    assert pred_ckpt.shape == (N_NODES, 3)
    assert torch.allclose(pred_base, pred_ckpt, atol=1e-6), "체크포인트 출력 불일치"

    (pred_base ** 2).mean().backward()
    (pred_ckpt ** 2).mean().backward()
    for p_b, p_c in zip(base.parameters(), ckpt.parameters()):
        assert p_c.grad is not None
        assert torch.allclose(p_b.grad, p_c.grad, atol=1e-5), "체크포인트 grad 불일치"


def test_cuda_vram():
    """cuda 가용 시 forward+backward peak VRAM 출력 (정보용 — 값 assert 없음)."""
    if not torch.cuda.is_available():
        print("  (cuda 미가용 — VRAM 프로브 생략)")
        return
    node_feat, edge_index, edge_feat = _random_graph()
    dev = torch.device("cuda")
    model = _make_model().train().to(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    pred = model(node_feat.to(dev), edge_index.to(dev), edge_feat.to(dev))
    assert pred.shape == (N_NODES, 3)
    (pred ** 2).mean().backward()
    torch.cuda.synchronize(dev)
    print(f"  cuda peak {torch.cuda.max_memory_allocated(dev) / 1e6:.1f} MB "
          f"(N={N_NODES}, 2E={2 * N_EDGES}, hidden={HIDDEN}, layers={LAYERS})")


def test_message_locality():
    """고립 성분 간 누설 금지: 노드 0 변화 → 노드 2/3 출력 불변, 이웃 1 은 변함."""
    edges = np.array([[0, 1], [2, 3]], dtype=np.int64)  # {0,1} / {2,3} 두 성분
    edge_index = torch.from_numpy(G.build_edge_index(edges))
    torch.manual_seed(SEED)
    node_feat = torch.randn(4, G.NODE_FEAT_DIM)
    edge_feat = torch.randn(4, G.EDGE_FEAT_DIM)
    model = _make_model().eval()
    with torch.no_grad():
        out_a = model(node_feat, edge_index, edge_feat)
        node_b = node_feat.clone()
        node_b[0] += 1.0  # 노드 0 피처만 변경
        out_b = model(node_b, edge_index, edge_feat)
    # 단절 성분 {2,3} 은 완전 불변 (그래프 국소성)
    assert torch.allclose(out_a[2], out_b[2], atol=1e-7), "노드 0 → 노드 2 누설"
    assert torch.allclose(out_a[3], out_b[3], atol=1e-7), "노드 0 → 노드 3 누설"
    # 노드 0 자신과 1-hop 이웃 노드 1 은 반드시 변함 (≥1층 메시지 전파)
    assert (out_a[0] - out_b[0]).abs().max().item() > 1e-6
    assert (out_a[1] - out_b[1]).abs().max().item() > 1e-6, "이웃으로 메시지 미전파"


def main():
    """pytest 없이 직접 실행할 때 전체 테스트 수행."""
    tests = [
        test_forward_shapes,
        test_backward_all_grads,
        test_grad_checkpoint,
        test_cuda_vram,
        test_message_locality,
    ]
    for fn in tests:
        fn()
        print(f"{fn.__name__}: OK")
    print("OK")


if __name__ == "__main__":
    main()
