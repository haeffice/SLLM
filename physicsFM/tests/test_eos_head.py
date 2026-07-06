"""EosHead 풀링 헤드 단위 테스트 — shape/순열 불변/배치-단건 일치.

요약 흐름:
  1) node_repr (N,16) + batch_idx (B=3) → 로짓 (3,); batch_idx=None → (1,)
  2) 순열 불변: 그래프 내 노드 셔플(및 전역 셔플 + batch_idx 동반 치환)에도
     로짓 동일 — mean+max 풀링의 구조 보장
  3) 배치 경로(index_add_/scatter_reduce_)와 단건 경로(mean/max)가 같은 로짓
pytest와 `python tests/test_eos_head.py` 양쪽에서 실행 가능.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.eos_head import EosHead

SEED = 0
DIM = 16
NODES = [7, 12, 5]  # B=3 그래프별 노드 수 (불균등 — bincount/평균 경로 검증)


def _make():
    """무작위 헤드 + 무작위 노드 표현/배치 인덱스 — 시드 고정 결정적."""
    torch.manual_seed(SEED)
    head = EosHead(DIM)
    node_repr = torch.randn(sum(NODES), DIM)
    batch_idx = torch.cat(
        [torch.full((n,), i, dtype=torch.int64) for i, n in enumerate(NODES)]
    )
    return head, node_repr, batch_idx


def test_shapes():
    """배치 로짓 (B,), 단건(batch_idx=None) 로짓 (1,)."""
    head, node_repr, batch_idx = _make()
    logits = head(node_repr, batch_idx)
    assert logits.shape == (len(NODES),)
    single = head(node_repr[: NODES[0]])
    assert single.shape == (1,)


def test_permutation_invariance():
    """노드 순서 셔플(batch_idx 동반 치환) → 로짓 동일 (1e-6 이내)."""
    head, node_repr, batch_idx = _make()
    base = head(node_repr, batch_idx)

    gen = torch.Generator().manual_seed(1)
    # 1) 한 그래프(두 번째, 노드 12개) 내부만 셔플 — batch_idx 는 그대로 유효
    lo, hi = NODES[0], NODES[0] + NODES[1]
    perm_in = torch.arange(sum(NODES))
    perm_in[lo:hi] = lo + torch.randperm(NODES[1], generator=gen)
    shuffled_in = head(node_repr[perm_in], batch_idx[perm_in])
    assert torch.allclose(base, shuffled_in, atol=1e-6), "그래프 내 셔플에 비불변"

    # 2) 전역 셔플 + batch_idx 동반 치환 (그래프 간 인터리브 포함)
    perm = torch.randperm(sum(NODES), generator=gen)
    shuffled = head(node_repr[perm], batch_idx[perm])
    assert torch.allclose(base, shuffled, atol=1e-6), "전역 셔플에 비불변"


def test_batched_vs_single():
    """배치 로짓 == 그래프별 단건 호출 로짓 (1e-6 이내)."""
    head, node_repr, batch_idx = _make()
    batched = head(node_repr, batch_idx)
    offset = 0
    for i, n in enumerate(NODES):
        single = head(node_repr[offset:offset + n])  # batch_idx=None 경로
        assert single.shape == (1,)
        assert torch.allclose(batched[i], single[0], atol=1e-6), \
            f"그래프 {i}: 배치 {batched[i].item():.6f} != 단건 {single[0].item():.6f}"
        offset += n


def main():
    """pytest 없이 직접 실행할 때 전체 테스트 수행."""
    tests = [
        test_shapes,
        test_permutation_invariance,
        test_batched_vs_single,
    ]
    for fn in tests:
        fn()
        print(f"{fn.__name__}: OK")
    print("OK")


if __name__ == "__main__":
    main()
