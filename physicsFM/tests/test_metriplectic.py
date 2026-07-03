"""MetriplecticHead 구조 보장 단위 테스트 — 무작위 가중치에서도 성립해야 한다.

요약 흐름:
  float64로 무작위 초기화 헤드 + 무작위 z(B,D)를 만들고,
  1) 에너지 보존 |dE/dt| ≈ 0  2) 엔트로피 생성 dS/dt ≥ 0
  3) 구조 직교성 Qᵀg_E = Pᵀg_S = 0  4) M = QQᵀ 대칭 작용/PSD
  5) Euler 스텝의 O(η²) 에너지 오차 수렴  6) rk4 장기 에너지 드리프트
  7) 출력 shape (B=1 포함)
  를 확인한다. pytest와 `python tests/test_metriplectic.py` 양쪽에서 실행 가능.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.metriplectic import MetriplecticHead

torch.set_default_dtype(torch.float64)

# 테스트 규모 — 구조 보장은 차원/랭크와 무관하지만 대표값으로 고정.
DIM = 32
RANK = 8
BATCH = 64
SEED = 0

# 허용 오차 — float64 기계 정밀도 여유분.
TOL_CONS = 1e-10   # 상대 에너지 보존
TOL_ORTHO = 1e-10  # 구조 직교성
TOL_NEG = 1e-12    # 음수 허용 하한 (부동소수점 잡음)


def _make(batch: int = BATCH):
    """무작위(기본 초기화) 헤드 + 무작위 z — 시드 고정으로 결정적."""
    torch.manual_seed(SEED)
    head = MetriplecticHead(DIM, rank=RANK)
    z = torch.randn(batch, DIM)
    return head, z


def test_conservation():
    """에너지 보존: |dE/dt| ≈ 0 (스케일 안전한 상대 오차)."""
    head, z = _make()
    out = head(z)
    _, _, g_E, _ = head._operators(z)
    denom = g_E.norm(dim=-1) * out["dz_dt"].norm(dim=-1) + 1e-12
    rel = (out["dE_dt"].abs() / denom).max().item()
    assert rel < TOL_CONS, f"에너지 보존 위반: 상대 |dE/dt| = {rel:.3e}"


def test_dissipation():
    """엔트로피 생성: dS/dt ≥ 0 (모든 배치 원소)."""
    head, z = _make()
    ds = head(z)["dS_dt"]
    assert (ds >= -TOL_NEG).all(), f"dS/dt 최소값 = {ds.min().item():.3e}"


def test_structural():
    """구조 직교성: Qᵀg_E = 0, Pᵀg_S = 0 (열 단위, 기계 정밀도)."""
    head, z = _make()
    Q, P, g_E, g_S = head._operators(z)
    qe = torch.einsum("bdr,bd->br", Q, g_E).abs().max().item()
    ps = torch.einsum("bdr,bd->br", P, g_S).abs().max().item()
    assert qe < TOL_ORTHO, f"max |Qᵀg_E| = {qe:.3e}"
    assert ps < TOL_ORTHO, f"max |Pᵀg_S| = {ps:.3e}"


def test_psd():
    """M = QQᵀ 대칭 작용 <x, My> = <Mx, y> 및 xᵀMx ≥ 0."""
    head, z = _make()
    Q, _, _, _ = head._operators(z)
    Q = Q.detach()
    x, y = torch.randn(BATCH, DIM), torch.randn(BATCH, DIM)

    def apply_m(v):
        return torch.einsum("bdr,br->bd", Q, torch.einsum("bdr,bd->br", Q, v))

    x_my = (x * apply_m(y)).sum(-1)
    mx_y = (apply_m(x) * y).sum(-1)
    rel = ((x_my - mx_y).abs() / (x_my.abs() + mx_y.abs() + 1e-12)).max().item()
    assert rel < TOL_ORTHO, f"대칭 작용 위반: {rel:.3e}"
    x_mx = (x * apply_m(x)).sum(-1)
    assert (x_mx >= -TOL_NEG).all(), f"xᵀMx 최소값 = {x_mx.min().item():.3e}"


def test_integration_order():
    """Euler 한 스텝의 에너지 오차는 O(η²) — η 반감 시 ~4배 감소, S는 비감소.

    η는 O(η²) 신호가 float64 잡음 바닥(≈1e-16·|E|) 위로 충분히(≥100배) 올라오게
    선택한다 — 기본 초기화에서 |dz/dt| ≈ 4e-5라 η=1e-3이면 신호가 잡음에 묻힌다.
    """
    head, z = _make()
    e0 = head.energy(z).detach()
    s0 = head.entropy(z).detach()
    errs = []
    for eta in (1e-1, 5e-2):
        z1 = head.step(z, eta, method="euler").detach()
        errs.append((head.energy(z1) - e0).abs().mean().item())
        assert (head.entropy(z1) >= s0 - 1e-9).all(), f"η={eta}에서 S 감소"
    ratio = errs[0] / (errs[1] + 1e-300)
    assert ratio > 3.0, f"O(η²) 수렴 실패: 오차비 = {ratio:.3f} (기대 ~4)"


def test_rk4_energy_drift():
    """rk4 200스텝(dt=1e-2) 상대 에너지 드리프트 < 1e-4."""
    head, z = _make()
    e0 = head.energy(z).detach()
    for _ in range(200):
        # 그래프가 스텝마다 쌓이지 않도록 detach (구조 보장 검증에는 무관).
        z = head.step(z, 1e-2, method="rk4").detach()
    drift = ((head.energy(z).detach() - e0).abs() / (e0.abs() + 1e-12)).max().item()
    assert drift < 1e-4, f"rk4 에너지 드리프트 = {drift:.3e}"


def test_shapes():
    """출력 dict shape 검증 — B=64와 B=1 모두."""
    for batch in (BATCH, 1):
        head, z = _make(batch)
        out = head(z)
        assert out["dz_dt"].shape == (batch, DIM)
        for key in ("E", "S", "dE_dt", "dS_dt"):
            assert out[key].shape == (batch,), f"{key} shape = {tuple(out[key].shape)}"
        assert head.step(z, 1e-3, method="euler").shape == (batch, DIM)
        assert head.step(z, 1e-3, method="rk4").shape == (batch, DIM)


def main():
    """pytest 없이 직접 실행할 때 전체 테스트 수행."""
    tests = [
        test_conservation,
        test_dissipation,
        test_structural,
        test_psd,
        test_integration_order,
        test_rk4_energy_drift,
        test_shapes,
    ]
    for fn in tests:
        fn()
        print(f"{fn.__name__}: OK")
    print("OK")


if __name__ == "__main__":
    main()
