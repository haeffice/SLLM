"""GENERIC(metriplectic) 구조 잠재 동역학 헤드 — dz/dt = L(z)∇E + M(z)∇S (순수 torch).

요약 흐름:
  z (B,D) → E(z), S(z) 스칼라 MLP → g_E, g_S = autograd 기울기 (create_graph=True)
  → 반대칭 이차형식으로 기울기에 "정확히" 직교하는 열들을 구성 (GFINN 방식,
    나눗셈·정규화·사영(P = I − ĝĝᵀ) 일절 없음):
      Q: q_i = u_i(v_iᵀg_E) − v_i(u_iᵀg_E)  ⇒ q_iᵀg_E = 0 (반대칭형 소거)
      P: p_i = a_i(c_iᵀg_S) − c_i(a_iᵀg_S)  ⇒ p_iᵀg_S = 0
  → L = PΩPᵀ (Ω = ω − ωᵀ 반대칭 → L 반대칭), M = QQᵀ (PSD)
  → dz/dt = L g_E + M g_S

구조 보장 (부동소수점 기계 정밀도 수준, 학습 파라미터와 무관하게 항상 성립):
  dE/dt = g_Eᵀ(L g_E) + (Qᵀg_E)ᵀ(Qᵀg_S) = 0 + 0        # 에너지 보존
  dS/dt = (Pᵀg_S)ᵀΩ(Pᵀg_E) + ‖Qᵀg_S‖²   = 0 + (≥ 0)    # 엔트로피 생성
  퇴화 조건: L g_S = PΩ(Pᵀg_S) = 0, M g_E = Q(Qᵀg_E) = 0 — 구성 자체로 성립.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# 반대칭 이차형식 열 개수 r — L, M의 실질 랭크 상한 (표현력 vs 파라미터 수).
RANK = 8
# E/S 스칼라 MLP 은닉 폭 (은닉 2층, Tanh — 이차 미분까지 매끄러움).
HIDDEN = 128
# ω 초기 표준편차 — 작게 시작해 초기 보존 동역학을 완만하게.
OMEGA_INIT_STD = 0.01


def _scalar_mlp(dim: int, hidden: int) -> nn.Sequential:
    """z (B,D) → 스칼라 (B,1) MLP. Tanh 활성으로 ∇E, ∇S가 매끄럽게 유지된다."""
    return nn.Sequential(
        nn.Linear(dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 1),
    )


class MetriplecticHead(nn.Module):
    """GENERIC 구조 잠재 동역학 헤드 — 보존(L∇E) + 소산(M∇S) 분해.

    E·S를 MLP로 배우되, L·M을 기울기 직교 열(Q, P)로 조립해 에너지 보존
    dE/dt = 0, 엔트로피 생성 dS/dt ≥ 0, 퇴화 조건 L∇S = M∇E = 0을
    구조적으로(학습 없이도) 보장한다.
    """

    def __init__(self, dim: int, rank: int = RANK, hidden: int = HIDDEN):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.E_net = _scalar_mlp(dim, hidden)
        self.S_net = _scalar_mlp(dim, hidden)
        # 반대칭 이차형식 재료: M = QQᵀ용 (U,V), L = PΩPᵀ용 (A,C).
        scale = 1.0 / math.sqrt(dim)
        self.U = nn.Parameter(torch.randn(dim, rank) * scale)
        self.V = nn.Parameter(torch.randn(dim, rank) * scale)
        self.A = nn.Parameter(torch.randn(dim, rank) * scale)
        self.C = nn.Parameter(torch.randn(dim, rank) * scale)
        # Ω = ω − ωᵀ 의 원료 — 작은 초기화.
        self.omega = nn.Parameter(torch.randn(rank, rank) * OMEGA_INIT_STD)

    # ---- 스칼라 퍼텐셜 -------------------------------------------------------

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """에너지 E(z) — (B,D) → (B,)."""
        return self.E_net(z).squeeze(-1)

    def entropy(self, z: torch.Tensor) -> torch.Tensor:
        """엔트로피 S(z) — (B,D) → (B,)."""
        return self.S_net(z).squeeze(-1)

    # ---- 구조 연산자 ---------------------------------------------------------

    @staticmethod
    def _skew_basis(x: torch.Tensor, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """기울기 g에 정확히 직교하는 열들 (B,D,r).

        열 i = x_i (y_iᵀg) − y_i (x_iᵀg) → 열ᵀg = (x_iᵀg)(y_iᵀg) − (y_iᵀg)(x_iᵀg) = 0.
        나눗셈이 없어 g = 0 근방에서도 특이점 없이 매끄럽다.
        """
        # (1,D,r)·(B,1,r) 브로드캐스트 — 열별 곱을 한 번에.
        return x.unsqueeze(0) * (g @ y).unsqueeze(1) - y.unsqueeze(0) * (g @ x).unsqueeze(1)

    def _operators(self, z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """(Q, P, g_E, g_S) — forward와 테스트가 공유하는 구조 연산자.

        Q (B,D,r): Qᵀg_E = 0 (소산부 M = QQᵀ 재료)
        P (B,D,r): Pᵀg_S = 0 (보존부 L = PΩPᵀ 재료)
        """
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)
        # no_grad 문맥에서도 기울기 계산이 되도록 국소적으로 grad를 켠다.
        with torch.enable_grad():
            g_E = torch.autograd.grad(self.energy(z).sum(), z, create_graph=True)[0]
            g_S = torch.autograd.grad(self.entropy(z).sum(), z, create_graph=True)[0]
        Q = self._skew_basis(self.U, self.V, g_E)
        P = self._skew_basis(self.A, self.C, g_S)
        return Q, P, g_E, g_S

    # ---- 동역학 --------------------------------------------------------------

    def forward(self, z: torch.Tensor) -> dict:
        """dz/dt = L∇E + M∇S 및 진단량.

        반환: {"dz_dt": (B,D), "E": (B,), "S": (B,), "dE_dt": (B,), "dS_dt": (B,)}
        dE_dt/dS_dt는 내적 <g_E, dz/dt>, <g_S, dz/dt>로 해석적으로 계산
        (로깅/테스트/후일 EOS 보조 입력용).
        """
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)
        Q, P, g_E, g_S = self._operators(z)
        # 소산부: M g_S = Q (Qᵀ g_S) — M = QQᵀ는 PSD, M g_E = 0.
        m_gs = torch.einsum("bdr,br->bd", Q, torch.einsum("bdr,bd->br", Q, g_S))
        # 보존부: L g_E = P Ω (Pᵀ g_E) — Ω = ω−ωᵀ로 L 반대칭, L g_S = 0.
        omega = self.omega - self.omega.T
        l_ge = torch.einsum("bdr,br->bd", P, torch.einsum("bdr,bd->br", P, g_E) @ omega.T)
        dz_dt = l_ge + m_gs
        return {
            "dz_dt": dz_dt,
            "E": self.energy(z),
            "S": self.entropy(z),
            "dE_dt": (g_E * dz_dt).sum(-1),
            "dS_dt": (g_S * dz_dt).sum(-1),
        }

    def step(self, z: torch.Tensor, dt: float, method: str = "rk4") -> torch.Tensor:
        """명시적 적분 한 스텝 (euler | rk4). no_grad 아님 — 학습 시 backprop 가능."""
        if method == "euler":
            return z + dt * self(z)["dz_dt"]
        if method == "rk4":
            k1 = self(z)["dz_dt"]
            k2 = self(z + 0.5 * dt * k1)["dz_dt"]
            k3 = self(z + 0.5 * dt * k2)["dz_dt"]
            k4 = self(z + dt * k3)["dz_dt"]
            return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        raise ValueError(f"지원하지 않는 적분 방법: {method!r} (euler | rk4)")
