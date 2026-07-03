"""EOS(정지) 헤드 — 노드 표현을 그래프 단위로 풀링해 settled 로짓을 낸다.

요약 흐름: mean+max 풀링(순열 불변) → MLP [2D→D→1]. "정지"는 전역 저에너지
상태라 processor latent(히스토리 문맥 포함)에서 읽는 것이 디코더 출력(Δ≈0,
바운스 정점과 구분 불가)보다 안정적이다. 후일 metriplectic 의 dS/dt 를
보조 입력으로 concat 하는 확장 지점.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EosHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, node_repr: torch.Tensor,
                batch_idx: torch.Tensor | None = None) -> torch.Tensor:
        """node_repr (N,D), batch_idx (N,) → 로짓 (B,). batch_idx=None 이면 B=1."""
        if batch_idx is None:
            pooled = torch.cat([node_repr.mean(0), node_repr.max(0).values])
            return self.mlp(pooled[None, :]).squeeze(-1)
        num_graphs = int(batch_idx.max().item()) + 1
        d = node_repr.shape[1]
        mean = torch.zeros(num_graphs, d, dtype=node_repr.dtype, device=node_repr.device)
        mean.index_add_(0, batch_idx, node_repr)
        counts = torch.bincount(batch_idx, minlength=num_graphs).clamp(min=1)
        mean = mean / counts[:, None].to(mean.dtype)
        mx = torch.full((num_graphs, d), torch.finfo(node_repr.dtype).min,
                        dtype=node_repr.dtype, device=node_repr.device)
        mx.scatter_reduce_(0, batch_idx[:, None].expand(-1, d), node_repr, reduce="amax")
        return self.mlp(torch.cat([mean, mx], dim=1)).squeeze(-1)
