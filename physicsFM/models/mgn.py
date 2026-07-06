"""MeshGraphNetLite — 그래프 라이브러리 무의존 encode-process-decode MGN.

요약 흐름: 노드/엣지 피처를 각각 MLP 로 인코딩 → L회 메시지 패싱
(엣지 갱신 e' = e + MLP([e, h_src, h_dst]); 노드 갱신 h' = h + MLP([h, Σ_{dst} e']))
→ 노드별 가속도형 변위 (N,3) 디코딩. 집계는 zeros.index_add_ 하나로 처리해
DGL/PyG 에 의존하지 않는다 (physicsnemo MGN 어댑터는 models/__init__.py 참고).

physicsnemo 가 설치돼 있으면 physicsnemo.Module 을 상속해 체크포인트/메타 기능을
얻고, 없으면 nn.Module 로 동작한다 (학습 경로는 동일).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

try:  # physicsnemo 는 선택 의존성 — 없어도 전 기능 동작
    from physicsnemo import Module as BaseModule
except ImportError:
    BaseModule = nn.Module


def make_mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2,
             layernorm: bool = True) -> nn.Sequential:
    """[Linear+SiLU]×layers → Linear(out) (+LayerNorm) — MGN 표준 블록."""
    mods: list[nn.Module] = []
    dim = in_dim
    for _ in range(layers):
        mods += [nn.Linear(dim, hidden), nn.SiLU()]
        dim = hidden
    mods.append(nn.Linear(dim, out_dim))
    if layernorm:
        mods.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*mods)


class _ProcessorLayer(nn.Module):
    """메시지 패싱 한 층 (residual)."""

    def __init__(self, hidden: int, mlp_layers: int):
        super().__init__()
        self.edge_mlp = make_mlp(3 * hidden, hidden, hidden, mlp_layers)
        self.node_mlp = make_mlp(2 * hidden, hidden, hidden, mlp_layers)

    def forward(self, h: torch.Tensor, e: torch.Tensor,
                edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        e = e + self.edge_mlp(torch.cat([e, h[src], h[dst]], dim=1))
        agg = torch.zeros_like(h).index_add_(0, dst, e)
        h = h + self.node_mlp(torch.cat([h, agg], dim=1))
        return h, e


class MeshGraphNetLite(BaseModule):
    """무의존 MGN. forward → pred (N,3) 또는 (pred, latent (N,hidden))."""

    def __init__(self, node_in: int, edge_in: int, hidden: int = 128, out_dim: int = 3,
                 num_layers: int = 15, mlp_layers: int = 2, grad_checkpoint: bool = False):
        super().__init__()
        self.node_encoder = make_mlp(node_in, hidden, hidden, mlp_layers)
        self.edge_encoder = make_mlp(edge_in, hidden, hidden, mlp_layers)
        self.processor = nn.ModuleList(
            _ProcessorLayer(hidden, mlp_layers) for _ in range(num_layers)
        )
        self.decoder = make_mlp(hidden, hidden, out_dim, mlp_layers, layernorm=False)
        self.grad_checkpoint = grad_checkpoint

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor,
                edge_feat: torch.Tensor, return_latent: bool = False):
        h = self.node_encoder(node_feat)
        e = self.edge_encoder(edge_feat)
        for layer in self.processor:
            if self.grad_checkpoint and self.training:
                h, e = checkpoint(layer, h, e, edge_index, use_reentrant=False)
            else:
                h, e = layer(h, e, edge_index)
        pred = self.decoder(h)
        return (pred, h) if return_latent else pred
