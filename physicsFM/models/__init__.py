"""모델 팩토리 — config 의 backbone 문자열로 백본을 고른다.

요약 흐름: 기본 local_mgn(무의존). physicsnemo_mgn 선택 시 physicsnemo+PyG 를
import 시도하고, 실패하면 경고 후 local_mgn 으로 폴백한다(학습 경로 보존).
두 백본 모두 forward(node_feat, edge_index, edge_feat, return_latent=True) →
(pred (N,3), latent) 계약을 지킨다. eos 헤드 입력 차원은 백본에 따라 다르다
(local: hidden / physicsnemo 어댑터: latent 미노출 → node_feat‖pred).
"""

from __future__ import annotations

import logging

import torch.nn as nn

from graph import EDGE_FEAT_DIM, NODE_FEAT_DIM
from models.mgn import MeshGraphNetLite


class _PhysicsNemoMGNAdapter(nn.Module):
    """physicsnemo MeshGraphNet(PyG 백엔드) → local 계약으로 감싸는 어댑터.

    physicsnemo 2.x 의 MGN 은 (node_features, edge_features, graph) 시그니처이며
    torch_geometric.data.Data 입력 시 PyG 백엔드를 쓴다. 내부 latent 를 노출하지
    않으므로 return_latent 는 node_feat‖pred 를 대신 돌려준다.
    """

    def __init__(self, cfg_model: dict):
        super().__init__()
        from physicsnemo.models.meshgraphnet import MeshGraphNet  # noqa: PLC0415

        self.net = MeshGraphNet(
            input_dim_nodes=NODE_FEAT_DIM,
            input_dim_edges=EDGE_FEAT_DIM,
            output_dim=3,
            processor_size=int(cfg_model["num_layers"]),
            hidden_dim_processor=int(cfg_model["hidden_dim"]),
            hidden_dim_node_encoder=int(cfg_model["hidden_dim"]),
            hidden_dim_edge_encoder=int(cfg_model["hidden_dim"]),
            hidden_dim_node_decoder=int(cfg_model["hidden_dim"]),
        )

    def forward(self, node_feat, edge_index, edge_feat, return_latent: bool = False):
        import torch  # noqa: PLC0415
        from torch_geometric.data import Data  # noqa: PLC0415

        graph = Data(edge_index=edge_index, num_nodes=node_feat.shape[0])
        pred = self.net(node_feat, edge_feat, graph)
        if return_latent:
            return pred, torch.cat([node_feat, pred], dim=1)
        return pred


def get_model(cfg_model: dict) -> tuple[nn.Module, int]:
    """(백본, eos 헤드 입력 차원). physicsnemo 실패 시 local_mgn 폴백."""
    backbone = cfg_model.get("backbone", "local_mgn")
    if backbone == "physicsnemo_mgn":
        try:
            model = _PhysicsNemoMGNAdapter(cfg_model)
            return model, NODE_FEAT_DIM + 3
        except Exception as exc:  # ImportError 포함 — 폴백 후 계속
            logging.warning("physicsnemo_mgn 사용 불가(%s) → local_mgn 폴백", exc)
    elif backbone != "local_mgn":
        raise ValueError(f"알 수 없는 backbone: {backbone!r}")
    model = MeshGraphNetLite(
        node_in=NODE_FEAT_DIM,
        edge_in=EDGE_FEAT_DIM,
        hidden=int(cfg_model["hidden_dim"]),
        num_layers=int(cfg_model["num_layers"]),
        mlp_layers=int(cfg_model["mlp_layers"]),
        grad_checkpoint=bool(cfg_model.get("grad_checkpoint", False)),
    )
    return model, int(cfg_model["hidden_dim"])
