"""EncoderProjectorQFormer (BAT 전용), vendored from SLAM-LLM.

원본: src/slam_llm/models/projector.py 의 `EncoderProjectorQFormer`.
SLAM-LLM 본 코드는 `model_config` (omegaconf DictConfig) 객체에서 `.get()`을
호출하지만, vendoring 환경에서는 단순한 dataclass / dict-like config로 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from transformers import Blip2QFormerConfig, Blip2QFormerModel


@dataclass
class QFormerConfig:
    """EncoderProjectorQFormer가 필요로 하는 필드만 모은 가벼운 config."""

    encoder_dim: int = 768
    llm_dim: int = 4096
    qformer_layers: int = 8
    query_len: int = 64


class EncoderProjectorQFormer(nn.Module):
    """Spatial-AST 출력(768d) → 64개의 query token (LLaMA hidden, 4096d) 으로 매핑."""

    def __init__(self, config: QFormerConfig):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim

        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = config.qformer_layers

        self.query_len = int(config.query_len)
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x: torch.Tensor, atts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: encoder hidden states, shape (B, S, encoder_dim).
            atts: attention mask over encoder hidden states (B, S) or None.
                  BAT inference에서는 SpatialAST 출력에 mask가 없으므로 None.
        Returns:
            query_proj: (B, query_len, llm_dim)
        """
        if atts is None:
            atts = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        query = self.query.expand(x.shape[0], -1, -1)

        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        return query_proj
