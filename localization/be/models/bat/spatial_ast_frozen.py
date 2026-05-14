import logging, torch, torchaudio

import torch.nn.functional as F

from ._ckpt import trusted_torch_load
from .spatial_ast import build_AST
import torch.nn as nn


class SpatialASTFrozen(nn.Module):
    def __init__(self,
                 ckpt_path: str,
                 freeze: bool = True,
                 logger = None,
                 num_classes: int = 355):
        super().__init__()

        self.encoder = build_AST(num_classes = num_classes,
                                 drop_path_rate = 0.0,
                                 num_cls_tokens = 3)
        self.logger = logger
        ckpt = trusted_torch_load(ckpt_path)
        state_dict = ckpt.get('model', ckpt)
        missing_keys, unexpected_keys = self.encoder.load_state_dict(state_dict, strict=False)
        total = len(state_dict)
        applied = total - len(unexpected_keys)
        if not missing_keys and not unexpected_keys:
            self.logger.info("Spatial-AST loaded successfully (%d/%d keys)", applied, total)
        else:
            self.logger.warning(
                "Spatial-AST partial load: %d/%d keys (missing=%d, unexpected=%d)",
                applied, total, len(missing_keys), len(unexpected_keys),
            )
        if applied == 0:
            self.logger.warning("No encoder keys matched - Spatial-AST is randomly initialized.")

        if freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
        
    def forward(self, waveforms: torch.Tensor, reverbs: torch.Tensor = None):
        m = self.encoder
        if reverbs is not None:
            waveforms = torchaudio.functional.fftconvolve(waveforms, reverbs, mode='full')[..., :waveforms.shape[-1]]
        B, C, T = waveforms.shape

        waveforms = waveforms.reshape(B * C, T)
        real, imag = m.spectrogram_extractor(waveforms)

        log_mel = m.logmel_extractor(torch.sqrt(real**2 + imag**2)).reshape(B, C, -1, 128)
        log_mel = m.bn(log_mel)

        IPD = torch.atan2(imag[1::2], real[1::2]) - torch.atan2(imag[::2], real[::2])
        x = torch.cat([log_mel, torch.matmul(torch.cat([torch.cos(IPD), torch.sin(IPD)], dim=1), m.logmel_extractor.melW)], dim=1)


        if x.shape[2] < m.target_frame:
            x = F.interpolate(x,
                              (m.target_frame, x.shape[3]),
                              mode='bicubic',
                              align_corners=True)
        
        x = m.conv_downsample(x)
        x = m.patch_embed(x)
        x = m.forward_features_mask(x, mask_t_prob=0.0, mask_f_prob=0.0)
        return x.detach()