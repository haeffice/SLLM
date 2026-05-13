"""Spatial-AST BinauralEncoder, vendored from SLAM-LLM.

원본:
- src/slam_llm/models/SpatialAST/SpatialAST.py  (BinauralEncoder, PatchEmbed_new, conv3x3)
- src/slam_llm/models/SpatialAST/vision_transformer.py  (VisionTransformer, Block, Attention, Mlp)

학습/시각화 코드 및 HybridEmbed/PatchEmbed 같은 미사용 변종은 제거하고
inference에 필요한 최소 클래스만 남김. forward는 32 kHz binaural waveform
`(B, 2, T)`를 입력받아 `(B, 3 + 512, 768)` 형태의 토큰 시퀀스를 반환.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn
from timm.models.layers import to_2tuple, DropPath, trunc_normal_
from torchlibrosa.stft import STFT, LogmelFilterBank


# -----------------------------------------------------------------------------
# ViT building blocks (vision_transformer.py에서 필요한 것만 발췌)
# -----------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _VisionTransformer(nn.Module):
    """BinauralEncoder가 상속받는 베이스 ViT.

    원본 VisionTransformer를 거의 그대로 옮겼고, classifier head/init은 유지함.
    `forward_features`/`forward`는 사용하지 않지만 가중치 키 호환성을 위해
    head와 norm은 그대로 둠 (BinauralEncoder에서 norm은 del됨).
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # PatchEmbed 자리: 서브클래스에서 다시 만들기 때문에 placeholder만 둠.
        # 가중치 키 호환을 위해 patch_embed.proj 같은 키가 BinauralEncoder가
        # 다시 만든 PatchEmbed_new로 모두 들어가도록 한다.
        self.patch_embed = _PatchEmbed(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class _PatchEmbed(nn.Module):
    """기본 PatchEmbed. BinauralEncoder가 PatchEmbed_new로 대체하지만
    super().__init__에서 한번 만들어진다."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


# -----------------------------------------------------------------------------
# Spatial-AST specific layers
# -----------------------------------------------------------------------------

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class PatchEmbed_new(nn.Module):
    """Flexible (overlap-able) Image to Patch Embedding.
    Spatial-AST는 stride=16, patch=16 (no overlap)로 사용."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        _, _, h, w = self.get_output_shape(img_size)
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        return self.proj(torch.randn(1, self.in_chans, img_size[0], img_size[1])).shape

    def forward(self, x):
        x = self.proj(x)         # (B, embed_dim, H', W')
        x = x.flatten(2)         # (B, embed_dim, H'*W')
        x = x.transpose(1, 2)    # (B, H'*W', embed_dim)
        return x


class BinauralEncoder(_VisionTransformer):
    """Spatial Audio Spectrogram Transformer (BAT의 인코더).

    Reference: Spatial-AST (https://arxiv.org/abs/2402.01591).
    입력: waveform `(B, 2, T)` @ 32 kHz, 10초(T=320000) 권장.
    출력: `(B, 3 + 512, 768)`  — 3 cls tokens + 512 patch tokens.
    """

    def __init__(self, num_cls_tokens=3, **kwargs):
        super().__init__(**kwargs)
        img_size = (1024, 128)
        in_chans = 1
        emb_dim = 768

        del self.cls_token
        self.num_cls_tokens = num_cls_tokens
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls_tokens, emb_dim))

        self.patch_embed = PatchEmbed_new(
            img_size=img_size, patch_size=(16, 16),
            in_chans=in_chans, embed_dim=emb_dim, stride=16,
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False
        )

        self.spectrogram_extractor = STFT(
            n_fft=1024, hop_length=320, win_length=1024, window='hann',
            center=True, pad_mode='reflect', freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=32000, n_fft=1024, n_mels=128, fmin=50,
            fmax=14000, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True,
        )

        self.conv_downsample = nn.Sequential(
            conv3x3(4, 1),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )

        self.bn = nn.BatchNorm2d(2, affine=False)
        del self.norm  # 원본도 ViT의 norm을 제거함

        self.target_frame = 1024

    def forward_features_mask(self, x):
        B = x.shape[0]
        x = x + self.pos_embed[:, 1:, :]

        cls_tokens = self.cls_tokens.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        return x

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        B, C, T = waveforms.shape

        waveforms = waveforms.reshape(B * C, T)
        real, imag = self.spectrogram_extractor(waveforms)

        log_mel = self.logmel_extractor(torch.sqrt(real ** 2 + imag ** 2)).reshape(B, C, -1, 128)
        log_mel = self.bn(log_mel)

        # IPD (Inter-channel Phase Difference): 채널 1 - 채널 0
        IPD = torch.atan2(imag[1::2], real[1::2]) - torch.atan2(imag[::2], real[::2])
        ipd_feat = torch.matmul(
            torch.cat([torch.cos(IPD), torch.sin(IPD)], dim=1),
            self.logmel_extractor.melW,
        )
        x = torch.cat([log_mel, ipd_feat], dim=1)  # (B, 4, T', 128)

        if x.shape[2] < self.target_frame:
            x = nn.functional.interpolate(
                x, (self.target_frame, x.shape[3]), mode="bicubic", align_corners=True,
            )

        x = self.conv_downsample(x)       # (B, 1, 1024, 128)
        x = self.patch_embed(x)           # (B, 512, 768)
        x = self.forward_features_mask(x) # (B, 3 + 512, 768)
        return x


def build_binaural_encoder() -> BinauralEncoder:
    """기본 BAT 하이퍼파라미터로 인코더를 만든다 (encoder.SpatialASTEncoder와 동일)."""
    return BinauralEncoder(
        num_classes=355,
        drop_path_rate=0.1,
        num_cls_tokens=3,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
