"""Standalone BAT (Spatial-AST + Q-Former + Llama-2-7b) evaluation script.

다른 레포로 옮겨도 작동하도록 SLLM 프로젝트 외부 의존성을 모두 제거한
self-contained 형태. SpatialSoundQA-style eval JSON에 대해 batched
inference를 수행하고 결과를 JSONL로 저장한다. `torchrun`으로 실행되며
`DistributedSampler`로 샘플을 rank별로 나눠 처리한다.

입력 JSON:
    {"data": [{"audio_id": str, "reverb_id": str, "question_id": int,
                "question": str, "answer": str, ...}, ...]}

출력 JSONL (one line per sample):
    {"question_id": int, "pred": str, "ans": str, "audio_path": str}

    audio_path  = os.path.join(audio_root,  audio_id)
    reverb_path = os.path.join(reverb_root, reverb_id)   # reverb_id가 있을 때

외부 라이브러리 의존성:
    torch, transformers, peft, timm, torchaudio, torchlibrosa,
    numpy, scipy, soundfile, omegaconf (체크포인트 unpickle 시 import 가능해야 함).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from tqdm import tqdm
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn as nn
import torchaudio
from scipy import signal
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchlibrosa.stft import STFT, LogmelFilterBank
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Blip2QFormerConfig,
    Blip2QFormerModel,
)
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict


logger = logging.getLogger("BAT")


# =============================================================================
# Constants (examples/seld_spatialsoundqa/seld_config.py 값을 그대로 옮김)
# =============================================================================

SAMPLE_RATE = 32_000
AUDIO_SAMPLES = 10 * SAMPLE_RATE     # 320,000
QUERY_LEN = 64

ENCODER_DIM = 768
LLM_DIM = 4096
QFORMER_LAYERS = 8

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


# =============================================================================
# Checkpoint helpers
# =============================================================================

def trusted_torch_load(path: str):
    """`torch.load` wrapper that disables the PyTorch 2.6+ safe-pickle default.

    SLAM-LLM/BAT 체크포인트는 가중치 외에 학습 시 사용된 omegaconf
    객체(`ListConfig`, `DictConfig`)를 메타데이터로 포함하고 있다. PyTorch 2.6
    부터 `torch.load`의 기본값이 `weights_only=True`로 바뀌면서 이런 임의
    클래스는 unpickling이 거부된다. 공식 BAT 릴리즈 출처를 신뢰하므로 명시적
    `weights_only=False`를 사용하고, pickle이 `ListConfig` 같은 클래스를 객체로
    복원하려면 그 모듈이 import 가능해야 하므로 환경에 `omegaconf`를 두어야 함.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_encoder_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Spatial-AST 체크포인트 로드. {'model': sd} 또는 sd 그대로 둘 다 지원."""
    blob = trusted_torch_load(path)
    if isinstance(blob, dict) and "model" in blob and isinstance(blob["model"], dict):
        return blob["model"]
    if isinstance(blob, dict):
        return blob
    raise ValueError(f"Unexpected encoder checkpoint format: {type(blob)}")


def _split_projector_lora(
    state: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """BAT model.pt를 (projector, lora) 두 그룹으로 분리.

    slam_model의 attribute 구조 (`encoder_projector.*`, `llm.*`)에 맞춰
    저장되어 있다. 가능한 prefix 변형을 모두 흡수한다.
    """
    proj: Dict[str, torch.Tensor] = {}
    lora: Dict[str, torch.Tensor] = {}
    unknown: list[str] = []

    for k, v in state.items():
        if k.startswith("encoder_projector."):
            proj[k[len("encoder_projector."):]] = v
        elif k.startswith("module.encoder_projector."):
            proj[k[len("module.encoder_projector."):]] = v
        elif k.startswith("llm."):
            # `.default` 슬롯은 set_peft_model_state_dict(adapter_name="default")이
            # 다시 주입하므로 ckpt에 이미 들어있으면 미리 제거 (중복 방지).
            lora[k[len("llm."):].replace(".default", "")] = v
        elif k.startswith("module.llm."):
            lora[k[len("module.llm."):].replace(".default", "")] = v
        elif k.startswith("base_model.model.") or k.startswith("model."):
            lora[k.replace(".default", "")] = v
        else:
            unknown.append(k)

    if unknown:
        logger.warning("BAT projector ckpt: %d keys not recognized", len(unknown))
    return proj, lora


# =============================================================================
# Vision Transformer building blocks
# (SLAM-LLM `src/slam_llm/models/SpatialAST/vision_transformer.py` 그대로)
# =============================================================================

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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
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


class PatchEmbed_new(nn.Module):
    """Flexible Image to Patch Embedding."""
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
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (timm-derived; SpatialAST의 부모 클래스)."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # SpatialAST가 patch_embed를 재정의하지만 호환을 위해 동일 형태로 둠.
        self.patch_embed = PatchEmbed_new(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, stride=patch_size,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
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
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# =============================================================================
# Spatial-AST encoder
# (SLAM-LLM `src/slam_llm/models/SpatialAST/SpatialAST.py` 기반)
# =============================================================================

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SpatialAST(VisionTransformer):
    def __init__(self, num_cls_tokens=3, **kwargs):
        super().__init__(**kwargs)
        img_size = (1024, 128)
        in_chans = 1
        emb_dim = 768

        del self.cls_token
        self.num_cls_tokens = num_cls_tokens
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls_tokens, emb_dim))
        torch.nn.init.normal_(self.cls_tokens, std=.02)

        self.patch_embed = PatchEmbed_new(
            img_size=img_size, patch_size=16, in_chans=in_chans,
            embed_dim=emb_dim, stride=16,
        )  # no overlap
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False
        )

        self.spectrogram_extractor = STFT(
            n_fft=1024, hop_length=320, win_length=1024, window="hann",
            center=True, pad_mode="reflect", freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=32000, n_fft=1024, n_mels=128, fmin=50, fmax=14000,
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True,
        )

        # ckpt key 매칭용 (forward에서 사용 안 함)
        from torchaudio.functional import melscale_fbanks
        self.melscale_filterbank = melscale_fbanks(
            n_freqs=1024, f_min=50, f_max=14000, n_mels=128, sample_rate=32000,
        )

        self.conv_downsample = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )

        # training augmentation modules — inference 경로엔 안 쓰임. ckpt 키 매칭만.
        self.timem = torchaudio.transforms.TimeMasking(192)
        self.freqm = torchaudio.transforms.FrequencyMasking(48)

        self.bn = nn.BatchNorm2d(2, affine=False)
        del self.norm  # remove the original norm

        self.target_frame = 1024

        # SEL/distance/azimuth/elevation 헤드 — inference forward 경로 미사용,
        # ckpt 키 매칭만 위해 보존.
        self.dis_norm = kwargs["norm_layer"](emb_dim)
        self.doa_norm = kwargs["norm_layer"](emb_dim)
        self.fc_norm = kwargs["norm_layer"](emb_dim)

        self.distance_head = nn.Linear(emb_dim, 21)
        self.azimuth_head = nn.Linear(emb_dim, 360)
        self.elevation_head = nn.Linear(emb_dim, 180)

        trunc_normal_(self.head.weight, std=2e-5)
        trunc_normal_(self.distance_head.weight, std=2e-5)
        trunc_normal_(self.azimuth_head.weight, std=2e-5)
        trunc_normal_(self.elevation_head.weight, std=2e-5)

    def forward_features_mask(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        B = x.shape[0]
        x = x + self.pos_embed[:, 1:, :]
        cls_tokens = self.cls_tokens.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return x


def build_AST(**kwargs):
    return SpatialAST(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


class SpatialASTFrozen(nn.Module):
    """Encoder 체크포인트 로드 + frozen forward path."""

    def __init__(self, ckpt_path: str, freeze: bool = True,
                 logger: Optional[logging.Logger] = None, num_classes: int = 355):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)

        self.encoder = build_AST(
            num_classes=num_classes, drop_path_rate=0.0, num_cls_tokens=3,
        )
        ckpt = trusted_torch_load(ckpt_path)
        state_dict = ckpt.get("model", ckpt)
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

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        m = self.encoder
        B, C, T = waveforms.shape
        waveforms = waveforms.reshape(B * C, T)
        real, imag = m.spectrogram_extractor(waveforms)

        log_mel = m.logmel_extractor(torch.sqrt(real ** 2 + imag ** 2)).reshape(B, C, -1, 128)
        log_mel = m.bn(log_mel)

        IPD = torch.atan2(imag[1::2], real[1::2]) - torch.atan2(imag[::2], real[::2])
        x = torch.cat([
            log_mel,
            torch.matmul(
                torch.cat([torch.cos(IPD), torch.sin(IPD)], dim=1),
                m.logmel_extractor.melW,
            ),
        ], dim=1)

        if x.shape[2] < m.target_frame:
            x = torch.nn.functional.interpolate(
                x, (m.target_frame, x.shape[3]), mode="bicubic", align_corners=True,
            )

        x = m.conv_downsample(x)
        x = m.patch_embed(x)
        x = m.forward_features_mask(x, mask_t_prob=0.0, mask_f_prob=0.0)
        return x.detach()


# =============================================================================
# Q-Former projector
# (SLAM-LLM `src/slam_llm/models/projector.py`의 EncoderProjectorQFormer)
# =============================================================================

@dataclass
class QFormerConfig:
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
        if atts is None:
            atts = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        query = self.query.expand(x.shape[0], -1, -1)
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        return self.norm(self.linear(query_output.last_hidden_state))


# =============================================================================
# Prompt + audio preprocessing
# (SLAM-LLM `examples/seld_spatialsoundqa/dataset/spatial_audio_dataset.py` 기반)
# =============================================================================

_PROMPT_NO_INPUT = (
    "Based on the audio you've heard, refer to the instruction and provide a response.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)


def format_prompt(instruction: str, input: Optional[str] = None) -> str:
    if input is None:
        return _PROMPT_NO_INPUT.format(instruction=instruction)
    return _PROMPT_WITH_INPUT.format(instruction=instruction, input=input)


def normalize_audio(audio_data: np.ndarray, target_dBFS: float = -14.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio_data ** 2)))
    if rms == 0:
        return audio_data
    current_dBFS = 20 * np.log10(rms)
    gain_dB = target_dBFS - current_dBFS
    gain_linear = 10 ** (gain_dB / 20)
    return audio_data * gain_linear


# =============================================================================
# BAT model wrapper
# =============================================================================

class BAT:
    """Encoder + Q-Former projector + Llama-2-7b(+LoRA) bundle."""

    def __init__(self, device, encoder, projector, llm, tokenizer):
        self.device = device
        self.encoder = encoder
        self.projector = projector
        self.llm = llm
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, device: torch.device, llama_path: str, encoder_ckpt: str,
             projector_ckpt: str) -> "BAT":
        logger.info("Loading BAT tokenizer + LLM from %s", llama_path)
        tokenizer = AutoTokenizer.from_pretrained(llama_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"  # causal LM batch generation

        llm = AutoModelForCausalLM.from_pretrained(llama_path, dtype=torch.float16)
        peft_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES, lora_dropout=LORA_DROPOUT,
            bias="none", task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, peft_cfg)
        llm.eval()
        if getattr(llm, "generation_config", None) is not None:
            llm.generation_config.max_length = None

        logger.info("Loading Spatial-AST encoder from %s", encoder_ckpt)
        encoder = SpatialASTFrozen(
            ckpt_path=encoder_ckpt, freeze=True, logger=logger, num_classes=355,
        )

        logger.info("Loading BAT projector + LoRA delta from %s", projector_ckpt)
        proj_cfg = QFormerConfig(
            encoder_dim=ENCODER_DIM, llm_dim=LLM_DIM,
            qformer_layers=QFORMER_LAYERS, query_len=QUERY_LEN,
        )
        projector = EncoderProjectorQFormer(proj_cfg)

        bat_state = trusted_torch_load(projector_ckpt)
        if (isinstance(bat_state, dict) and "model" in bat_state
                and isinstance(bat_state["model"], dict)):
            bat_state = bat_state["model"]
        if not isinstance(bat_state, dict):
            raise ValueError(f"Unexpected projector ckpt format: {type(bat_state)}")

        proj_keys = [k for k in bat_state.keys() if "encoder_projector" in k]
        logger.info(
            "BAT projector ckpt: total=%d, encoder_projector keys=%d",
            len(bat_state), len(proj_keys),
        )

        proj_state, lora_state = _split_projector_lora(bat_state)
        if proj_state:
            mk, uk = projector.load_state_dict(proj_state, strict=False)
            total = len(proj_state)
            applied = total - len(uk)
            if not mk and not uk:
                logger.info("Projector loaded successfully (%d/%d keys)", applied, total)
            else:
                logger.warning(
                    "Projector partial load: %d/%d keys (missing=%d, unexpected=%d)",
                    applied, total, len(mk), len(uk),
                )
        else:
            logger.warning("No projector keys found in BAT ckpt — projector randomly initialized!")
        projector.eval()

        if lora_state:
            load_result = set_peft_model_state_dict(llm, lora_state, adapter_name="default")
            uk = list(getattr(load_result, "unexpected_keys", []) or [])
            total = len(lora_state)
            applied = total - len(uk)
            if not uk:
                logger.info("LoRA loaded successfully (%d/%d delta keys)", applied, total)
            else:
                logger.warning(
                    "LoRA partial load: %d/%d delta keys (unexpected=%d)",
                    applied, total, len(uk),
                )
        else:
            logger.warning("No LoRA keys found in BAT ckpt — LLM uses base weights only.")

        encoder = encoder.to(device)
        projector = projector.to(device)
        llm = llm.to(device)

        return cls(device, encoder, projector, llm, tokenizer)


# =============================================================================
# Dataset
# =============================================================================

class SpatialSoundQADataset(Dataset):
    """SpatialSoundQA eval JSON → (waveform, prompt, meta) 변환."""

    def __init__(self, json_path: str, audio_root: str, reverb_root: str,
                 normalize: bool = True):
        with open(json_path) as f:
            self.data = json.load(f)["data"]
        self.audio_root = audio_root
        self.reverb_root = reverb_root
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        s = self.data[idx]
        audio_id = s["audio_id"]
        reverb_id = s.get("reverb_id")
        audio_path = os.path.join(self.audio_root, audio_id)
        reverb_path = (
            os.path.join(self.reverb_root, reverb_id) if reverb_id else None
        )
        waveform = self._load_waveform(audio_path, reverb_path)
        prompt = format_prompt(s["question"], None)
        return {
            "waveform": waveform,
            "prompt": prompt,
            "question_id": int(s["question_id"]),
            "answer": s.get("answer", ""),
            "audio_path": audio_path,
        }

    def _load_waveform(self, audio_path: str, reverb_path: Optional[str]) -> torch.Tensor:
        wave, sr = sf.read(audio_path)
        if wave.ndim > 1:
            wave = wave[:, 0]
        if sr != SAMPLE_RATE:
            wave = signal.resample_poly(wave, SAMPLE_RATE, sr)
        if self.normalize:
            wave = normalize_audio(wave, -14.0)
        wave = np.asarray(wave, dtype=np.float32).reshape(1, -1)
        if reverb_path is not None:
            reverb = np.load(reverb_path).astype(np.float32)
            wave = signal.fftconvolve(wave, reverb, mode="full")
        else:
            # reverb 없는 샘플은 mono를 양쪽으로 복제해 stereo 확보
            wave = np.repeat(wave, 2, axis=0)
        wave_t = torch.from_numpy(np.ascontiguousarray(wave)).float()
        if wave_t.shape[1] < AUDIO_SAMPLES:
            wave_t = torch.nn.functional.pad(
                wave_t, (0, AUDIO_SAMPLES - wave_t.shape[1])
            )
        else:
            wave_t = wave_t[:, :AUDIO_SAMPLES]
        return wave_t


def collate_fn(batch: list[dict]) -> dict:
    return {
        "waveforms": torch.stack([b["waveform"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "question_ids": [b["question_id"] for b in batch],
        "answers": [b["answer"] for b in batch],
        "audio_paths": [b["audio_path"] for b in batch],
    }


# =============================================================================
# Batched inference
# =============================================================================

@torch.no_grad()
def infer_batch(
    model: BAT,
    waveforms: torch.Tensor,
    prompts: list[str],
    device: torch.device,
    num_beams: int = 4,
    max_new_tokens: int = 200,
) -> list[str]:
    """N audios + N prompts (서로 다른 쌍) batch 추론."""
    waveforms = waveforms.to(device)
    B = waveforms.size(0)

    tokenizer = model.tokenizer
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    prompt_ids = encoded.input_ids
    prompt_attn = encoded.attention_mask

    audio_pseudo = torch.full((B, QUERY_LEN), -1, dtype=prompt_ids.dtype)
    audio_attn = torch.ones((B, QUERY_LEN), dtype=prompt_attn.dtype)

    input_ids = torch.cat([audio_pseudo, prompt_ids], dim=1).to(device)
    attention_mask = torch.cat([audio_attn, prompt_attn], dim=1).to(device)
    modality_mask = input_ids.eq(-1)

    encoder_outs = model.encoder(waveforms)            # (B, 3+512, 768)
    proj_out = model.projector(encoder_outs)           # (B, 64, 4096)
    llm_dtype = next(model.llm.parameters()).dtype
    proj_out = proj_out.to(llm_dtype)

    safe_ids = input_ids.clamp(min=0)
    inputs_embeds = model.llm.get_input_embeddings()(safe_ids).clone()
    if inputs_embeds.dtype != proj_out.dtype:
        inputs_embeds = inputs_embeds.to(proj_out.dtype)
    for b in range(B):
        inputs_embeds[b, modality_mask[b]] = proj_out[b]

    out_ids = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        num_beams=num_beams,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        min_length=1,
        top_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return [r.strip() for r in tokenizer.batch_decode(out_ids, skip_special_tokens=True)]


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", required=True, help="eval JSON 파일 경로")
    p.add_argument("--audio-root", required=True, help="audio_id의 기준 디렉터리")
    p.add_argument("--reverb-root", required=True, help="reverb_id의 기준 디렉터리")
    p.add_argument("--output-dir", required=True, help="JSONL 저장 디렉터리")
    p.add_argument("--llama-path", required=True, help="Llama-2-7b-hf 로컬 경로")
    p.add_argument("--encoder-ckpt", required=True, help="SpatialAST finetuned.pth")
    p.add_argument("--projector-ckpt", required=True, help="BAT model.pt (Q-Former + LoRA)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--no-normalize", action="store_true",
                   help="-14 dBFS RMS 정규화 비활성화")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if rank == 0:
        logger.info("world_size=%d, device=%s", world_size, device)
        logger.info("loading BAT model...")

    model = BAT.load(
        device,
        llama_path=args.llama_path,
        encoder_ckpt=args.encoder_ckpt,
        projector_ckpt=args.projector_ckpt,
    )

    if rank == 0:
        logger.info("BAT model ready")

    dataset = SpatialSoundQADataset(
        args.eval_json,
        args.audio_root,
        args.reverb_root,
        normalize=not args.no_normalize,
    )
    if rank == 0:
        logger.info("dataset size = %d", len(dataset))

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    rank_out = os.path.join(args.output_dir, f"predictions.rank{rank}.jsonl")

    t0 = time.monotonic()
    with open(rank_out, "w", encoding="utf-8") as fout:
        for batch in tqdm(loader, desc=f"rank{rank}", disable=(rank != 0)):
            preds = infer_batch(
                model,
                batch["waveforms"],
                batch["prompts"],
                device,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
            for qid, pred, ans, apath in zip(
                batch["question_ids"], preds, batch["answers"], batch["audio_paths"]
            ):
                fout.write(json.dumps({
                    "question_id": qid,
                    "pred": pred,
                    "ans": ans,
                    "audio_path": apath,
                }, ensure_ascii=False) + "\n")

    dist.barrier()

    if rank == 0:
        merged = os.path.join(args.output_dir, "predictions.jsonl")
        total = 0
        with open(merged, "w", encoding="utf-8") as fout:
            for r in range(world_size):
                shard = os.path.join(args.output_dir, f"predictions.rank{r}.jsonl")
                with open(shard, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
                        total += 1
                os.remove(shard)
        elapsed = time.monotonic() - t0
        logger.info("wrote %d predictions → %s (%.1fs)", total, merged, elapsed)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
