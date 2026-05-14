"""BAT (Spatial-AST + Q-Former + Llama-2-7b) inference adapter.

SLAM-LLM의 `slam_model_seld` model_factory와 `slam_model.forward/generate`
파이프라인에서 inference에 필요한 부분만 직접 옮겨와, AudioLLM 인터페이스에
맞게 재구성한다. 외부 SLAM-LLM 패키지에 의존하지 않는다.

가중치 파일 (환경변수로 지정):
  - BAT_LLAMA_PATH    : Llama-2-7b HF checkpoint directory
                        (AutoTokenizer/AutoModelForCausalLM로 로드)
  - BAT_ENCODER_CKPT  : Spatial-AST finetuned.pth
                        ({'model': state_dict} 또는 state_dict 둘 다 지원)
  - BAT_PROJECTOR_CKPT: BAT model.pt
                        (Q-Former projector + LoRA delta가 합쳐진 체크포인트)

추론 흐름:
  1. wav_bytes → (1, 2, 320000) waveform @ 32 kHz, stereo
  2. encoder(waveform) → (1, 3 + 512, 768)
  3. projector(encoder_out) → (1, 64, 4096)
  4. format_prompt(question) + 64 audio-placeholder tokens (id = -1)
  5. modality_mask로 audio embed를 LLM input embed에 채워넣고 llm.generate
  6. tokenizer.batch_decode → response 문자열
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Dict

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base import AudioLLM
from ._ckpt import trusted_torch_load
from .preprocess import (
    AUDIO_SAMPLES,
    SAMPLE_RATE,
    format_prompt,
    preprocess_waveform,
)
from .projector import EncoderProjectorQFormer, QFormerConfig
from .spatial_ast_frozen import SpatialASTFrozen

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants (examples/seld_spatialsoundqa/seld_config.py 값을 그대로 옮김)
# -----------------------------------------------------------------------------

ENCODER_DIM = 768
LLM_DIM = 4096
QFORMER_LAYERS = 8
QUERY_LEN = 64

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_NUM_BEAMS = 4


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------

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
    """BAT model.pt를 두 그룹으로 분리.

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
            # 이미 peft 형식인 키 — `.default` 중복만 방지하고 그대로 보존.
            lora[k.replace(".default", "")] = v
        else:
            unknown.append(k)

    if unknown:
        logger.warning(
            "BAT projector ckpt: %d keys not recognized",
            len(unknown),
        )
    return proj, lora


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class BAT(AudioLLM):
    model_id = "bat"

    def __init__(self, device, encoder, projector, llm, tokenizer):
        super().__init__(device)
        self.encoder = encoder
        self.projector = projector
        self.llm = llm
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, device: torch.device) -> "BAT":
        llama_path = os.environ.get("BAT_LLAMA_PATH")
        encoder_ckpt = os.environ.get("BAT_ENCODER_CKPT")
        projector_ckpt = os.environ.get("BAT_PROJECTOR_CKPT")
        missing = [
            name for name, val in [
                ("BAT_LLAMA_PATH", llama_path),
                ("BAT_ENCODER_CKPT", encoder_ckpt),
                ("BAT_PROJECTOR_CKPT", projector_ckpt),
            ] if not val
        ]
        if missing:
            raise EnvironmentError(
                f"BAT requires env vars: {', '.join(missing)}"
            )

        logger.info("Loading BAT tokenizer + LLM from %s", llama_path)
        tokenizer = AutoTokenizer.from_pretrained(llama_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # batch generation은 causal LM 특성상 left-padding이 필요
        # (오른쪽 끝 토큰에서 다음 토큰을 이어 생성하므로 모든 시퀀스의 종단이 정렬돼야 함)
        tokenizer.padding_side = "left"

        llm = AutoModelForCausalLM.from_pretrained(
            llama_path,
            dtype=torch.float16,
        )
        peft_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, peft_cfg)
        llm.eval()

        # max_new_tokens를 명시적으로 쓰므로 max_length 디폴트 경고 억제
        if getattr(llm, "generation_config", None) is not None:
            llm.generation_config.max_length = None

        logger.info("Loading Spatial-AST encoder from %s", encoder_ckpt)
        # SLAM-LLM의 src/slam_llm/models/encoder.py:SpatialASTEncoder.load와 동일한
        # 생성 인자. num_classes=355는 AudioSet head 차원이라 forward 경로에선 안 쓰이지만
        # ckpt 키 모양을 맞추기 위해 그대로 둔다.
        encoder = SpatialASTFrozen(
            ckpt_path=encoder_ckpt,
            freeze=True,
            logger=logger,
            num_classes=355
        )

        logger.info("Loading BAT projector + LoRA delta from %s", projector_ckpt)
        proj_cfg = QFormerConfig(
            encoder_dim=ENCODER_DIM,
            llm_dim=LLM_DIM,
            qformer_layers=QFORMER_LAYERS,
            query_len=QUERY_LEN,
        )
        projector = EncoderProjectorQFormer(proj_cfg)

        bat_state = trusted_torch_load(projector_ckpt)
        if isinstance(bat_state, dict) and "model" in bat_state and isinstance(bat_state["model"], dict):
            bat_state = bat_state["model"]
        if not isinstance(bat_state, dict):
            raise ValueError(f"Unexpected projector ckpt format: {type(bat_state)}")

        # encoder_projector.* 키 존재 여부 빠른 확인 (projector 가중치가 들어있는지)
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
            logger.warning("No projector keys found in BAT ckpt — projector remains randomly initialized!")
        projector.eval()

        if lora_state:
            # PEFT 공식 API. `load_state_dict`을 직접 부르면 PEFT 래퍼가 만든
            # 키 prefix(`base_model.model.…`)나 adapter_name(`.default.`) 변형을
            # 우리가 직접 맞춰야 하는데, `set_peft_model_state_dict`는 ckpt 키를
            # 현재 모델의 명명에 맞춰 변환해주고 LoRA 타입에 한정해서 적용한다.
            load_result = set_peft_model_state_dict(llm, lora_state, adapter_name="default")
            uk = list(getattr(load_result, "unexpected_keys", []) or [])
            # PEFT 모델의 missing은 늘 base LLM 가중치 수만큼 잡힘(이미 from_pretrained로
            # 로드돼 있음)이라 LoRA 적용 성공 여부는 unexpected만 보면 됨.
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

        # 디바이스로 이동
        encoder = encoder.to(device)
        projector = projector.to(device)
        # peft 모델은 dtype을 유지해야 함 — float16 그대로
        llm = llm.to(device)

        return cls(device, encoder, projector, llm, tokenizer)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(self, wav_bytes: bytes, questions: list[str]) -> dict:
        if not questions:
            raise ValueError("at least one question is required")

        # 1) waveform 전처리 (단일 오디오, 모든 question에 대해 공유)
        waveform, n_input_samples = preprocess_waveform(wav_bytes)
        waveform = waveform.to(self.device)  # (1, 2, 320000)

        # 2) 각 question을 Alpaca 템플릿으로 감싸고 left-padding으로 batch 화
        prompts = [format_prompt(q, None) for q in questions]
        logger.info("Batch input prompts (B=%d):", len(prompts))
        for i, p in enumerate(prompts):
            logger.info("  [%d] %s", i, p)

        encoded = self.tokenizer(
            prompts,
            padding=True,                # tokenizer.padding_side='left' (load 시 설정)
            return_tensors="pt",
        )
        prompt_ids = encoded.input_ids                  # (B, max_prompt_len)
        prompt_attn = encoded.attention_mask            # (B, max_prompt_len)

        B = prompt_ids.size(0)
        audio_pseudo = torch.full((B, QUERY_LEN), -1, dtype=prompt_ids.dtype)
        audio_attn = torch.ones((B, QUERY_LEN), dtype=prompt_attn.dtype)

        # 구조: [audio×64, (pad×P), prompt_tokens]
        input_ids = torch.cat([audio_pseudo, prompt_ids], dim=1).to(self.device)
        attention_mask = torch.cat([audio_attn, prompt_attn], dim=1).to(self.device)
        modality_mask = input_ids.eq(-1)                # (B, T)

        # 3) 인코더 + 프로젝터 (오디오 1회만 실행, batch 차원으로 expand)
        encoder_outs = self.encoder(waveform)           # (1, 3+512, 768)
        proj_out = self.projector(encoder_outs)         # (1, 64, 4096)
        proj_out = proj_out.to(self.llm.dtype if hasattr(self.llm, "dtype") else torch.float16)
        proj_out = proj_out.expand(B, -1, -1)           # (B, 64, 4096) — 동일 audio 공유

        # 4) LLM input embedding + modality 자리에 audio embed 채워넣기
        safe_ids = input_ids.clamp(min=0)
        inputs_embeds = self.llm.get_input_embeddings()(safe_ids).clone()
        if inputs_embeds.dtype != proj_out.dtype:
            inputs_embeds = inputs_embeds.to(proj_out.dtype)

        for b in range(B):
            inputs_embeds[b, modality_mask[b]] = proj_out[b]

        # 5) Generate (batch)
        out_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=DEFAULT_NUM_BEAMS,
            do_sample=False,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            min_length=1,
            top_p=1.0,
            repetition_penalty=1.0,
            length_penalty=1.0,
            temperature=1.0,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        responses = [
            r.strip()
            for r in self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        ]

        # FE의 기존 `response` 필드 호환 위해 Q/A 형식 묶음 문자열도 동봉
        joined = "\n\n".join(
            f"Q: {q}\nA: {r}" for q, r in zip(questions, responses)
        )

        return {
            "responses": responses,
            "questions": questions,
            "response": joined,
            "model_id": self.model_id,
            "sample_rate": SAMPLE_RATE,
            "audio_samples": AUDIO_SAMPLES,
            "input_audio_samples": int(n_input_samples),
            "batch_size": B,
        }
