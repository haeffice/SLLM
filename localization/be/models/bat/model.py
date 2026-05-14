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
from typing import Dict

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base import AudioLLM
from .preprocess import (
    AUDIO_SAMPLES,
    SAMPLE_RATE,
    format_prompt,
    preprocess_waveform,
)
from .projector import EncoderProjectorQFormer, QFormerConfig
from .spatial_ast import build_binaural_encoder

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

def _trusted_torch_load(path: str):
    """`torch.load` wrapper that disables the PyTorch 2.6+ safe-pickle default.

    SLAM-LLM/BAT 체크포인트는 가중치 외에 학습 시 사용된 omegaconf
    객체(`ListConfig`, `DictConfig`)를 메타데이터로 포함하고 있다. PyTorch 2.6
    부터 `torch.load`의 기본값이 `weights_only=True`로 바뀌면서 이런 임의
    클래스는 unpickling이 거부된다. 우리는 zhisheng01/SpatialAudio HF 데이터셋
    원본을 신뢰하므로 명시적으로 `weights_only=False`를 사용한다. 또한
    pickle이 ListConfig 같은 클래스를 객체로 복원하려면 그 모듈이 import
    가능해야 하므로 `requirements.txt`에 `omegaconf`를 유지한다.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_encoder_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Spatial-AST 체크포인트 로드. {'model': sd} 또는 sd 그대로 둘 다 지원."""
    blob = _trusted_torch_load(path)
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
            lora[k[len("llm."):]] = v
        elif k.startswith("module.llm."):
            lora[k[len("module.llm."):]] = v
        elif k.startswith("base_model.model.") or k.startswith("model."):
            # 이미 peft 형식인 키 — 그대로 보존 후 peft 모델에 부어넣을 때
            # llm.load_state_dict가 처리하도록 한다.
            lora[k] = v
        else:
            unknown.append(k)

    if unknown:
        sample = unknown[:5]
        logger.warning(
            "BAT projector ckpt: %d keys not recognized (sample: %s)",
            len(unknown), sample,
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
        encoder = build_binaural_encoder()
        enc_state = _load_encoder_state_dict(encoder_ckpt)
        missing_keys, unexpected_keys = encoder.load_state_dict(enc_state, strict=False)
        # ckpt가 모델보다 많은 키를 가질 수 있음 (classification head 등 inference에 불필요한 가지)
        # → unexpected는 무시해도 되지만 어떤 키가 버려졌는지 확인할 수 있게 sample 노출
        applied = len(enc_state) - len(unexpected_keys)
        logger.info(
            "Spatial-AST: applied=%d, missing=%d, unexpected=%d "
            "(sample missing: %s, sample unexpected: %s)",
            applied, len(missing_keys), len(unexpected_keys),
            missing_keys[:5], unexpected_keys[:5],
        )
        if applied == 0:
            logger.warning("No encoder keys matched — Spatial-AST is randomly initialized!")
        encoder.eval()

        logger.info("Loading BAT projector + LoRA delta from %s", projector_ckpt)
        proj_cfg = QFormerConfig(
            encoder_dim=ENCODER_DIM,
            llm_dim=LLM_DIM,
            qformer_layers=QFORMER_LAYERS,
            query_len=QUERY_LEN,
        )
        projector = EncoderProjectorQFormer(proj_cfg)

        bat_state = _trusted_torch_load(projector_ckpt)
        if isinstance(bat_state, dict) and "model" in bat_state and isinstance(bat_state["model"], dict):
            bat_state = bat_state["model"]
        if not isinstance(bat_state, dict):
            raise ValueError(f"Unexpected projector ckpt format: {type(bat_state)}")

        # 디버그: 첫 몇 개 키만 노출 (prefix 추정 잘못된 경우 사용자가 식별 가능)
        sample_keys = list(bat_state.keys())[:6]
        logger.info("BAT projector ckpt top-level keys (sample): %s", sample_keys)

        proj_state, lora_state = _split_projector_lora(bat_state)
        if proj_state:
            mk, uk = projector.load_state_dict(proj_state, strict=False)
            logger.info("Projector load: missing=%d unexpected=%d", len(mk), len(uk))
        else:
            logger.warning("No projector keys found in BAT ckpt — projector remains randomly initialized!")
        projector.eval()

        if lora_state:
            # PEFT 공식 API. `load_state_dict`을 직접 부르면 PEFT 래퍼가 만든
            # 키 prefix(`base_model.model.…`)나 adapter_name(`.default.`) 변형을
            # 우리가 직접 맞춰야 하는데, `set_peft_model_state_dict`는 ckpt 키를
            # 현재 모델의 명명에 맞춰 변환해주고 LoRA 타입에 한정해서 적용한다.
            load_result = set_peft_model_state_dict(llm, lora_state, adapter_name="default")
            mk = list(getattr(load_result, "missing_keys", []) or [])
            uk = list(getattr(load_result, "unexpected_keys", []) or [])
            sample_lora = next(iter(lora_state.keys()), "<none>")
            logger.info(
                "LoRA load via set_peft_model_state_dict: applied=%d delta keys "
                "(sample key: %s), missing=%d (= base LLM 가중치, from_pretrained로 "
                "이미 로드됨), unexpected=%d",
                len(lora_state), sample_lora, len(mk), len(uk),
            )
            if uk:
                logger.warning(
                    "LoRA: %d unexpected keys did NOT match the PEFT model — "
                    "ckpt 키 형식이 예상과 다를 수 있습니다. Sample: %s",
                    len(uk), uk[:5],
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
    def infer(self, wav_bytes: bytes, question: str) -> dict:
        # 1) waveform 전처리
        waveform, n_input_samples = preprocess_waveform(wav_bytes)
        waveform = waveform.to(self.device)  # (1, 2, 320000)

        # 2) 텍스트 프롬프트 + audio placeholder tokens
        prompt = format_prompt(question, None)
        audio_pseudo = torch.full((QUERY_LEN,), -1, dtype=torch.int64)
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
        input_ids = torch.cat([audio_pseudo, prompt_ids]).unsqueeze(0).to(self.device)

        attention_mask = input_ids.ge(-1)            # (1, T) — 모두 True
        modality_mask = input_ids.eq(-1)             # (1, T) — audio 자리만 True

        # 3) 인코더 + 프로젝터
        # encoder 출력은 float32지만 LLM은 float16 → cast 필요
        encoder_outs = self.encoder(waveform)        # (1, 3 + 512, 768)
        proj_out = self.projector(encoder_outs)      # (1, 64, 4096)
        proj_out = proj_out.to(self.llm.dtype if hasattr(self.llm, "dtype") else torch.float16)

        # 4) LLM input embedding 만들고 modality 위치에 proj_out 채워넣기
        safe_ids = input_ids.clamp(min=0)
        inputs_embeds = self.llm.get_input_embeddings()(safe_ids).clone()
        # proj_out의 dtype에 맞춰 inputs_embeds도 일치시킴 (혹시 다를 수 있어)
        if inputs_embeds.dtype != proj_out.dtype:
            inputs_embeds = inputs_embeds.to(proj_out.dtype)

        B, T = modality_mask.shape
        for b in range(B):
            inputs_embeds[b, modality_mask[b]] = proj_out[b]

        # 5) Generate
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
        response = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

        return {
            "response": response,
            "model_id": self.model_id,
            "sample_rate": SAMPLE_RATE,
            "audio_samples": AUDIO_SAMPLES,
            "input_audio_samples": int(n_input_samples),
        }
