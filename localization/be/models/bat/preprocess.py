"""BAT 추론을 위한 오디오/프롬프트 전처리 유틸리티.

원본:
- examples/seld_spatialsoundqa/dataset/spatial_audio_dataset.py
  (`format_prompt`, `SpatialAudioDatasetJsonl.normalize_audio`,
   `SpatialAudioDatasetJsonl.load_waveform`)
- src/slam_llm/datasets/base_dataset.py
  (`BaseDataset.padding`)

학습용 reverb 처리, 두 번째 오디오 믹싱 같은 multi-source 합성 분기는
inference에 불필요하므로 제거.

요약 흐름 (wav_bytes → tensor):
1. soundfile.read → (T, C) numpy
2. 모노이면 stereo로 복제, >2채널이면 처음 두 채널만 사용
3. 32 kHz로 resample_poly
4. (2, T) numpy 로 transpose
5. RMS 기반 -14 dBFS 정규화
6. 10초(320,000 samples)로 padding/crop
7. torch.float32 텐서 (2, 320000)으로 반환
"""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
from scipy import signal


SAMPLE_RATE = 32000
AUDIO_SAMPLES = 10 * SAMPLE_RATE  # 320,000
TARGET_DBFS = -14.0


# -----------------------------------------------------------------------------
# Prompt formatting (Alpaca / Stanford instruction template, no input variant)
# -----------------------------------------------------------------------------

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


def format_prompt(instruction: str, input: str | None = None) -> str:
    """BAT가 학습/평가 시 사용하는 Alpaca-스타일 프롬프트로 감싼다."""
    if input is None:
        return _PROMPT_NO_INPUT.format(instruction=instruction)
    return _PROMPT_WITH_INPUT.format(instruction=instruction, input=input)


# -----------------------------------------------------------------------------
# Audio normalization / padding
# -----------------------------------------------------------------------------

def normalize_audio(audio_data: np.ndarray, target_dBFS: float = TARGET_DBFS) -> np.ndarray:
    """RMS-기반 dBFS 정규화. 완전한 무음이면 그대로 반환."""
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms == 0:
        return audio_data
    current_dBFS = 20 * np.log10(rms)
    gain_dB = target_dBFS - current_dBFS
    gain_linear = 10 ** (gain_dB / 20)
    return audio_data * gain_linear


def padding(sequence: torch.Tensor, padding_length: int, padding_idx: float = 0.0,
            padding_side: str = "right") -> torch.Tensor:
    """오디오 시퀀스를 padding_length만큼 늘리거나 잘라낸다.

    원본 BaseDataset.padding의 텐서 케이스만 발췌. 음수 padding_length이면 crop.
    """
    if sequence.ndimension() == 2:
        if padding_length >= 0:
            return torch.nn.functional.pad(sequence, (0, padding_length), value=padding_idx)
        return sequence[:, :padding_length]
    if padding_length >= 0:
        pad = torch.full(
            [padding_length] + list(sequence.size())[1:],
            padding_idx,
            dtype=sequence.dtype,
            device=sequence.device,
        )
        if padding_side == "left":
            return torch.cat((pad, sequence))
        return torch.cat((sequence, pad))
    return sequence[:padding_length]


# -----------------------------------------------------------------------------
# Top-level: wav bytes → (1, 2, 320000) torch tensor
# -----------------------------------------------------------------------------

def preprocess_waveform(wav_bytes: bytes) -> Tuple[torch.Tensor, int]:
    """wav 바이트를 BAT 인코더가 받을 수 있는 binaural 텐서로 변환.

    Returns:
        waveform: (1, 2, 320000) float32 tensor — 배치=1, stereo, 10s @ 32kHz
        n_samples: 원본 (resample 전) 샘플 수 — 디버그/응답용
    """
    audio, sr = sf.read(io.BytesIO(wav_bytes), always_2d=True)  # (T, C)
    n_input_samples = audio.shape[0]

    # 1) 채널 정리 → 정확히 2채널 stereo
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    # 2) 32 kHz로 리샘플
    if sr != SAMPLE_RATE:
        # 채널별로 따로 resample (resample_poly의 axis 옵션 사용)
        audio = signal.resample_poly(audio, SAMPLE_RATE, sr, axis=0)

    # 3) (2, T) 로 transpose
    audio = np.ascontiguousarray(audio.T, dtype=np.float32)  # (2, T)

    # 4) RMS 정규화 (좌우 합쳐 한번에)
    audio = normalize_audio(audio, TARGET_DBFS).astype(np.float32)

    # 5) 10초로 패딩/크롭
    waveform = torch.from_numpy(audio)  # (2, T)
    cur_len = waveform.shape[1]
    pad_len = AUDIO_SAMPLES - cur_len
    waveform = padding(waveform, padding_length=pad_len, padding_idx=0.0)

    # 안전장치 — 부동소수 보정 등으로 길이가 안 맞을 경우
    if waveform.shape[1] != AUDIO_SAMPLES:
        waveform = waveform[:, :AUDIO_SAMPLES]
        if waveform.shape[1] < AUDIO_SAMPLES:
            waveform = torch.nn.functional.pad(
                waveform, (0, AUDIO_SAMPLES - waveform.shape[1]), value=0.0
            )

    return waveform.unsqueeze(0), n_input_samples  # (1, 2, 320000)
