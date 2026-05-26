# PhaseCoder — 마이크 배열 형상에 무관한 공간 오디오 인코더

스피커/마이크 배열이 받은 **다채널 오디오 + 각 마이크의 3-D 좌표**를 입력받아, 배열의
형상(마이크 개수·배치)에 **무관한** 압축 표현 — **spatial audio token** — 을 만든다.
배열 정보가 가중치가 아니라 **positional encoding**으로만 들어가므로, 한 번 학습한
모델이 **임의의 마이크 개수/배치**(논문: 3–8개, 구경 7–18 cm)에 그대로 적용된다.
이 토큰으로 ① 음원의 **방위각/고도/거리**를 추정하고, ② 멀티모달 LLM(Gemma 3n)에
주입해 **공간 추론·특정 방향 화자 전사**를 수행한다.

> Artem Dementyev, Wazeer Zulfikar, Sinan Hersek, Pascal Getreuer, et al.,
> **"PhaseCoder: Microphone Geometry-Agnostic Spatial Audio Understanding for
> Multimodal LLMs"**, Google DeepMind & Google AR, 2026.
> arXiv: [2601.21124](https://arxiv.org/abs/2601.21124) (2026-01-28)

**공개 코드/데이터:** 작성 시점 기준 **공식 코드 공개는 확인되지 않았다.** 따라서 본
구현은 논문 본문·수식 서술에 충실하게 from-scratch 작성했다. 마이크 positional
encoding(MPE)은 논문이 명시적으로 차용한 **GI-DOAENet**의 수식을 따른다:
- GI-DOAENet: *"DNN-Based Geometry-Invariant DOA Estimation With Microphone
  Positional Encoding and Complexity Gradual Training"*, IEEE TASLP 2025
  (α=7.0, β=4.0 상수의 출처).
공식 코드가 공개되면 README에 경로를 추가하고 그에 맞춰 재정렬할 것.

## 이 논문을 선정한 이유

- **요청 정합성:** "2026년 1월 구글에서 발표한 PhaseCoder"를 직접 지정한 요청. 본 논문은
  Google DeepMind·Google AR가 2026-01-28 arXiv에 공개한 바로 그 PhaseCoder다.
- **저장소 주제와의 연결:** 본 repo는 Speech-LLM(`demo/`, `localization/`)과 공간
  음향(`paper/AudioSpaceMap/`)을 다룬다. PhaseCoder는 **공간 음향 + LLM**을 잇는,
  저장소 흐름에 정확히 부합하는 최신 연구다.
- **형상 무관(geometry-agnostic) 일반화:** 대부분의 DOA/공간 모델이 특정 배열에
  고정되어 재학습이 필요한 반면, PhaseCoder는 마이크 좌표를 positional encoding으로
  넣어 **하나의 모델로 임의 배열**을 처리한다(핵심 기여).
- **경량·실용:** D=256, 5 블록, ~수백만 파라미터의 소형 인코더로 모바일/AR 장치에
  적합하며, USM 기반 mono 오디오 토큰과 나란히 LLM에 주입된다.
- **데이터 획득 가능성:** 학습은 **시뮬레이션 RIR + LibriSpeech**(아래 다운로드
  경로 명시), 평가는 공개 벤치마크 **LOCATA**(Zenodo)와 **RSL2019**(요청 시)로
  재현 가능하다.

## 아키텍처 (논문 Sec. 3)

```
waveform  X ∈ R^{C×T}   +  mic coords  M ∈ R^{C×3}      C mics, 16 kHz, ~250 ms(=4096 샘플, 33 프레임)
  └─ 채널별 STFT             Hann win=256, hop=128 → 129 bins
  └─ patch feature           per (mic,frame): [ |S|(129) ‖ ∠S(129) ] = 258 → Linear → D=256
  └─ + positional encodings  ① 1-D sinusoidal(시퀀스) ⊕ ② frame-level sinusoidal ⊕ ③ MPE(마이크 구면좌표)
  └─ [CLS] + L=C·F patches → Transformer (5 blocks, 4 heads, D=256, FFN 256, GELU, pre-norm)
  └─ CLS → MLP(256→256→256, ReLU) = spatial audio token  z ∈ R^{256}
  ├─ heads   azimuth(38) / elevation(18) / distance(13)   (softmax 분류, 각 +no-source)
  └─ projector  z → R^{2048}  (2-layer GELU MLP; Gemma 3n 주입용)
```

- **MPE (마이크 positional encoding, GI-DOAENet):** 마이크 좌표를 배열 중심 기준 구면
  좌표(반경 r, 고도 θ, 방위 φ)로 변환 후
  `P_i = α·r_i·[cos(2πβv+θ), sin(2πβv+θ), cos(2πβv+φ), sin(2πβv+φ)]`,
  `v=(4/D)·[0..D/4−1]`, α=7.0, β=4.0. 이 D차원 벡터를 그 마이크의 모든 패치에 더한다.
- **형상 무관 배치:** 시퀀스/프레임/MPE 모두 길이에 따라 **실행 시 계산**되고 학습
  파라미터는 [CLS]·가중치뿐이라, 한 배치 안에서도 마이크 개수가 다른 샘플을
  패딩 + `channel_mask`(attention padding)로 함께 학습한다.
- **손실 (Sec. 3):** `L = λ_az·CE + λ_el·CE + λ_di·CE`, λ=(1.0, 1.0, 0.5).

## 파일 구성

| 파일 | 역할 |
|---|---|
| `PhaseCoder.py` | 추론 모델(STFT front end + MPE + ViT + heads) + LLM projector + 손실/지표 + `from_checkpoint`/`predict`. |
| `make_phasecoder_dataset.py` | **합성 공간 데이터셋 생성기**. free-field(numpy) 기본 / pyroomacoustics RIR(`--sim rir`). 샘플마다 무작위 3–8 mic 배열. |
| `train_phasecoder.py` | HF `PreTrainedModel` 래퍼 + `Trainer` 학습 + 가변 마이크 수 collate + 콜백. |
| `eval_phasecoder.py` | 학습과 분리된 load/inference + az/el/dist top-1 정확도 + 각도/거리 MAE. |
| `config.yaml` | 모든 학습/모델/데이터 인자(단일 진실 공급원). |
| `run_train_PhaseCoder.sh` | 실행 스크립트(yaml만 인자, 기존 체크포인트 존재 시 시작 거부). |
| `requirements.txt` | CPU 의존성, torch==2.8.0. |

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt
# pyroomacoustics는 --sim rir 일 때만, soundfile은 --speech-dir 일 때만 필요
```

## 데이터셋 준비 (다운로드 / 획득 방법)

PhaseCoder의 학습 데이터는 단일 다운로드가 아니라 **시뮬레이션으로 생성**한다.

### (A) 합성 데이터 — 본 저장소 생성기 (다운로드 불필요)

`make_phasecoder_dataset.py`가 음원 방향/거리 단서(채널 간 위상·시간차, 수신
레벨)를 갖는 다채널 클립을 만든다. 매 샘플 **무작위 배열**(3–8 mic, 7–18 cm)을 뽑으므로
형상 무관 학습이 가능하다.

```bash
# free-field (numpy 전용, 빠름): 위상/TDOA 단서로 방위/고도 학습
python make_phasecoder_dataset.py --out data/train --n 200000 --seed 0
python make_phasecoder_dataset.py --out data/val   --n 4000   --seed 1

# RIR (pyroomacoustics, 잔향 포함 → 거리 단서 현실화; 논문에 더 가까움)
python make_phasecoder_dataset.py --out data/train --n 200000 --sim rir --seed 0
```

### (B) 실 음원/벤치마크 (논문 재현용)

- **음원(speech):** **LibriSpeech** (논문 학습 소스). HuggingFace:
  [`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr)
  또는 원본 <https://www.openslr.org/12>. 받은 폴더를
  `--speech-dir /path/to/librispeech` 로 넘기면 합성 소스 대신 실제 발화를 사용한다
  (이때 `soundfile` 필요).
- **distractor 소음:** **Freesound** <https://freesound.org> (논문이 사용한 방해 음원).
- **평가 — LOCATA:** 공개 코퍼스. Zenodo <https://zenodo.org/records/3630471> /
  공식 <https://www.locata.lms.tf.fau.de/datasets/>. (NAO 12-mic 중 8개 사용)
- **평가 — RSL2019:** 4-mic 실측 화자 위치 코퍼스. 프로젝트 페이지
  <https://bidishasharma.github.io/RSL2019/> 에서 **연구 목적 요청 시 제공**.

실 데이터를 쓰려면 각 코퍼스를 위 (A)와 동일한 `.npz`(audio (C,T), mic_coords (C,3),
az/el/dist 라벨) + `manifest.json` 형식으로 변환해 주입하면 된다(좌표·라벨 로더는
코퍼스 라이선스상 본 저장소에 포함하지 않음).

### CPU smoke 테스트 (다운로드 없이 배관 검증)

```bash
python make_phasecoder_dataset.py --out data/train --n 24 --seed 0
python make_phasecoder_dataset.py --out data/val   --n 8  --seed 1
# config.yaml의 train/valid_manifest를 위 경로로 맞춘 뒤:
NPROC_PER_NODE=1 ./run_train_PhaseCoder.sh config.yaml
```

## 학습

```bash
# config.yaml의 data.train_manifest / train.output_dir 등을 채운 뒤:
NPROC_PER_NODE=8 ./run_train_PhaseCoder.sh config.yaml
```

- 옵티마이저: AdamW + linear-warmup + cosine. `config.yaml`이 유일한 인자.
- 논문 2-stage 커리큘럼: stage1(클린, 670k step, lr 1e-4) → stage2(소음/SNR −5~15 dB,
  30k step, lr 1e-5). 본 구현은 단일 stage가 기본이며, stage2는 소음 데이터를 추가
  생성하고 `optim.learning_rate`/`init_from`(stage1 체크포인트)로 이어 학습하면 된다.
- 로그: 모듈별 학습 가능 파라미터 수, 첫 배치 첫 샘플의 **오디오 경로**(feed 이전),
  `logging_steps`마다 step/train_loss/valid_loss/lr, `save_steps`마다
  `phasecoder_step{step}.pt` 저장.
- `run_train_PhaseCoder.sh`는 `output_dir`에 체크포인트가 있으면 시작을 거부한다.

## 평가

```bash
python eval_phasecoder.py --config config.yaml \
    --checkpoint /path/to/ckpts/phasecoder/phasecoder_final.pt \
    --manifest data/test/manifest.json
```

- `PhaseCoder.from_checkpoint`가 파라미터를 **freeze + eval()** 하고 0개 매칭 시 로드를
  거부한다(무작위 가중치 추론 사고 방지). `predict`도 `eval()/no_grad` 재확인.
- 출력: 방위/고도/거리 **top-1 정확도**, 방위/고도 **각도 MAE(°)**, 거리 **MAE(m)**.

## LLM 통합 (Gemma 3n)

`SpatialTokenProjector`(256→2048→2048 GELU)가 spatial token을 LLM 임베딩 공간으로
사영한다. 논문은 30초 입력을 160 ms hop으로 재샘플해 ~188개의 spatial token을 만들고
`[BSA] T₁…T₁₈₈ [ESA] [mono audio tokens]` 형태로 prepend한 뒤, Gemma 3n(gemma-3n-e4b-it)을
LoRA(r=8, α=16)로 미세조정한다. 본 저장소는 **인코더 + projector**까지 제공하며, LLM
미세조정 자체(LoRA·5-stage 태스크 커리큘럼)는 별도 대형 파이프라인이라 범위에서 제외했다.

## 추론 예시

```python
import torch
from PhaseCoder import PhaseCoder, CLIP_SAMPLES
m = PhaseCoder.from_checkpoint("phasecoder_final.pt")     # freeze + eval
audio = torch.randn(1, 5, CLIP_SAMPLES)                    # (B, C=5 mics, T)
coords = torch.randn(1, 5, 3) * 0.05                       # 마이크 3-D 좌표 (m)
out = m.predict(audio, coords)
print(out["azimuth_deg"], out["elevation_deg"], out["distance_m"])
```

## 참고 / 주의사항

- **파라미터 수:** 논문 명시 하이퍼파라미터(D=256, 5 blocks, 4 heads, **FFN 256(1×)**)를
  그대로 구현하면 학습 가능 파라미터는 **약 2.2 M**이다. 논문은 "≈6 M"으로 보고하는데,
  본문에 충분히 명시되지 않은 추가 구성요소(예: FFN 확장 배수, 별도 feature 레이어)에서
  비롯된 차이로 보인다. 본 구현은 **명시된 값에 충실**하며, `model.ffn_dim`을 키우면
  보고치에 근접시킬 수 있다(설정으로 노출).
- **MPE 각도 규약:** GI-DOAENet 수식대로 θ=고도, φ=방위로 두고, 배열 중심 기준 구면
  좌표(`φ=atan2(y,x)`, `θ=atan2(z, hypot(x,y))`)를 사용한다.
- **free-field 거리 단서의 한계:** 기본 free-field 시뮬은 잔향이 없어 거리가 1/r 레벨
  단서로만 표현되어(소스 RMS를 1로 정규화해 부여) 가장 약한 태스크다. 논문처럼 거리
  단서를 현실화하려면 `--sim rir`(잔향, DRR)로 생성할 것.
- **under-specified 세부:** patch projection 뒤 LayerNorm, pre-norm Transformer는
  논문에 명시되지 않아 표준 ViT 관례로 선택했다. 공식 코드 공개 시 정렬 예정.
- **클립 길이:** 논문의 "250 ms, 33 프레임"에 맞춰 hop=128 기준 33 프레임이 나오는
  4096 샘플(256 ms)을 기본값으로 사용한다.
