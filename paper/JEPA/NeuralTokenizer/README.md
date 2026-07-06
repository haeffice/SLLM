# JEPA as a Neural Tokenizer (arXiv:2512.07168)

**가역(reversible) 음성 토크나이저**를 JEPA로 학습하는 2-stage 파이프라인의
PyTorch 2.8 자가완결 재구현. raw 24 kHz waveform을 **2.5 Hz**의 의미
표현으로 압축하고, FSQ + mixed-radix 패킹으로 **LM-friendly한 이산 토큰
(47.5 tok/s)** 을 만든 뒤 HiFi-GAN으로 파형을 복원한다. 이 repository가
지향하는 **Speech-LLM** 의 입력단(speech tokenizer)에 가장 직접 맞닿는
논문이다.

- **Paper:** *JEPA as a Neural Tokenizer: Learning Robust Speech
  Representations with Density Adaptive Attention*, Ioannides, Constantinou,
  Chadha, Elkins, Pang, Shwartz-Ziv, **LeCun** (2025-12) —
  https://arxiv.org/abs/2512.07168
- **Lineage:** I-JEPA (2301.08243) → data2vec/audio JEPA → **JEPA Neural
  Tokenizer** (JEPA SSL + 신경 코덱의 결합).

## 이 논문을 선정한 이유

JEPA 최신 트렌드 중 **오디오/음성** 축의 대표작이자 LeCun 공저 논문으로,
masked-latent JEPA(표현 학습)와 **가역 토크나이저**(LM 입력)를 한 파이프라인
으로 결합한다. 기존 neural codec(EnCodec 등) 대비 (1) 파형 재구성에서
분리된 JEPA 의미 표현, (2) **DAAM**(밀도 적응 게이팅)으로 2.5 Hz의 낮은
프레임레이트에서 계층적 음성 구조 발견, (3) 학습 codebook 없이 **FSQ +
mixed-radix**로 완전 가역·고압축 토큰을 만든다는 점이 신선하다.

## 2-stage 구조

**Stage 1 — JEPA + DAAM 인코더 (SSL)** · `Stage1Trainer`
- Conv1D 다운샘플 스택: 24 kHz → **2.5 Hz** (hop 9600), 채널 64→128→256→384→512→512, downsample stride `[8,8,5,5,6]`.
- **DAAM**: K=4 Gaussian-mixture 게이팅 — `y_t = x_t·exp(α·logG(x_t))` (α=0.05).
- 8× **Conformer** 블록(d=512, 16 heads).
- **block masking**(ρ=0.5, span 2…T/4) + **EMA teacher**(τ=0.996) + 경량 predictor → masked-latent MSE.

**Stage 2 — FSQ + mixed-radix + HiFi-GAN (복원)** · `Stage2Trainer`
- Stage-1 인코더 **freeze**. `proj_in(512→128)` → **FSQ**(per-dim level 4, tanh-bound, STE) → `proj_out(128→512)`.
- **mixed-radix packing**: 128차원을 G=7로 묶어 19 그룹 → `token = Σ iₖ·∏ rⱼ` (가역, vocab 4⁷=16384/group) → **47.5 tok/s** @ 2.5 Hz.
- **HiFi-GAN** 디코더(ConvTranspose1D, stride `[6,5,5,8,8]`, MRF res-block) → 파형.
- **손실**: L1 + multi-resolution STFT(λ=2.0) + GAN(MPD+MSD, feature matching, λ=0.1, disc warmup 5000) — 2-옵티마이저(G/D) 루프를 `transformers.Trainer` 서브클래스가 `training_step` 안에서 처리.

## 구성 파일

| 파일 | 역할 |
|---|---|
| `NeuralTokenizer.py` | 배포용 토크나이저(인코더+DAAM+Conformer / FSQ / mixed-radix codec / HiFi-GAN) + `tokenize`/`detokenize`/`reconstruct` + `from_checkpoint` |
| `NeuralTokenizer_Trainer.py` | `Stage1Trainer`(JEPA SSL) + `Stage2Trainer`(FSQ+디코더) + MPD/MSD 판별자 + MR-STFT 손실 |
| `train_neural_tokenizer.py` | `stage:` 분기 학습 — stage1은 표준 Trainer, stage2는 GAN Trainer 서브클래스 |
| `eval_neural_tokenizer.py` | wav→tokens→wav 라운드트립: token rate / 가역성 / 복원 L1·MR-STFT |
| `config.yaml` | 학습/평가 단일 설정 (`stage` 선택자) |
| `run_train_NeuralTokenizer.sh` | torchrun 런처 (기존 checkpoint 존재 시 시작 거부) |
| `run_eval_NeuralTokenizer.sh` | 평가 런처 |
| `make_synthetic_manifest.py` | CPU 스모크용 합성 24 kHz wav + stage1/stage2 config 생성 |
| `requirements.txt` | 의존성 (torch==2.8.0, soundfile, scipy) |

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt
```

## 데이터셋

라벨 없는 음성 코퍼스. 매니페스트:

```json
{"data": [{"audio_id": "spk/utt0001.wav"}, ...]}
```

`audio_path = data.audio_root / audio_id`. 임의 SR/채널을 24 kHz mono로
리샘플·RMS 정규화한다. 논문은 **LibriLight**(~9k h, 영어) 사용:
https://github.com/facebookresearch/libri-light

## 학습 (2-stage)

```bash
# Stage 1 — JEPA SSL 인코더 (config.yaml: stage: 1)
bash run_train_NeuralTokenizer.sh config.yaml

# Stage 2 — FSQ+HiFi-GAN (config: stage: 2, train.stage1_ckpt = stage1 .pt)
bash run_train_NeuralTokenizer.sh config_stage2.yaml
```

학습 로그: 모듈별 학습 파라미터 수(+frame/token rate) · 첫 배치 첫 샘플의
**wav 경로**(feed 전) · `logging_steps`마다 step/train_loss/sub-loss
(stage1: mask_frac / stage2: l1·stft·adv·fm·d_loss)/lr · `save_steps`마다
step 번호가 들어간 checkpoint(`nt_stage1_step{N}.pt` / `nt_stage2_step{N}.pt`,
`NeuralTokenizer.from_checkpoint`로 로드). 0개 로드면 즉시 중단.

## CPU 스모크 테스트

```bash
python make_synthetic_manifest.py --out /tmp/nt_smoke
bash run_train_NeuralTokenizer.sh /tmp/nt_smoke/config.yaml          # stage 1
bash run_train_NeuralTokenizer.sh /tmp/nt_smoke/config_stage2.yaml   # stage 2 (GAN)
bash run_eval_NeuralTokenizer.sh /tmp/nt_smoke/config_stage2.yaml \
     /tmp/nt_smoke/stage2_ckpts/nt_stage2_step4.pt
```

(스모크 config는 hop을 512로 줄여 CPU에서 수 step 만에 끝나며,
`disc_warmup: 0`으로 GAN 경로까지 실행한다.)

## 평가 (round-trip)

```bash
bash run_eval_NeuralTokenizer.sh config.yaml /path/to/ckpts/.../nt_stage2_step29000.pt
```

token rate(tok/s)·frame rate(Hz), **mixed-radix 가역성**(unpack∘pack==identity),
복원 **L1 / MR-STFT** 를 출력한다. `from_checkpoint`이 `model.eval()` +
`requires_grad=False`를 적용하며, Stage-1 전용 ckpt를 로드하면 디코더가
missing이라 partial-load 경고가 정상적으로 뜬다(단, 0개 로드면 즉시 예외).
