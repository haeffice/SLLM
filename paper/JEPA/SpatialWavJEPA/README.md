# Spatial WavJEPA (WavJEPA-Nat)

Spatial 정보를 활용하는 자기지도학습(SSL) 오디오 인코더. WavJEPA
([arXiv:2509.23238](https://arxiv.org/abs/2509.23238))의 멀티채널 변형인
**WavJEPA-Nat**을 논문에 충실하게 구현했으며, AudioSet 파형에 **binaural
room impulse response(BRIR)** 를 convolution 하여 만든 2채널 공간 오디오로
레이블 없이 JEPA 사전학습한다.

## 아키텍처

```
binaural wav (B, 2, 32159) @ 16 kHz, 약 2초
  ├─ ch0 ─► ConvFeatureExtractor #1 (wav2vec2, 320x stride) ─► (B, 200, 512)
  └─ ch1 ─► ConvFeatureExtractor #2 (독립)                   ─► (B, 200, 512)
        공유 LayerNorm → 공유 Linear(512→768)
        channel-major concat → (B, 400, 768)
        + 2D sincos 위치 임베딩 (channel × time)
        → 12-layer Transformer encoder (d=768, h=12, FF=3072)   ◄ 배포 대상
JEPA: EMA teacher 타깃(data2vec-2 top-K) + mask-token decoder predictor
```

- 채널별 **독립** conv 인코더 2개(WavJEPA-Nat), 두 채널을 concat 한
  **2N = 400** 토큰 시퀀스를 공유 Transformer 가 처리.
- 2D **(channel × time)** sinusoidal 위치 임베딩. context/target mask
  블록은 **두 채널에서 공유**된다.
- **추론 시에는 context(student) 인코더를 사용** — teacher / decoder /
  predictor 는 학습 전용이며 추론 체크포인트에서 제거된다(논문 및
  `WavJEPA.py`의 체크포인트 필터링으로 확인).

## 파일 구성

| 파일 | 역할 |
|---|---|
| `SpatialWavJEPA.py` | 추론 모델 + `make_sincos_pos_embed_2d` + `from_checkpoint`. |
| `SpatialWavJEPA_Trainer.py` | `SpatialWavJEPATrainer` (`WavJEPATrainer` 상속). |
| `train_spatial_wavjepa.py` | Dataset + HF `Trainer` 학습 파이프라인 + 콜백. |
| `config.yaml` | 모든 학습 인자(단일 진실 공급원). |
| `run_train_SpatialWavJEPA.sh` | 실행 스크립트(yaml만 인자, 기존 체크포인트 존재 시 시작 거부). |
| `make_synthetic_manifest.py` | CPU smoke 테스트용 합성 데이터 생성기. |
| `requirements.txt` | CPU 의존성, torch==2.8.0. |

`../WavJEPA/{WavJEPA,WavJEPA_Trainer}.py`(conv front-end, span masker,
EMA/타깃/loss/decoder)를 재사용하므로 해당 디렉터리를 함께 유지해야 한다.

## 데이터

매니페스트 JSON: `{"data": [{"audio_id": "x.wav", "reverb_id": "r.npy"}, ...]}`.
`audio_path = audio_root/audio_id`(mono AudioSet wav, SR 무관),
`reverb_path = reverb_root/reverb_id`(binaural BRIR `.npy`, shape `(2, R)`).
파이프라인(BAT 방식): mono → 16 kHz 리샘플 → RMS 정규화 → BRIR `fftconvolve`
→ binaural → 약 2초 랜덤 crop → joint `(C,T)` zero-mean/unit-std
(채널 간 레벨 단서 ILD 보존).

## 학습

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt
# config.yaml 경로 수정 후:
NPROC_PER_NODE=8 ./run_train_SpatialWavJEPA.sh config.yaml
```

논문 하이퍼파라미터(`config.yaml`): AdamW β=(0.9,0.98), wd 0.04, peak LR
2e-4, linear-warmup 100k + cosine decay 375k step; EMA τ 0.999→0.99999
(100k step); masking p_ctx 0.065 / p_tgt 0.025 / len 10 / K=8 /
context ≥10%. 유효 배치 ≈128 = `per_device_train_batch_size ×
gradient_accumulation_steps × world_size`.

로그: 모듈별 학습 가능 파라미터 수, 첫 배치 첫 샘플의 **wav 경로**(feed
이전), `logging_steps`마다 step/loss/lr/ema. `save_steps`마다 HF
`checkpoint-{step}/` **및** `spatial_wavjepa_student_step{step}.pt`
(student 전용, 배포 가능) 저장.

## CPU smoke 테스트

```bash
python make_synthetic_manifest.py --out /tmp/swj_smoke
bash run_train_SpatialWavJEPA.sh /tmp/swj_smoke/config.yaml
```

## 추론

```python
from SpatialWavJEPA import SpatialWavJEPA
m = SpatialWavJEPA.from_checkpoint("spatial_wavjepa_student_step4.pt")
emb = m(torch.randn(1, 2, 32159))     # (1, 400, 768) channel-major
```

## 참고 / 주의사항

- `process_audio_seconds=2.01` 유지(`int(16000*2.01)=32159` 샘플 → 채널당
  200 patch → 2N=400). 상위 `WavJEPA.py`의 `# 32_160` 주석은 부정확하나
  내부적으로 일관됨. 2.0초로 바꾸면 199 patch 가 되어 레이아웃이 깨진다.
- 공개된 단일채널 `wavjepa-base` 체크포인트는 호환되지 않는다(이중
  extractor + 400 토큰 2D 위치 임베딩) — 논문대로 from-scratch 학습.
  `from_checkpoint`는 shape 불일치 위치 임베딩을 자동으로 버린다.
- 400 토큰 attention 은 base 대비 약 4배 + K=8 decoder pass — 실제
  학습은 GPU 필요. smoke 설정은 배치/step 을 최소화한다.
- 부모 `_SpanMasker`의 `in_channels>1` 분기는 `ctx`는 channel-major,
  `target`은 time-interleaved 로 펼쳐 본 레이아웃과 불일치한다. 따라서
  `generate_masks`에서 채널당 샘플 후 channel-major 로 직접 tile 하여
  ctx/target/ctx∪target, `[w1;w2]` concat, 2D 위치 임베딩 순서를
  일관되게 맞춘다.
