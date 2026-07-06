# V-JEPA 2 — Self-Supervised Video JEPA (arXiv:2506.09985)

Self-contained PyTorch 2.8 reimplementation of **V-JEPA 2** — the
deployable video encoder + an attentive-probe head for inference, and a
full JEPA self-supervised pre-training pipeline (student encoder + EMA
teacher + predictor + 3D multi-block masking).

- **Paper:** *V-JEPA 2: Self-Supervised Video Models Enable Understanding,
  Prediction and Planning*, Assran et al., Meta AI, June 2025 —
  https://arxiv.org/abs/2506.09985
- **Official code (referenced heavily for the architecture):**
  https://github.com/facebookresearch/vjepa2  (MIT-licensed)
- **Pretrained checkpoints:**
  - HuggingFace: https://huggingface.co/facebook/vjepa2-vitg-fpc64-384
    (also `vitl`, `vith` variants)
  - torch.hub: `torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')`
- **Lineage:** I-JEPA (arXiv:2301.08243) → V-JEPA (arXiv:2404.08471) →
  V-JEPA 2.

## 이 논문을 선정한 이유

`paper/`에는 이미 오디오 계열 JEPA(WavJEPA, SpatialWavJEPA, BAT)가 구현되어
있어, **비전/비디오 도메인의 대표 JEPA**를 추가해 JEPA 계열을 폭넓게
커버하고자 V-JEPA 2를 선정했습니다. 구체적 근거:

1. **최신성 (2025):** 2025년 6월 공개된 Meta AI의 최신 JEPA 논문으로,
   TODO의 "최신(25년도~)" 조건을 충족합니다.
2. **성능 (SOTA):** 모션 이해 Something-Something v2 **77.3 top-1**,
   행동 예측 Epic-Kitchens-100 **39.7 recall@5 (SOTA)**, LLM 정렬 후
   8B 규모에서 다수의 비디오 QA 벤치마크 SOTA, 그리고 라벨 없는 로봇
   비디오 62시간만으로 실제 Franka 로봇에 zero-shot pick-and-place
   (V-JEPA 2-AC) — 단일 SSL backbone이 understanding/prediction/planning을
   모두 커버합니다.
3. **인용·관심도:** 1M 시간 이상의 인터넷 비디오로 학습된 공개 월드
   모델로, 공개 직후 다수 매체(Meta AI, MarkTechPost 등)에 보도되고
   공식 GitHub(`facebookresearch/vjepa2`)가 수천 stars를 기록하는 등
   2025년 JEPA 논문 중 가장 영향력이 큽니다.
4. **SLLM 적합성:** V-JEPA 2는 LLM과 정렬해 비디오 QA에 활용되는 구조라
   본 repository(Speech/Multimodal LLM)의 방향과 직접 맞닿아 있습니다.
5. **공개 코드 존재:** MIT 라이선스 공식 구현이 있어 아키텍처를 충실히
   재현할 수 있습니다(아래 "참고 구현" 참조).

## 참고 구현 (Reference)

아키텍처는 공식 저장소 `facebookresearch/vjepa2`를 기준으로 재현했습니다:

- `src/models/vision_transformer.py` — `PatchEmbed3D`(Conv3d, tubelet),
  3D RoPE 어텐션, variant별 (depth, embed_dim, heads, mlp_ratio),
  CLS 토큰 없음.
- `src/models/predictor.py` — narrow predictor, mask token, 3D 위치.
- multiblock-3d 마스킹 / EMA teacher / latent L1 목적함수.

## 구성 파일

| 파일 | 역할 |
|---|---|
| `VJEPA2.py` | 추론 전용 인코더(3D ViT + RoPE) + `AttentiveProbe` + `from_checkpoint` |
| `VJEPA2_Trainer.py` | JEPA 학습 래퍼: student + EMA teacher + predictor + 3D 멀티블록 마스커 |
| `train_vjepa2.py` | `transformers.Trainer` 기반 SSL 사전학습 (Dataset/DataLoader/콜백) |
| `eval_vjepa2.py` | frozen 인코더 + attentive probe top-1 평가 |
| `config.yaml` | 학습/평가 단일 설정 파일 |
| `run_train_VJEPA2.sh` | torchrun 런처 (기존 checkpoint 존재 시 시작 거부) |
| `run_eval_VJEPA2.sh` | 평가 런처 |
| `make_synthetic_manifest.py` | CPU 스모크용 합성 비디오/매니페스트 생성 |
| `requirements.txt` | 의존성 (torch==2.8.0 기준, CPU) |

### 아키텍처 변형(variant)

| variant | depth | embed_dim | heads | mlp_ratio |
|---|---|---|---|---|
| `vit_large` | 24 | 1024 | 16 | 4 |
| `vit_huge` | 32 | 1280 | 16 | 4 |
| `vit_giant` | 40 | 1408 | 16 | 48/11 |
| `vit_gigantic` | 48 | 1664 | 16 | 64/13 |

입력: `(B, 3, num_frames, img_size, img_size)`, 픽셀 `[0,1]` → ImageNet
정규화. 토큰 수 `N = (T/tubelet)·(H/patch)·(W/patch)`, CLS 토큰 없음.

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt
```

## 데이터셋

V-JEPA 2는 라벨이 필요 없는 **순수 SSL 비디오 사전학습**입니다. 매니페스트
JSON만 맞추면 임의의 비디오 코퍼스로 학습할 수 있습니다.

매니페스트 형식:

```json
{"data": [{"video_id": "rel/path/clip0001.mp4"}, ...]}
```

`video_path = data.video_root / video_id`. `.npy`(형상 `(T,H,W,3)` uint8
또는 `(3,T,H,W)` float)와 `decord`로 디코딩되는 영상 파일을 모두 지원합니다.

권장 공개 데이터셋(논문에서 사용/평가):

- **Something-Something v2** (모션 이해, 평가):
  https://www.qualcomm.com/developer/software/something-something-v-2-dataset
- **Kinetics-400/700** (대규모 사전학습):
  https://github.com/cvdfoundation/kinetics-dataset
- **Epic-Kitchens-100** (행동 예측, 평가):
  https://epic-kitchens.github.io/2020-100
- **HowTo100M** (대규모 인터넷 비디오 사전학습):
  https://www.di.ens.fr/willow/research/howto100m/

다운로드 후 위 형식의 `train.json` / `valid.json`을 만들고
`config.yaml`의 `data` 섹션을 가리키면 됩니다.

평가(`eval:` 섹션)는 라벨이 필요하며 매니페스트에 `"label": int`를
추가합니다: `{"data":[{"video_id":"...","label":3}, ...]}`.

## 학습

```bash
# (선택) 다른 인코더 크기:  model.variant: vit_giant
bash run_train_VJEPA2.sh config.yaml
# 멀티-GPU:
NPROC_PER_NODE=4 bash run_train_VJEPA2.sh config.yaml
```

`config.yaml` 한 파일에 모든 인자가 정리되어 있으며, 셸 스크립트는
`train.output_dir`에 checkpoint가 이미 있으면 학습을 시작하지 않습니다.
저장 시 HF checkpoint와 함께 step 번호가 포함된 student-only `.pt`
(`vjepa2_student_step{N}.pt`)가 저장되고, 이는
`VJEPA2.from_checkpoint(...)`로 바로 로드됩니다.

학습 로그(요청 사항 충족):
- 모듈별 학습 가능 파라미터 수 (학습 시작 시 1회)
- 첫 배치 첫 샘플의 **비디오 경로** (모델 feed 전)
- `logging_steps`마다 step / train_loss / (valid_loss) / lr / ema_decay
- `save_steps`마다 step 번호가 들어간 checkpoint 저장
- `init_from` 워밍스타트 시 0개 로드면 즉시 중단

## CPU 스모크 테스트

```bash
python make_synthetic_manifest.py --out /tmp/vjepa2_smoke
bash run_train_VJEPA2.sh /tmp/vjepa2_smoke/config.yaml
```

(스모크 config는 인코더를 작게 축소해 CPU에서 수 step 만에 끝납니다.)

## 평가 (attentive probe)

```bash
bash run_eval_VJEPA2.sh config.yaml /path/to/ckpts/vjepa2/vjepa2_student_step2000.pt
```

frozen 인코더 위에 attentive probe만 학습 후 top-1 정확도를 출력합니다
(`from_checkpoint`이 `model.eval()` + `requires_grad=False` 적용).

## 공식 사전학습 가중치 로드

`VJEPA2.from_checkpoint`는 공식 torch.hub blob
(`{"encoder":..., "target_encoder":..., "predictor":...}`, EMA
`target_encoder` 우선)과 HuggingFace `VJEPA2Model` 키 네이밍, 그리고
래퍼 prefix(`module.`/`_orig_mod.`/`backbone.`/`encoder.`)를 자동
처리합니다. 단 한 개의 텐서도 로드되지 않으면(아키텍처 불일치) 즉시
예외를 던집니다.
