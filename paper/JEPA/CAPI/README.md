# CAPI — Cluster & Predict Latent Patches (2025, Meta FAIR)

Self-contained PyTorch 2.8 reimplementation of **CAPI**: a *latent* masked
image model that is the JEPA-family synthesis of **I-JEPA** (predict in latent
space, not pixels) and **iBOT/DINO** (online-clustered targets). A student ViT
encodes the **visible** patches; a cross-attention predictor fills in the
**masked** patches; the target is the **cluster assignment** that an EMA
teacher gives those masked patches, balanced online with **Sinkhorn-Knopp**.
No pixel decoder, no contrastive pairs, no augmentation-view collapse tricks.

- **Paper:** *Cluster and Predict Latent Patches for Improved Masked Image
  Modeling*, Timothée Darcet, Federico Baldassarre, Maxime Oquab, Julien
  Mairal, Piotr Bojanowski (Meta FAIR), 2025 —
  https://arxiv.org/abs/2502.08769
- **Official code (referenced heavily for the design):**
  https://github.com/facebookresearch/capi  (Apache-2.0, ~134★)
- **Lineage:** I-JEPA (latent prediction) + iBOT/DINO (online clustering,
  Sinkhorn) → CAPI. Complementary to this repo's augmentation-view image
  JEPAs (`../EB-JEPA` VICReg, `../LeJEPA` SIGReg) and the EMA-teacher
  video/audio JEPAs (`../VJEPA2`, `../WavJEPA`).

## 이 논문을 선정한 이유 (2025년 최신)

본 repo의 이미지 JEPA는 **증강 뷰(view) 기반**(EB-JEPA=VICReg, LeJEPA=SIGReg)
이거나 **EMA teacher + feature regression**(V-JEPA2 등) 계열입니다. 그런데
**마스킹 기반 latent 예측(I-JEPA)** 과 **온라인 클러스터링 타깃(iBOT/DINO,
Sinkhorn)** 을 결합한 축은 비어 있었습니다. CAPI는 정확히 그 공백을 메우는
2025년 대표작입니다.

1. **최신성 (2025):** 2025년 2월 arXiv 공개(2502.08769)로 TODO의
   "최신(25년도~)" 조건을 충족합니다.
2. **신뢰도/공개코드:** **Meta FAIR**(DINOv2 팀)의 공식 저장소
   `facebookresearch/capi`(Apache-2.0)가 공개되어 있어 구현을 충실히
   참고했습니다.
3. **성능:** latent MIM에서 ImageNet-1K **attentive/linear probe** 기준
   기존 MIM(MAE/I-JEPA/data2vec)을 능가하고 DINOv2급에 근접하는 표현을
   보고하며, ADE20k segmentation 등 dense 태스크에서도 강합니다(ViT-L/14).
4. **방법론적 의의 (라인업 보완):** 본 repo의 다른 어떤 구현도 갖지 않은
   **(a) 마스킹된 패치의 latent 예측, (b) 학습형 prototype + Sinkhorn-Knopp
   균형 클러스터링 타깃, (c) self-attention 없는 cross-attention predictor,
   (d) EMA teacher(μ=1−lr)** 를 모두 포함합니다. collapse 방지를 정규화가
   아니라 **균형 클러스터링 문제**로 환원한 점이 핵심 차별성입니다.
5. **재현 용이성:** 픽셀 디코더·대조쌍이 없어 손실이 단순한 cross-entropy
   하나이며, CPU에서 작은 ViT/prototype으로 스모크 검증이 가능합니다.

## 방법 (구현 기준)

```
h_vis    = Enc(visible patches + registers)             # student (마스킹 입력)
z_pred   = Pred(mask-queries  ⨯-attend  h_vis)          # cross-attn predictor
t_full   = EMA-Enc(all patches)                         # teacher (전체 입력)
target_i = SinkhornKnopp( prototypes · normalize(t_full[i]) )   # 균형 soft 타깃
a_i      = softmax( prototypes · normalize(z_pred[i]) / τ )     # 예측 분포
L        = − mean_i  Σ_k  target_i(k) · log a_i(k)      # cross-entropy
```

- **마스킹:** inverse-block masking(연속 블록을 visible로 유지 + 토로이드
  roll), 기본 mask ratio 0.65. visible/masked 패치 수를 배치 내 고정해
  batch 처리합니다.
- **온라인 클러스터링:** 학습형 prototype `C∈R^{p×d}`(기본 p=16384)에 대해
  teacher 패치를 L2 정규화 후 코사인 로짓 → **Sinkhorn-Knopp** 으로 배치
  내 균형 soft 할당(타깃, stop-grad). 클러스터링 파라미터는 backbone의
  **0.5× lr** 로 학습합니다.
- **EMA teacher:** student 인코더의 EMA 복제본(grad 없음). momentum
  **μ=1−lr**(config에서 `ema_momentum`을 비우면 매 step lr을 따라 자동 설정,
  값을 지정하면 고정).
- **collapse 방지:** Sinkhorn이 클러스터 사용량을 균등하게 강제 → 타깃
  엔트로피가 `log(p)` 근처로 유지(스모크에서 `target_entropy=log(64)=4.159`
  로 안정 확인).

> 인코더는 논문에서 ViT-L/14입니다. 본 구현은 동일 구조의 compact ViT(기본
> ViT-S급)로, register 토큰·cross-attention predictor·Sinkhorn까지 포함해
> 자기완결적으로 재현합니다. (논문의 *positional-collapse 방지용 modified
> Sinkhorn* 은 표준 batch Sinkhorn으로 단순화했고, 차이는 README에 명시.)

## 구성 파일

| 파일 | 역할 |
|---|---|
| `CAPI.py` | ViT 인코더(visible/full forward) + cross-attn predictor + prototypes + `sinkhorn_knopp` + EMA teacher + `compute_loss`(학습) + `encode`/`encode_patches`(추론) + `from_checkpoint` |
| `train_capi.py` | `transformers.Trainer` 기반 SSL 사전학습; 마스킹 Dataset + EMA 콜백 + prototype 0.5× lr 그룹 |
| `config.yaml` | 학습 단일 설정 파일 |
| `run_train_CAPI.sh` | torchrun 런처 (기존 checkpoint 존재 시 시작 거부) |
| `make_synthetic_manifest.py` | CPU 스모크용 합성 이미지/매니페스트 생성 |
| `requirements.txt` | 의존성 (torch==2.8.0, CPU; ViT/Sinkhorn/마스킹 모두 pure-torch+numpy) |

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt
```

증강(random-resized-crop / flip)과 inverse-block 마스킹은 pure-torch로
구현되어 torchvision이 필요 없습니다.

## 데이터셋

이미지를 `.npy`((H,W,3) uint8/float)로 저장하고 매니페스트로 가리킵니다:

```json
{"data": [{"image_id": "train/img0001.npy"}, ...]}
```

`image_path = data.image_root / image_id`. 학습 시 각 이미지에서 1개의
random-resized-crop view를 만들고 패치를 visible/masked로 나눕니다(라벨
불필요, 순수 SSL).

논문이 사용하는 대표 데이터셋:

- **ImageNet-1K** (사전학습 + linear/attentive probe 평가):
  https://www.image-net.org/ (HF: `imagenet-1k`). 이미지를 `(H,W,3)`
  `.npy`로 덤프하고 위 스키마의 `train.json`을 만듭니다.
- **Places205 / LVD-142M** 등 대규모 비라벨 이미지로도 동일 스키마 사용
  가능합니다. 더 작게 검증하려면 **CIFAR/TinyImageNet** 을
  `img_size`/`patch_size`만 줄여 사용하세요.

다운로드 후 `config.yaml`의 `data` 섹션이 매니페스트/`image_root`를 가리키게
설정합니다. (공식 저장소 `facebookresearch/capi`의 데이터/마스킹 파이프라인
및 `default_pretrain_config.yaml` 참고.)

## 학습

```bash
bash run_train_CAPI.sh config.yaml
NPROC_PER_NODE=4 bash run_train_CAPI.sh config.yaml   # 멀티-GPU
```

모든 인자는 `config.yaml` 한 파일에 정리되어 있고, 셸 스크립트는
`train.output_dir`에 checkpoint가 있으면 학습을 시작하지 않습니다. HF
checkpoint와 함께 step 번호가 들어간 `.pt`(`capi_step{N}.pt`)가 저장되어
`CAPI.from_checkpoint(...)`로 바로 로드됩니다.

학습 로그(요청 사항 충족): 모듈별(encoder/predictor/prototypes/teacher)
학습 가능 파라미터 수; 첫 배치 첫 샘플의 **이미지 경로 + visible/masked
패치 수**(모델 feed 전); `logging_steps`마다 step / train_loss / ce_loss /
target_entropy / (valid_loss) / lr; `save_steps`마다 step 번호 포함
checkpoint; `init_from` 0개 로드 시 즉시 중단.

> 참고: CAPI는 **매우 큰 배치(논문 16384)** 와 큰 prototype 수에서
> Sinkhorn 균형이 정확해집니다. CPU/소규모에서는 배치·prototype을 줄여
> 검증하세요. `target_entropy`가 `log(num_prototypes)` 근처를 유지하면
> collapse가 없다는 신호입니다. GPU에서는 `optim.bf16: true` 권장.

## CPU 스모크 테스트

```bash
python make_synthetic_manifest.py --out /tmp/capi_smoke
bash run_train_CAPI.sh /tmp/capi_smoke/config.yaml
```

스모크 config는 ViT/predictor/prototype/패치 수를 작게 축소해 CPU에서 수
step 만에 끝납니다(loss 4.95→4.62 감소, target_entropy=log(64)=4.159 안정,
checkpoint 로드 후 eval + requires_grad=False 및 probe shape 확인됨).

## 추론 (frozen feature → linear/attentive probe)

```python
from CAPI import CAPI
m = CAPI.from_checkpoint("capi_step100000.pt")   # eval + requires_grad=False
feat   = m.encode(images)          # (B, D)    mean-pooled patch features — probe 입력
tokens = m.encode_patches(images)  # (B, N, D) dense 패치 표현 — segmentation/dense 태스크
```
