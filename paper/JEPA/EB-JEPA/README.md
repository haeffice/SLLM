# EB-JEPA — Energy-Based JEPA (2026, arXiv:2602.03604)

Self-contained PyTorch 2.8 reimplementation of the **image** example of
**EB-JEPA** (Meta FAIR's *energy-based* Joint-Embedding Predictive
Architecture). EB-JEPA prevents representation collapse with explicit
**VICReg-style variance + covariance regularization** (the "energy")
instead of an EMA teacher / stop-gradient — yielding a simple objective
with two interpretable coefficients (`std_coeff`, `cov_coeff`).

- **Paper:** *A Lightweight Library for Energy-Based Joint-Embedding
  Predictive Architectures*, Terver, Balestriero, …, LeCun, Bar (Meta AI
  / FAIR), 2026 — https://arxiv.org/abs/2602.03604
- **Official code (referenced heavily for the architecture):**
  https://github.com/facebookresearch/eb_jepa  (Apache-2.0, ~634★)
- **Regularizer lineage:** VICReg (Bardes et al. 2022, arXiv:2105.04906).
  Energy-based / non-EMA collapse-prevention shared with LeWM
  (`../LeWorldModel`).

## 이 논문을 선정한 이유 (2026년 기준)

이전 iteration에서 2026 JEPA 중 LeWorldModel(SIGReg 정규화)을 구현했고,
그 다음으로 의미 있는 2026 논문으로 EB-JEPA를 선정했습니다. 근거:

1. **최신성 (2026):** 2026년 arXiv 공개(2602.03604)로 "최신(26년도~)"
   조건을 충족합니다.
2. **공개 코드 + 신뢰도:** **Meta AI / FAIR 공식** 저장소
   `facebookresearch/eb_jepa`(Apache-2.0)가 공개되어 있고, JEPA 원류
   기관의 레퍼런스 라이브러리로서 image/video/action-conditioned video +
   planning을 모두 다룹니다. github stars(~634)도 2026 JEPA 저장소 중
   상위권입니다.
3. **방법론적 의의:** EMA/stop-grad 없이 **에너지(분산+공분산 정규화)**로
   collapse를 막는 패러다임을 명확한 두 계수(`std_coeff`, `cov_coeff`)로
   제시합니다. LeWM(SIGReg, 분포-정규성 기반)과 **다른 축의 non-EMA 정규화
   (VICReg, 모멘트 기반)**라 본 repo의 JEPA 라인업을 상호 보완합니다.
4. **재현 용이성·검증성:** image 예제(CIFAR-10)가 가장 단순해 코드로
   충실히 재현하고 CPU에서 스모크 검증하기에 적합합니다.

> 본 구현은 공식 저장소의 세 예제 중 **image JEPA(VICReg-proj)** 를
> 자기완결적으로 재현합니다. video / action-conditioned 예제는 `../VJEPA2`
> 및 `../LeWorldModel`이 각각 인접 영역을 이미 커버합니다.

## 방법 (구현 기준)

```
z  = Proj(Enc(x )) ,  z' = Proj(Enc(x'))     # 두 augmented view
p  = Pred(z)         (대칭으로 p' = Pred(z'))
L_inv = MSE(p, z')                            # prediction / invariance
L_var = std_coeff * [var_hinge(z) + var_hinge(z')]
L_cov = cov_coeff * [cov_pen(z)  + cov_pen(z')]
L     = L_inv + L_var + L_cov
```

- `var_hinge(Z) = mean_j relu(γ - sqrt(Var(Z_j)+eps))`, γ=1 — 각 차원의
  표준편차를 1 이상으로 유지(collapse 방지).
- `cov_pen(Z) = Σ_{i≠j} Cov(Z)_{ij}² / D` — 차원 간 상관 제거.
- 두 분기 모두 gradient 전파(에너지 기반, **no EMA / no stop-gradient**).

기본 계수는 image config 기준 `std_coeff=1.0`, `cov_coeff=80.0`.

## 구성 파일

| 파일 | 역할 |
|---|---|
| `EBJEPA.py` | ResNet 인코더 + 확장 projector + predictor + VICReg 에너지 손실 + `compute_loss`(학습) + `encode`(추론) + `from_checkpoint` |
| `train_ebjepa.py` | `transformers.Trainer` 기반 SSL 사전학습; two-view 이미지 Dataset(pure-torch augmentation) + 콜백 |
| `config.yaml` | 학습 단일 설정 파일 |
| `run_train_EBJEPA.sh` | torchrun 런처 (기존 checkpoint 존재 시 시작 거부) |
| `make_synthetic_manifest.py` | CPU 스모크용 합성 이미지/매니페스트 생성 |
| `requirements.txt` | 의존성 (torch==2.8.0, CPU; torchvision 불필요) |

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt
```

증강(random-resized-crop / flip / brightness·contrast jitter / grayscale)은
pure-torch로 구현되어 torchvision이 필요 없습니다.

## 데이터셋

이미지를 `.npy`((H,W,3) uint8/float)로 저장하고 매니페스트로 가리킵니다:

```json
{"data": [{"image_id": "train/img0001.npy"}, ...]}
```

`image_path = data.image_root / image_id`. 학습 시 각 이미지에서 두 개의
독립 증강 view를 생성합니다(라벨 불필요, 순수 SSL).

논문 image 예제가 사용하는 데이터셋:

- **CIFAR-10** (사전학습 + linear-probe 평가):
  https://www.cs.toronto.edu/~kriz/cifar.html
  (HF: `uoft-cs/cifar10`). 다운로드 후 각 이미지를 `(32,32,3)` `.npy`로
  덤프하거나, 배치 파일을 풀어 per-image `.npy`로 저장하면 됩니다.

다운로드 후 위 스키마의 `train.json`/`valid.json`을 만들고 `config.yaml`의
`data` 섹션을 가리키면 됩니다. (공식 저장소
`facebookresearch/eb_jepa`의 `examples/image_jepa` 데이터 스크립트 참고.)

## 학습

```bash
bash run_train_EBJEPA.sh config.yaml
NPROC_PER_NODE=4 bash run_train_EBJEPA.sh config.yaml   # 멀티-GPU
```

모든 인자는 `config.yaml` 한 파일에 정리되어 있고, 셸 스크립트는
`train.output_dir`에 checkpoint가 있으면 학습을 시작하지 않습니다. HF
checkpoint와 함께 step 번호가 들어간 `.pt`(`ebjepa_step{N}.pt`)가 저장되어
`EBJEPA.from_checkpoint(...)`로 바로 로드됩니다.

학습 로그(요청 사항 충족): 모듈별 학습 가능 파라미터 수; 첫 배치 첫
샘플의 **이미지 경로**(모델 feed 전); `logging_steps`마다 step /
train_loss / inv_loss / var_loss / cov_loss / (valid_loss) / lr;
`save_steps`마다 step 번호 포함 checkpoint; `init_from` 0개 로드 시 즉시
중단.

> 참고: `cov_coeff`가 커서(기본 80) 배치가 임베딩 차원보다 작으면 공분산
> 추정이 rank-deficient가 되어 초기 손실/grad가 큽니다. 실제 학습은 큰
> 배치(예: 256)와 큰 projector(2048) 기준입니다.

## CPU 스모크 테스트

```bash
python make_synthetic_manifest.py --out /tmp/ebjepa_smoke
bash run_train_EBJEPA.sh /tmp/ebjepa_smoke/config.yaml
```

스모크 config는 ResNet/projector를 작게 축소해 CPU에서 수 step 만에
끝납니다(cov_loss·총손실 감소 확인됨).

## 추론 (frozen feature → linear probe)

```python
from EBJEPA import EBJEPA
m = EBJEPA.from_checkpoint("ebjepa_step50000.pt")   # eval + requires_grad=False
feats = m.encode(images)        # (B, enc_dim) 백본 표현 — linear/kNN probe 입력
```
