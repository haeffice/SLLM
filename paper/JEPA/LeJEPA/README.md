# LeJEPA — Provable & Scalable JEPA, Heuristics-Free (2025)

Self-contained PyTorch 2.8 reimplementation of **LeJEPA**: the paper that
gives JEPAs a *theory*. It proves that the **isotropic Gaussian** is the
optimal distribution for a JEPA's embeddings (it minimises downstream
prediction risk) and reaches it with a single, heuristics-free regulariser —
**SIGReg** (Sketched Isotropic Gaussian Regularization). LeJEPA has **no EMA
teacher, no stop-gradient, no schedulers, no collapse tricks**; the whole
objective has **one** trade-off hyper-parameter.

- **Paper:** *LeJEPA: Provable and Scalable Self-Supervised Learning Without
  the Heuristics*, Randall Balestriero, Yann LeCun, 2025 —
  https://arxiv.org/abs/2511.08544
- **후속 이론 (identifiability, 2026):** *When Does LeJEPA Learn a World
  Model?*, Klindt Lab — https://arxiv.org/abs/2605.26379 (핵심 정리 4개를
  Lean 4로 형식검증). 가우시안 선택을 *식별가능성*으로 정당화 — 아래
  "이론적 근거" 절 참조.
- **Official code (referenced heavily for the design):**
  https://github.com/rbalestr-lab/lejepa  (~1.1k★)
- **Lineage:** SIGReg here is the *source* regulariser later reused by the
  world-model `../LeWorldModel`. Distinct collapse-prevention axis from
  VICReg-based `../EB-JEPA` (variance+covariance moments) and from the
  EMA-teacher JEPAs in this repo (`../VJEPA2`, `../WavJEPA`, …).

## 이 논문을 선정한 이유 (2025년 최신)

본 repo의 JEPA 라인업은 EMA-teacher 계열(V-JEPA2/WavJEPA/Point-JEPA 등)과
non-EMA 정규화 계열(EB-JEPA=VICReg, LeWorldModel=SIGReg 월드모델)로 이미
폭넓게 채워져 있습니다. 그런데 **그 SIGReg를 처음 제안한 원전(原典)
LeJEPA 자체**는 `../LeWorldModel`·`../EB-JEPA`의 README에서 *인용만* 될 뿐
독립 구현이 없었습니다. 이 핵심 공백을 메우기 위해 LeJEPA를 선정했습니다.

1. **최신성 (2025):** 2025년 11월 arXiv 공개(2511.08544)로 TODO의
   "최신(25년도~)" 조건을 충족합니다.
2. **인용/주목도 (최상위):** Yann LeCun·Randall Balestriero의 후속작으로
   공개 직후 SSL 커뮤니티에서 가장 많이 회자된 2025 논문 중 하나이며,
   공식 저장소 `rbalestr-lab/lejepa`가 **~1.1k stars**입니다.
3. **성능:** ImageNet-1K linear-probe에서 **ViT-H/14 79%**, **10+개
   데이터셋·60+개 아키텍처(ResNet/ViT/ConvNet)·최대 1.8B ViT-g**까지
   하이퍼파라미터를 거의 바꾸지 않고 안정적으로 학습됨을 보고합니다.
4. **방법론적 의의 (이론 + 단순성):** "임베딩 분포는 **등방성 가우시안**일
   때 downstream 위험이 최소"임을 증명하고, 이를 **SIGReg** 단일 항으로
   달성합니다. EMA teacher·stop-gradient·스케줄러·predictor collapse 트릭을
   **모두 제거**하고 trade-off 하이퍼파라미터를 **1개**로 줄였습니다.
5. **기존 구현과의 차별성:** 본 repo가 보유한 `LeWorldModel`의 SIGReg는
   *월드모델/제어*에 적용된 파생형이고, `EB-JEPA`는 *모멘트 기반(VICReg)*
   정규화입니다. LeJEPA는 **이미지 SSL의 정통 형태에서 분포-정합(특성함수)
   기반 SIGReg**를 쓰는 원전으로, 라인업의 이론적 기반을 완성합니다.

## 방법 (구현 기준)

```
z_v    = Enc(view_v)                            # ViT 임베딩 (뷰별)
p_v    = Pred(z_v)                              # JEPA predictor
L_pred = mean_{s, g != s} || Pred(z_s) - z_g ||^2   # 모든 뷰 -> global 뷰 임베딩 예측
L_reg  = SIGReg({ z_v })                         # 임베딩 -> 등방성 가우시안
L      = L_pred + lambda * L_reg                 # 단일 하이퍼파라미터 lambda
```

- **뷰 구성 (DINO식 multi-crop):** global 2장(224, scale 0.3–1.0) + local
  6장(98, scale 0.05–0.3). global 임베딩이 예측 타깃입니다.
- **stop-gradient 없음:** 타깃 임베딩에도 gradient가 흐릅니다(에너지/정규화
  기반). collapse는 SIGReg가 막습니다.

**SIGReg (Sketched Isotropic Gaussian Regularization):** 임베딩을
`num_slices`개의 무작위 단위 방향에 사영(sketch)한 뒤, 각 1-D 사영 분포가
N(0,1)을 따르는지를 **Epps–Pulley** 경험적 특성함수(empirical characteristic
function) 적합도 통계량으로 측정해 평균합니다. `num_points`개의 Gauss–Hermite
구적점으로 적분을 근사합니다:

```
phi_n(t) = (1/n) Σ_k exp(i t y_k)                  # 경험적 특성함수
T_slice  = ∫ |phi_n(t) - e^{-t²/2}|² e^{-t²/2} dt
         ≈ Σ_m w_m [ (Re phi_n(t_m) - e^{-t_m²/2})² + Im phi_n(t_m)² ]
SIGReg   = mean_slice T_slice
```

`T ≥ 0`, 모든 1-D 사영이 N(0,1)과 일치(=등방성 가우시안)할 때 0. 미분
가능하며 표본 수·차원에 선형이라 CPU에서도 가볍습니다. (스모크 검증:
가우시안 표본 **0.0024** vs collapsed **0.41**; collapsed 시작점을 SIGReg로
최소화하면 200 step 만에 **1.92→0.002**, mean→0.015·std→1.012로 N(0,1)에
수렴함을 확인.) 기본값은 논문 따라 `num_slices=1024`, `num_points=17`.

> 인코더는 논문에서 아키텍처-무관(ResNet/ViT/ConvNet)입니다. 본 구현은
> 해상도가 다른 multi-crop을 한 네트워크로 처리하고 linear-probe가 마지막
> 레이어들의 CLS 토큰을 읽을 수 있도록 **compact ViT**를 사용합니다(`../EB-JEPA`
> 의 ResNet과 상보적).

## 이론적 근거: 왜 *반드시* 가우시안인가 (식별가능성)

후속 이론 논문 **"When Does LeJEPA Learn a World Model?"** (Klindt Lab, 2026,
arXiv:2605.26379; 핵심 정리 4개를 **Lean 4로 형식검증**)이 LeJEPA의 가우시안
선택에 원 논문보다 강한 근거를 줍니다. 원 LeJEPA가 "등방성 가우시안이
downstream 위험을 최소화한다"는 *통계적* 주장이었다면, 이 논문은 인코더가
**세계의 잠재변수를 회전만 빼고 선형 복원(linear identifiability)하며, 그것이
성립하는 잠재분포는 오직 가우시안뿐**임을 증명합니다 — 즉 가우시안은 인코더가
진짜 world model이 되기 위한 *유일한 필요조건*입니다.

**세계 가정:** 잠재 독립 + 정상성(`p(z)=p(z')`) + 가법잡음 전이
`z'_i = m_i(z_i) + η_i`. 핵심 사례는 가우시안 잠재 `z ~ N(0,I)`의 OU 전이
`z' = ρz + √(1-ρ²)·η` 로, 정렬용 positive pair(같은 대상의 두 뷰)를 만든다.
정렬 손실은 pair 상관 최대화와 동치: `L(h) = 2n - 2·Σ_i E[h_i(z')·h_i(z)]`.

- **정리 5.1 (선형 식별가능성):** `h(z) ~ N(0,I)`인 임의의 측정가능 `h`에 대해
  `L(h) ≥ 2(1-ρ)n`, **등호 ⟺ `h(z) = Qz`, `Q ∈ O(n)`**. 정렬 + 가우시안
  (whitening) 제약의 최적해는 진짜 잠재를 회전/반사만 빼고 복원한 것뿐이다.
- **정리 5.2 (가우시안 유일성):** 가우시안 측도에서 Hermite 분해 시 차수 `d`
  성분의 시간 상관 기여가 `ρ^d` 라, `ρ<1` 이면 `d≥2` 비선형 성분은 선형항보다
  항상 작다 → 최적해는 직교 선형(`w₁=1`). 역으로 첫 고유함수가 affine이려면
  `log p(z) ∝ -(z-μ)²`, 즉 **가우시안만** 선형 식별을 보장한다(비가우시안은
  단조변환까지로 약화). 실험상 복원 R²이 `α=2`(가우시안)에서 뾰족하게 정점.
- **정리 5.3 (근사 식별):** 정렬 갭 `δ`·whitening 오차 `ε`, `D = δ/(2ρ(1-ρ))`
  일 때 `∃Q: E‖h(z) - Qz‖² ≤ D + (ε+D)²` — 오차가 우아하게 열화하며 **정렬이
  병목, whitening은 거의 공짜**(SIGReg의 유계 gradient·둔감한 `λ`와 부합).
- **정리 5.4 (계획):** `h(z)=Qz` + `O(n)`-불변 비용이면 학습 잠재공간의 최적
  가치·정책이 진짜 세계의 것과 동일 — 절대좌표 복원 없이도 제어 가능(=world
  model). DMC Reacher planning 품질이 linear identifiability와 단조 상관.

정리하면 본 repo 라인업에서 가우시안의 위상은 세 층위로 정당화된다:
**collapse 방지**(최대엔트로피·등방성) → **downstream 최적**(원 LeJEPA, risk
최소화) → **world model**(이 논문, 선형 식별의 유일 분포). 마지막 층위는
`../LeWorldModel`(SIGReg를 월드모델/제어에 재사용)의 직접적 이론 근거다.

## 구성 파일

| 파일 | 역할 |
|---|---|
| `LeJEPA.py` | ViT 인코더 + JEPA predictor + `SIGReg`(Epps–Pulley+slicing) + `compute_loss`(학습) + `encode`/`embed`(추론) + `from_checkpoint` |
| `train_lejepa.py` | `transformers.Trainer` 기반 SSL 사전학습; multi-crop Dataset(pure-torch augmentation) + 콜백 |
| `config.yaml` | 학습 단일 설정 파일 |
| `run_train_LeJEPA.sh` | torchrun 런처 (기존 checkpoint 존재 시 시작 거부) |
| `make_synthetic_manifest.py` | CPU 스모크용 합성 이미지/매니페스트 생성 |
| `requirements.txt` | 의존성 (torch==2.8.0, CPU; ViT/SIGReg/multi-crop 모두 pure-torch+numpy) |

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

`image_path = data.image_root / image_id`. 학습 시 각 이미지에서 global/local
multi-crop 뷰를 생성합니다(라벨 불필요, 순수 SSL).

논문이 사용하는 대표 데이터셋:

- **ImageNet-1K** (사전학습 + linear-probe 평가):
  https://www.image-net.org/ (HF: `imagenet-1k`). 각 이미지를 `(H,W,3)`
  `.npy`로 덤프(또는 디코딩 후 저장)하고 위 스키마의 `train.json`을 만듭니다.
- 더 작게 검증하려면 **CIFAR-10/100**, **TinyImageNet** 등도 동일 스키마로
  사용 가능합니다(`global_size`/`local_size`를 해상도에 맞게 줄이세요).

다운로드 후 `config.yaml`의 `data` 섹션이 매니페스트/`image_root`를 가리키게
설정합니다. (공식 저장소 `rbalestr-lab/lejepa`의 데이터/증강 파이프라인 참고.)

## 학습

```bash
bash run_train_LeJEPA.sh config.yaml
NPROC_PER_NODE=4 bash run_train_LeJEPA.sh config.yaml   # 멀티-GPU
```

모든 인자는 `config.yaml` 한 파일에 정리되어 있고, 셸 스크립트는
`train.output_dir`에 checkpoint가 있으면 학습을 시작하지 않습니다. HF
checkpoint와 함께 step 번호가 들어간 `.pt`(`lejepa_step{N}.pt`)가 저장되어
`LeJEPA.from_checkpoint(...)`로 바로 로드됩니다.

학습 로그(요청 사항 충족): 모듈별 학습 가능 파라미터 수; 첫 배치 첫
샘플의 **이미지 경로**(모델 feed 전); `logging_steps`마다 step /
train_loss / pred_loss / reg_loss / (valid_loss) / lr; `save_steps`마다
step 번호 포함 checkpoint; `init_from` 0개 로드 시 즉시 중단.

> 참고: SIGReg는 큰 배치에서 사영 분포 추정이 정확해집니다(논문은 큰 배치
> + bf16). `sigreg_coeff`(=lambda)가 유일한 trade-off 하이퍼파라미터로,
> 키우면 등방성 가우시안 제약이 강해집니다. GPU에서는 `optim.bf16: true`
> 권장.

## CPU 스모크 테스트

```bash
python make_synthetic_manifest.py --out /tmp/lejepa_smoke
bash run_train_LeJEPA.sh /tmp/lejepa_smoke/config.yaml
```

스모크 config는 ViT/뷰/slice를 작게 축소해 CPU에서 수 step 만에 끝납니다
(loss 1.83→1.61, pred_loss 0.98→0.79 감소 확인; checkpoint 로드 후 eval +
requires_grad=False, probe feature shape 확인됨).

## 추론 (frozen feature → linear probe)

```python
from LeJEPA import LeJEPA
m = LeJEPA.from_checkpoint("lejepa_step100000.pt")   # eval + requires_grad=False
feats = m.encode(images)        # (B, k*D) 마지막 k개 블록 CLS concat — linear/kNN probe 입력
z     = m.embed(images)         # (B, D) 마지막 레이어 CLS 임베딩 (SSL 타깃)
```
