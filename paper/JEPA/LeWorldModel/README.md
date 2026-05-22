# LeWorldModel (LeWM) — Stable End-to-End JEPA from Pixels (2026)

Self-contained PyTorch 2.8 reimplementation of **LeWorldModel (LeWM)**: the
first Joint-Embedding Predictive Architecture that trains *stably
end-to-end from raw pixels* with only **two loss terms** and **no EMA
teacher / stop-gradient / pretrained encoder**. Representation collapse is
prevented by **SIGReg** (Sketched Isotropic Gaussian Regularization), which
reduces the tunable loss hyperparameters from six to **one**.

- **Paper:** *LeWorldModel: Stable End-to-End Joint-Embedding Predictive
  Architecture from Pixels*, Lucas Maes, Quentin Le Lidec, Damien Scieur,
  Yann LeCun, Randall Balestriero, 2026 — https://arxiv.org/abs/2603.19312
- **Official code (referenced heavily for the architecture):**
  https://github.com/lucas-maes/le-wm  (MIT, ~3.4k★)
- **Regularizer lineage:** SIGReg from **LeJEPA** (Balestriero & LeCun
  2025). World-model lineage: I-JEPA → V-JEPA 2 (`../VJEPA2`) → LeWM.

## 이 논문을 선정한 이유 (2026년 기준)

`paper/`에는 2025년까지의 주요 JEPA(오디오/비디오/언어/3D, 모두 **EMA
teacher + stop-gradient** 기반)가 구현되어 있습니다. 2026년 신규 논문 중
가장 의미 있는 것으로 LeWM을 선정했습니다. 구체적 근거:

1. **최신성 (2026):** 2026년 3월 arXiv 공개(2603.19312)로 TODO의
   "최신(26년도~)" 조건을 충족합니다.
2. **github stars (최상위):** 공식 구현 `lucas-maes/le-wm`이 **~3.4k
   stars**로, 2026년 공개된 JEPA 저장소 중 가장 높습니다
   (cf. Meta `eb_jepa` 634★).
3. **저자/영향력:** Yann LeCun·Randall Balestriero 등 JEPA 원류 저자진의
   후속 연구로, LeJEPA(SIGReg) 계열의 핵심 구현체입니다.
4. **방법론적 진보 (collapse 해결):** 기존 JEPA가 의존하던 **EMA
   teacher·stop-gradient·pretrained encoder**를 모두 제거하고, SIGReg
   하나로 collapse를 막아 **end-to-end** 학습을 가능케 합니다. 손실
   하이퍼파라미터를 6→1로 줄였습니다.
5. **성능:** Push-T/Cube/TwoRooms/Reacher 제어 벤치마크에서 경쟁력 있는
   성능과 함께 foundation-model 기반 대비 **48× 빠른 planning**을 보고.
6. **기존 구현과의 차별성:** 본 repo의 모든 선행 JEPA가 EMA 기반인 반면
   LeWM은 **non-EMA·정규화 기반**으로, JEPA 라인업에 패러다임적으로
   새로운 축을 추가합니다. 또한 action-conditioned **world model**이라
   비디오/제어/플래닝으로 확장됩니다.

## 방법 (구현 기준)

```
z_t        = Enc(o_t)                         # ViT frame encoder
z_hat_{t+1}= Pred(z_<=t, a_<=t)               # causal AR predictor
L_pred     = || z_hat_{t+1} - z_{t+1} ||^2    # 양쪽 모두 gradient (end-to-end)
L_reg      = SIGReg({z})                       # 아래
L          = L_pred + lambda * L_reg           # 단일 하이퍼파라미터 lambda
```

**SIGReg (Sketched Isotropic Gaussian Regularization):** 임베딩을 무작위
단위 방향들에 사영(sketch)한 뒤, 각 1-D 사영 분포가 N(0,1)을 따르는지를
**Epps–Pulley** 적합도 통계량으로 측정해 평균합니다. 본 구현의 닫힌 형태
(가중치 exp(-t²/2)):

```
T = √(2π)/n² · Σ_jk exp(-(y_j-y_k)²/2)
    - 2√π/n · Σ_k exp(-y_k²/4)
    + √π
```

`T ≥ 0`, 경험적 특성함수가 N(0,1)과 일치할 때 0. 미분 가능하며 CPU에서도
가볍습니다. (스모크 테스트에서 Gaussian 표본 0.33 vs collapsed 0.73으로
collapse를 실제로 페널티함을 확인.)

## 구성 파일

| 파일 | 역할 |
|---|---|
| `LeWorldModel.py` | ViT 인코더 + action embedder + causal AR predictor + SIGReg + `compute_loss`(학습) + `encode`/`rollout`(추론) + `from_checkpoint` |
| `train_lewm.py` | `transformers.Trainer` 기반 SSL 사전학습 (episode Dataset/DataLoader/콜백) |
| `config.yaml` | 학습 단일 설정 파일 |
| `run_train_LeWorldModel.sh` | torchrun 런처 (기존 checkpoint 존재 시 시작 거부) |
| `make_synthetic_manifest.py` | CPU 스모크용 합성 episode(움직이는 패치) 생성 |
| `requirements.txt` | 의존성 (torch==2.8.0, CPU; SIGReg/encoder/predictor 모두 pure-torch) |

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt
```

## 데이터셋

episode 단위 `.npz`(`obs` (T,H,W,3) uint8/float, `actions` (T,action_dim)
float)를 매니페스트로 가리킵니다:

```json
{"data": [{"episode_id": "pusht/ep0001.npz"}, ...]}
```

`episode_path = data.data_root / episode_id`. 학습 시 `seq_len` 길이
구간을 샘플링(train 랜덤 / eval 앞부분), obs는 `[0,1]`로 정규화 후
`img_size`로 리사이즈됩니다.

논문에서 사용한 공개 제어 벤치마크(공식 저장소가 HDF5 episode를 제공):

- **Push-T / Cube / TwoRooms / Reacher** — 공식 저장소
  https://github.com/lucas-maes/le-wm 의 `stable-worldmodel[env]`
  데이터 다운로드 스크립트로 받을 수 있습니다.
- 일반 비디오·제어 데이터셋도 위 `.npz` 스키마(프레임+행동)로 변환하면
  그대로 사용 가능합니다. 행동이 없는 순수 비디오는 `action_dim`을 1로
  두고 0 벡터를 넣으면 무행동 world model로 학습됩니다.

## 학습

```bash
bash run_train_LeWorldModel.sh config.yaml
NPROC_PER_NODE=4 bash run_train_LeWorldModel.sh config.yaml   # 멀티-GPU
```

모든 인자는 `config.yaml` 한 파일에 정리되어 있고, 셸 스크립트는
`train.output_dir`에 checkpoint가 있으면 학습을 시작하지 않습니다. HF
checkpoint와 함께 step 번호가 들어간 `.pt`(`lewm_step{N}.pt`)가 저장되어
`LeWorldModel.from_checkpoint(...)`로 바로 로드됩니다.

학습 로그(요청 사항 충족): 모듈별 학습 가능 파라미터 수; 첫 배치 첫
샘플의 **episode 경로**(모델 feed 전); `logging_steps`마다 step /
train_loss / pred_loss / reg_loss / (valid_loss) / lr; `save_steps`마다
step 번호 포함 checkpoint; `init_from` 0개 로드 시 즉시 중단.

## CPU 스모크 테스트

```bash
python make_synthetic_manifest.py --out /tmp/lewm_smoke
bash run_train_LeWorldModel.sh /tmp/lewm_smoke/config.yaml
```

스모크 config는 인코더/예측자를 작게 축소해 CPU에서 수 step 만에 끝납니다
(pred_loss 감소 확인됨).

## 추론 (frozen encode / rollout)

```python
from LeWorldModel import LeWorldModel
m = LeWorldModel.from_checkpoint("lewm_step50000.pt")   # eval + requires_grad=False
z = m.encode(obs)                       # (B, T, latent) 프레임 임베딩
z_future = m.rollout(obs0, actions)     # (B, T, latent) 상상한 미래 latent
```
