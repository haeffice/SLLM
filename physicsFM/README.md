# physicsFM — 물리 Foundation Model (경로 A: PhysicsNeMo 스택)

"초기 mesh 상태 + action → autoregressive 상태열 생성 → 정지(eos)" 를 목표로 하는
물리 서로게이트. Phase 1 = MeshGraphNet 베이스라인 + 데이터 파이프라인 (이 저장소),
Phase 2 = UPT 식 잠재 롤아웃 + metriplectic(GENERIC) 제약. 설계 배경/외부 데이터셋은
**[docs/design.md](docs/design.md)** 참고.

데이터 소스는 `physics/be/models/free_fall/trajectory.py` (닫힌형 절차 생성기 —
"학습된 생성 모델 자리의 mock"). 이 mock 을 학습 모델로 대체하는 부트스트랩이
본 프로젝트의 정체성이다. 생성기는 importlib 로 원본을 로드한다(미러 금지).

## 셋업

```bash
cd physicsFM
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # True
```

## 파이프라인

```bash
# 1) 데이터 생성 (plate41/can48x24/smartphone ×150 + hubble 50 평가 전용 → ~1.1GB)
.venv/bin/python generate_rollouts.py --out data/rollouts.h5
.venv/bin/python generate_rollouts.py --inspect data/rollouts.h5   # 불변식 검사

# 2) 학습 (RTX 3050 6GB 기준 hidden 128 / 15층 / batch 2 / bf16)
.venv/bin/python train.py --run mgn_h128_l15

# 3) 평가 — N-step autoregressive 롤아웃 + eos
.venv/bin/python rollout.py --ckpt runs/mgn_h128_l15/latest.pt --split test --save-frames

# 테스트
for t in tests/test_*.py; do .venv/bin/python "$t"; done
```

모든 CLI 는 `--set key.sub=value` 로 config.yaml 을 오버라이드한다.

## 물리 시간 복원 (중요)

생성기의 타임라인은 정규화 t∈[0,1] 이지만 **비율적으로 물리적**(낙하 t_c1, 바운스
지속 2·t_c1·eᵏ)이라 균일 사상 하나로 강체 운동이 정확히 뉴턴역학이 된다:
`T_phys = √(2h/g)/t_c1`, 고정 dt=12ms, **프레임 수가 (h,e)에 따라 가변**.
g = 9.81 mesh-units/s² (실스케일 아님 — hubble 실스케일 6.9 m/unit 은 주석 저장만).
`--inspect` 가 낙하 포물선 항등식 z = h − ½gτ² 을 모든 rollout 에서 검사한다.
잔여 아티팩트: dent rise/ring 시간상수는 정규화 단위라 물리 시간에선 낙하높이 의존.

## 구조

| 파일 | 역할 |
|---|---|
| `meshes.py` | 메쉬 레지스트리 (OBJ 파서, plate/can 절차 생성, 면적 lumped mass, 부품/fragility) |
| `generate_rollouts.py` | 배치 생성 → HDF5 (물리 dt, z_rigid, ke/pe, 접촉 메타, hold-extension) |
| `graph.py` | 피처 계약 — 노드 8(속도+floor_prox+fragility), 엣지 8(Δu_rest+Δx) |
| `dataset.py` | 1-step Dataset (노이즈 주입+해석적 통계 보정, eos 라벨 로더 파생) |
| `models/mgn.py` | MeshGraphNetLite — DGL/PyG 무의존 encode-process-decode |
| `models/eos_head.py` | mean+max 풀링 → settled 로짓 |
| `models/metriplectic.py` | GFINN 식 GENERIC 헤드 (dE/dt=0, dS/dt≥0 기계 정밀도 보장) — Phase 2 용 |
| `train.py` / `rollout.py` | 1-step 학습 / N-step autoregressive 평가(RMSE@k, eos 오차) |

HDF5 스키마: `/meshes/<id>` (공유 메쉬+질량+부품), `/rollouts/r%05d`
(positions f32 + 물리 원료), `/splits`. **라벨(eos)·피처·통계는 전부 로더 파생** —
ε/노이즈/피처 변경에 데이터 재생성 불요. 통계는 `data/stats_<hash>.json` 캐시.

## 6GB VRAM 참고

기본값(hidden 128/15층/batch 2/bf16)으로 학습 시 ~1.1GB. OOM 사다리:
`train.batch_size=1` → `model.grad_checkpoint=true` → `model.num_layers=10` →
`model.hidden_dim=96`. physicsnemo 백본은 `model.backbone=physicsnemo_mgn`
(설치 필요: `pip install nvidia-physicsnemo torch_geometric`; 실패 시 자동 폴백).

## 베이스라인 (v0 — mgn_h128_l15, 15k steps, ~28분/RTX 3050)

1-step VAL mse 0.199 (정규화 단위), eos BCE 0.236. test split autoregressive 롤아웃:

| mesh | n | rmse@1 | rmse@10 | full | final | \|eos_err\| |
|---|---|---|---|---|---|---|
| plate41 | 15 | 2.3e-03 | 8.4e-02 | 9.1e-01 | 1.1e+00 | 69.8 |
| can48x24 | 15 | 2.4e-03 | 8.6e-02 | 9.5e-01 | 1.2e+00 | 79.5 |
| smartphone | 15 | 2.2e-03 | 8.4e-02 | 8.1e-01 | 9.9e-01 | 76.1 |
| **hubble (학습 미포함)** | 50 | 2.2e-03 | 8.5e-02 | 7.7e-01 | 9.5e-01 | 65.7 |

관찰:
- **rmse@1 ≈ 2.2e-3 ≈ 노이즈 σ(3e-3) 수준** — 1-step 정확도는 목표치 도달.
- **hubble 일반화 성공** — 학습에 없던 2.2× 큰 메쉬에서 학습 메쉬와 동등한 오차.
- 장기 롤아웃 드리프트(full ~0.8, bbox 대각 2.0 대비)와 eos 조기 발화는 v0 한계 —
  1-step 학습 15k steps 의 전형적 거동(DeepMind MGN 은 10M steps). Phase 2 에서
  롤아웃 인지 학습(pushforward)·잠재 롤아웃·metriplectic 결선으로 개선 예정.
