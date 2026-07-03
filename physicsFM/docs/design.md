# physicsFM — 물리 Foundation Model 설계 노트

> 2026-07 설계 대화 전체 정리. "초기 mesh 상태 + action → autoregressive 상태열 생성 → 정지(eos)"
> 라는 물리 FM 을 목표로, 아키텍처 후보 비교 → 물리 제약 주입 → PhysicsNeMo 매핑 →
> 본 저장소 구현(경로 A)까지의 의사결정 기록.

---

## 1. 방향 설정 — 왜 시뮬레이터 서로게이트인가

물리 이해 FM 의 두 갈래:

| 접근 | 내용 | 판단 |
|---|---|---|
| **신경망 시뮬레이터(neural surrogate)** | 상태 시퀀스를 GNN/트랜스포머로 학습, "다음 상태 예측" | **채택** — 자체 시뮬레이터가 저비용 학습 데이터 소스 |
| 텍스트 LLM + 물리 QA 파인튜닝 | 언어 추론 강함 | 연속값 동역학 외삽에 근본적으로 약함 |

- 트레이드오프: 시뮬레이터 기반은 데이터가 깨끗하고 스케일 쉬움 ↔ **sim-to-real 갭**(엔진이 가정한 물리 범위를 못 벗어남) → 시나리오 다양화·실측 혼합으로 극복.
- **GPT 유비**: 초기 mesh 상태(위치·방향·속력) + action → 이후 상태가 autoregressive 하게 생성되고 정지(eos)에서 종료. LLM 과 형식이 동형 — 데이터 타입만 다름.
- 핵심 갈림길 = 상태 토큰화: ① 연속값 + 회귀/디퓨전 헤드(정밀도 유지, LLM 도구 포기) vs ② VQ 이산화(GPT 재사용, 양자화 오차가 롤아웃에서 누적). eos 는 "운동에너지 임계 이하" 물리 정지조건으로 치환.

## 2. 프레임워크 — NVIDIA PhysicsNeMo

Modulus(구 SimNet) 후속 오픈소스 Physics-ML 프레임워크, PyTorch 네이티브.

- 모듈: `physicsnemo.models`(NO/GNN/Transformer/Diffusion) · `datapipes`(mesh/point cloud) · `distributed` · `sym`(PINN/PDE residual) · CFD/Curator/Earth-2 위성 패키지
- 두 패러다임: **데이터 주도**(서로게이트 학습 — 본 프로젝트) vs **물리 주도**(`sym`, PDE residual)
- 커스텀 모델 = 1급 시민: `physicsnemo.Module` 은 `torch.nn.Module` 드롭인 대체
- GNN 백엔드: DGL → **PyG 이행 중**(`torch_geometric.data.Data` 입력 시 PyG 자동 선택). DGL deprecated
- 문서: https://docs.nvidia.com/physicsnemo/ · 코드: https://github.com/NVIDIA/physicsnemo

## 3. 아키텍처 후보 3종 비교

### MeshGraphNet (MGN) — Pfaff et al. 2021, arXiv:2010.03409
- Encode-Process-Decode on graph; mesh-space + world-space 이중 엣지(접촉 표현); 노드별 가속도 예측 → 적분 → autoregressive 롤아웃; training noise 로 롤아웃 안정화
- 강점: 비정형 메시 네이티브, **접촉·충돌 등 국소 현상에 강함**, 데이터 효율
- 약점: 비용 ∝ 엣지×MP스텝(대형 메시 부적합), 정보 전파 거리 제한, 오차 누적, 이산화 종속

### Neural Operator (FNO arXiv:2010.08895 / DeepONet arXiv:1910.03193)
- 함수공간 간 연산자 학습(해상도 독립); FNO 는 푸리에 공간 전역 합성곱
- 강점: 해상도 zero-shot, 전역 수용영역, 극속 추론
- 약점: 균일 격자 선호, **spectral bias → 충격/접촉 뭉갬**, autoregressive 생성 패러다임과 결이 다름(eos 부자연)

### Universal Physics Transformer (UPT) — Alkin & Brandstetter 2024, arXiv:2402.12365
- Encoder(supernode 메시지패싱 ~2048 → perceiver → **512 잠재 토큰**) → Approximator(**잠재공간 롤아웃** 트랜스포머) → Decoder(conditional neural field, 임의 (x,t) 질의)
- 강점: 비용이 512 토큰에 고정(메시 크기 디커플), 잠재 롤아웃 저렴, 격자/메시/입자 통합 — **"잠재 토큰열 autoregressive + eos" 유비의 최직접 대응물**
- 약점: 압축 병목(국소 디테일 손실), 신생(노하우 부족), supernode 무작위 샘플링 분산

| 축 | MGN | FNO | UPT |
|---|---|---|---|
| 상태 표현 | 그래프 | 함수(격자) | 잠재 토큰열(512) |
| 수용영역 | 국소(MP 제한) | 전역(1층) | 전역(잠재 어텐션) |
| 메시 확장성 | 나쁨 | 중간 | **우수** |
| 접촉/충격 | **강함** | 약함 | 중간 |
| 시간 전개 | 상태공간 AR | one-shot 위주 | **잠재 AR** |
| 성숙도 | 높음 | 높음 | 낮음 |

**로드맵**: Phase 1 = MGN 으로 물리 충실도 베이스라인(롤아웃 안정성 + eos 검증) → Phase 2 = UPT 잠재 롤아웃으로 스케일/생성 구조 확장. FNO 는 보조.

## 4. 아키텍처 레벨 물리 제약 (UPT/일반)

제약은 "어느 층에 붙이느냐"로 나뉜다:

1. **Encoder — 등변성(Noether: 대칭↔보존)**: EGNN/SEGNN/SE(3)-Transformer. 절대좌표 대신 상대량(거리·상대벡터)만 쓰면 회전/병진 등변이 **수학적으로 정확히** 보장. CNN 의 가중치 공유가 병진 등변을 하드로 박는 것과 동일한 원리 — 형태 제약 안에서 파라미터는 자유 학습.
2. **Approximator — 동역학 형태 제약 (핵심)**:
   - 보존계: Hamiltonian NN — `ż = J∇H`, J 반대칭 → `dH/dt = ∇H·J∇H = 0` (대수적 사실)
   - **소산계(본 프로젝트): metriplectic/GENERIC** — `ż = L∇E + M∇S` (L 반대칭, M PSD) + 퇴화조건 `L∇S = 0, M∇E = 0` → 에너지 정확 보존 + 엔트로피 단조 증가. **엔트로피가 계를 정지로 끌고 가는 메커니즘 = eos 가 아키텍처 보장 성질이 됨**
   - E(에너지)/S(엔트로피) = 학습 스칼라 헤드, L(Poisson)/M(마찰) = 구조 강제된 연산자
   - 참고: Structure-Preserving NN(arXiv:2004.04653), GFINN, Neural Metriplectic Systems(arXiv:2405.16305), GENERIC-FNO(arXiv:2606.08343)
3. **잠재 register 토큰 + projection**: 보존량(질량·운동량) 전용 토큰 + 미분가능 사영
4. **Decoder — 물리공간 하드 제약**: 비발산 출력 `u = ∇×A`(∇·u≡0), 하드 경계조건 `u = g + D·N`(D=근사 거리함수), 양수/범위 제약(softplus/sigmoid — 밀도>0, 접촉 임펄스≥0, 마찰원뿔, ‖q‖=1)

### ⚠ 구현 중 발견한 교정 (중요)
대화에서 제안했던 출력 사영식 퇴화조건 `M̃∇S = M∇S − (∇EᵀM∇S/|∇E|²)∇E` 는 **틀렸다** —
유효 연산자 `P_E·M` 이 비대칭이 되어 dS/dt < 0 반례가 존재(D=2 에서 확인).
올바른 구성은 **GFINN 식 나눗셈 없는 구성**:
```
q_i = u_i(v_iᵀg_E) − v_i(u_iᵀg_E)   → q_iᵀg_E = 0 정확 (skew 이차형식)
M∇S := Q(Qᵀg_S)                      → M=QQᵀ: PSD, M∇E=0 정확
L∇E := PΩ(Pᵀg_E), Ω=ω−ωᵀ            → L=PΩPᵀ: 반대칭, L∇S=0 정확 (P 열 ⊥ g_S)
```
`models/metriplectic.py` 가 이 구성이며, 단위테스트로 |dE/dt|~1e-16, dS/dt≥0,
200-step rk4 에너지 드리프트 7e-13 을 검증했다.

## 5. 경로 A — PhysicsNeMo 매핑

| 설계 요소 | 지원 | 수단 |
|---|---|---|
| mesh/point cloud 데이터 | Native | datapipes, PhysicsNeMo-Mesh |
| 인코더 병목 | Native(유사) | Transolver Physics-Attention(learnable slices ≈ UPT supernode) |
| 잠재 AR 롤아웃 | 직접 구현 | MGN 예제 롤아웃 패턴 |
| 등변 인코더 | 실험적 | symmetry 모듈(2D/3D 회전) 또는 e3nn |
| metriplectic/eos/하드제약 | **커스텀** | `physicsnemo.Module` 서브클래스 |
| 분산/AMP/ckpt/ONNX | Native | distributed 등 |

접촉·충격 도메인 선례: Automotive Crash Dynamics with ML (arXiv:2510.15201).

## 6. 본 저장소 구현 (physicsFM v0)

**전제 발견**: physics/ 의 `free_fall_trajectory` 는 물리엔진이 아니라 **닫힌형 절차 생성기**이고
코드에 "학습된 생성 모델 자리의 mock" 이라 명시 → 이 mock 을 학습 모델로 대체하는
부트스트랩이 본 작업. 파이프라인 자체는 생성기 불가지론적(스키마가 물리 원료만 저장).

핵심 결정(상세는 코드 docstring):
- **D1 물리 시간 복원**: 생성기 타임라인이 비율적으로 물리적(낙하 t_c1, 바운스 2·t_c1·eᵏ) →
  `T_phys = √(2h/g)/t_c1`, 고정 `dt=12ms`, **가변 프레임 수** → 강체 운동이 정확히 뉴턴역학
  (inspect 가 낙하 포물선 항등식 z = h−½gτ² 을 검사). g=9.81 mesh-units/s².
- **D2 가속도형 타깃** `y = x_{t+1}−2x_t+x_{t−1}` (자유낙하 중 상수 → 높이 외삽 유리)
- **D5 eos 라벨**: KE 역방향 cummax ≤ ε·maxKE (바운스 정점 KE≈0 면역) + hold-extension.
  고정 dt·가변 길이가 정규화 시간의 라벨 퇴화(settle 시점이 e 만의 함수로 뭉침)를 해소
- **D8 GFINN metriplectic** (위 교정 참조)
- **D9 노이즈 주입 + 해석적 통계 보정**: 이 생성기는 x/y 가 상수라 클린 std_x/y≈0 —
  노이즈 분산(타깃 +5σ², 속도 +2σ²/dt², 엣지 Δx +2σ²)을 통계에 반영하지 않으면
  표준화 타깃 폭발(실측 3e11). 스모크에서 발견·수정
- 백본: 무의존 MeshGraphNetLite 기본(physicsnemo MGN 은 opt-in 어댑터 — DGL 리스크 격리)

알려진 생성기 아티팩트(문서화): dent rise/ring 시간상수(정규화 단위)가 물리 시간에서
낙하높이 의존이 됨; x/y 정적; 접촉 임펄스/질량 부재(면적 lumped mass 규약으로 대체).

## 7. 활용 가능한 외부 데이터셋

| 데이터셋 | 내용 / 규모 | 포맷 | 본 프로젝트 용도 | 링크 |
|---|---|---|---|---|
| **Kubric MOVi-A/B/C** | 강체 다물체 낙하+접촉 (PyBullet; A: 단순형상 51–64노드, C: 스캔 가정용품 평균 7.9k노드) | TFRecord/영상+메타 | **최유사** — 실물리 강체 접촉으로 확장 시 1순위 | https://github.com/google-research/kubric |
| **DeepMind MeshGraphNets** | cylinder_flow, flag_simple, deforming_plate, sphere_simple 등 10종 | TFRecord (download_dataset.sh) | MGN 베이스라인 재현/사전학습 | https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets |
| **GNS (Learning to Simulate)** | WaterDrop/Sand/Goop 입자 시뮬 | TFRecord/npz | 입자(Lagrangian) 확장 | https://github.com/geoelements/gns (DesignSafe 미러) |
| **The Well** (PolymathicAI) | 15TB, 16개 물리 도메인 시뮬 모음 | HDF5 (HuggingFace) | FM 사전학습 코퍼스 | https://huggingface.co/polymathic-ai |
| **PDEBench** | 1D–3D PDE 벤치마크 (advection/Burgers/NS/Darcy/SWE) | HDF5 | NO 계열 비교 실험 | https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986 |
| **PDEArena** | NS/SWE/Maxwell 서로게이트 벤치마크 | HDF5 | 〃 | https://github.com/pdearena/pdearena |
| **DrivAerNet++** | 차량 8,000대 CFD (다양 형상) | VTK/STL 등 (HF) | Transolver/UPT 스케일 실험 | https://huggingface.co/datasets/MoElrefaie/DrivAerNet |
| **DrivAerML** | DrivAer 변형 500대 고충실 CFD | HF | 〃 (AB-UPT 벤치마크) | https://huggingface.co/datasets/neashton/drivaerml |
| **ShapeNet-Car** (Umetani & Bickel) | 자동차 889대 표면 압력/유동 | 메시+필드 | UPT 논문 벤치마크 재현 | https://www.nobuyuki-umetani.com/publication/2018_sigg_carAero/ |
| (참고 논문) AB-UPT | 자동차 공력 UPT 스케일링 | — | Phase 2 설계 참고 | https://arxiv.org/abs/2502.09692 |

자체 데이터: `generate_rollouts.py` → `data/rollouts.h5` (schema v1 — meshes/rollouts/splits,
물리 dt·z_rigid·ke/pe·접촉 메타 포함; eos 등 라벨은 전부 로더 파생이라 재생성 불요).

## 8. Phase 2 예고

1. hold-extension/크롭으로 시퀀스 학습 데이터화 → UPT 식 잠재 롤아웃(Transolver 백본 검토)
2. MetriplecticHead 를 잠재 동역학에 결선 (dS/dt → eos 보조 입력)
3. 등변 인코더(EGNN) 도입, Kubric MOVi 로 실물리 접촉 확장
4. physics/be 모델 레지스트리에 학습 모델 등록 → mock 대체 (원래 의도 완성)
