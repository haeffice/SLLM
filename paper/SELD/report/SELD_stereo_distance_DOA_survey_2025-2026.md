# SELD 최신 논문 정리 — Stereo/Spatial Audio 기반 Sound Event 거리·방향 동시 추정 (2025~2026)

> 작성일: 2026-06-02
> 범위: stereo(2채널)/spatial audio로부터 sound event의 **거리(distance) + 방향(DOA)** 을 동시에 추정하는 최신 논문
> 검증: 각 논문 arXiv 원문에서 제목·ID·핵심 수치 직접 확인 (적대적 3-vote 검증 통과)

---

## 0. 배경 — DCASE Task 3의 변화와 본 주제의 위치

| 연도 | 입력 포맷 | 거리추정 | 핵심 지표 | 비고 |
|---|---|---|---|---|
| DCASE 2024 Task 3 | FOA(4ch Ambisonics) / MIC(4ch) | ✅ 최초 도입 | F, DOA error(LE), **RDE** | 거리추정이 SELD에 공식 편입 |
| **DCASE 2025 Task 3** | **Stereo(2ch) + Video** | ✅ | **F20°/1**, DOA error | "일반 비디오 콘텐츠" 현실 세팅, 본 주제의 핵심 무대 |

- 2채널 stereo는 좌우 모호성(front-back) 때문에 **방위각(azimuth)** 중심 + **거리** 동시 추정이 과제.
- ⚠️ **2024(FOA/4ch)와 2025(stereo/2ch)는 포맷·지표가 달라 점수 직접 비교 불가.** 같은 챌린지·트랙 내에서만 비교 유효.

---

## 1. 핵심 논문 6편 비교표

| # | 논문 | 챌린지/트랙 | 입력 | 핵심 성능 | 데이터셋 | 모델 크기 | 코드 |
|---|---|---|---|---|---|---|---|
| ① | **NTU SNTL** (2507.00874) | DCASE 2025 T3 (audio-only) | Stereo | **F20°/1 = 45.32%** | STARSS23 stereo + 합성 | ~4M (baseline급) | ✅ 공개 |
| ② | **BiMambaAC** (2506.13455) | DCASE 2025 T3 | Stereo | **F20°/1 ≈ 39.6%** | DCASE2025 T3 dev | **76M** | ✅ (PSELDnet 기반) |
| ③ | **Surrey** (2509.06598) | DCASE 2025 T3 **Track B(AV) 2위** | Stereo + Video | **F = 48.0%** (Setup C) | DCASE2025 T3 stereo AV + 자체 합성 | (미공개) | 기술리포트 |
| ④ | **ToS** (2601.17611) | DCASE 2025 T3 (AV) | Stereo + Video | **DOA error = 16.7°**, 거리+방향 동시 | DCASE2025 T3 stereo dev | 앙상블(3 sub-net) | 기술리포트 |
| ⑤ | **EINV2 ResNet-Conformer** (2507.17941) | DCASE 2024 T3A | FOA(4ch) | **F=40.2%, DOA=17.7°, RDE=0.32** | DCASE2024 T3A dev | (미공개) | 기술리포트 |
| ⑥ | **USTC/NERC-SLIP** (2501.10755) | DCASE 2024 T3 **1위** | FOA(4ch) | 챌린지 1위 (joint vs separate 분석) | DCASE2024 T3 dev | (대형 앙상블) | 분석 논문 |

- **①②③④ = stereo (본 주제 핵심)**, **⑤⑥ = FOA(2024, 거리추정 SELD 직전 SOTA 계보·참고용)**

---

## 2. 논문별 상세 + 링크

### ① NTU SNTL — *stereo audio-only 최우선 추천*
- **제목**: *Improving Stereo 3D Sound Event Localization and Detection: Perceptual Features, Stereo-specific Data Augmentation, and Distance Normalization*
- **저자/소속**: Jun-Wei Yeow, Ee-Leng Tan, Santi Peksi, Woon-Seng Gan (NTU, 싱가포르)
- **요지**: 거대 모델 대신 **(a) 지각적 특징, (b) stereo 전용 데이터 증강, (c) distance normalization(거리를 [0,1]로 스케일링)** 세 엔지니어링으로 성능 향상. **F20°/1 = 45.32%** (audio-only 최상위권).
- **모델 크기**: baseline급 소형(~4M 추정) — "작은 모델 + 좋은 feature/augmentation" 철학.
- **장점**: 재현 가능(코드+데이터 공개), 경량, distance normalization이 범용적.
- **단점**: audio-only라 AV 대비 상한 존재. 제3자 리뷰 글 거의 없음.
- **링크**
  - 논문: https://arxiv.org/abs/2507.00874
  - 코드: https://github.com/itsjunwei/NTU_SNTL_Task3 (CC-BY 4.0)

### ② BiMambaAC — *방법론 최신성 + 유일한 제3자 리뷰*
- **제목**: *Stereo Sound Event Localization and Detection based on PSELDnet Pretraining and BiMamba Sequence Modeling*
- **요지**: 대규모 사전학습 SELD 모델 **PSELDnet**으로 pretrain 후 Conformer를 **양방향 Mamba(BiMamba)** 로 교체. SSM(State Space Model)을 SELD에 본격 적용.
- **성능**: F20°/1 ≈ 39.6% (baseline 대비 대폭 개선).
- **모델 크기**: **76M** (본 목록 중 명시 확인된 유일한 대형 수치).
- **장점**: Mamba로 long-sequence 효율↑, 사전학습 활용. **제3자 리뷰 글 존재**.
- **단점**: 무거움(76M). F-score는 NTU(45.32) 대비 낮음.
- **링크**
  - 논문: https://arxiv.org/abs/2506.13455
  - 리뷰글: https://www.themoonlight.io/en/review/stereo-sound-event-localization-and-detection-based-on-pseldnet-pretraining-and-bimamba-sequence-modeling
  - PSELDnet(사전학습 기반): https://github.com/Jinbo-Hu/PSELDNets

### ③ Surrey — *DCASE 2025 AV 트랙 2위, 최고 F-score*
- **제목**: *Integrating Spatial and Semantic Embeddings for Stereo Sound Event Localization in Videos*
- **요지**: **공간(spatial) + 의미(semantic) 임베딩** 융합. 대규모 합성 audio/AV 데이터로 pretrain. 최종(Setup C: MSI+FAFS+시각 후처리) **F = 48.0%**.
- **장점**: AV 트랙 공식 2위, 본 목록 최고 F. 영상-오디오 융합 레시피.
- **단점**: 파라미터·코드 공개 불명확(기술 리포트).
- **링크**
  - 논문: https://arxiv.org/abs/2509.06598

### ④ ToS (Team of Specialists) — *2026 최신, 거리+방향 동시 명시*
- **제목**: *ToS: A Team of Specialists ensemble framework for Stereo Sound Event Localization and Detection with distance estimation in Video*
- **요지**: **3개 전문가 서브넷 앙상블** — spatio-linguistic / spatio-temporal / tempo-linguistic. 거리·방향 동시 추정, **DOA error 16.7°**.
- **장점**: 사용자 핵심 요구("거리+방향 동시")를 가장 명시적으로 다룸. 최신.
- **단점**: 앙상블이라 무거움, 코드 공개 불명확.
- **링크**
  - 논문: https://arxiv.org/abs/2601.17611

### ⑤ EINV2 ResNet-Conformer — *FOA 계보, 지표가 가장 완전*
- **제목**: *ResNet-Conformer Network with Shared Weights and Attention Mechanism for Sound Event Localization, Detection, and Distance Estimation*
- **저자**: Quoc Thinh Vo, David Han
- **성능**: **F=40.2%, DOA=17.7°, RDE=0.32** (DCASE 2024 T3A audio-only). F/DOA/RDE 3지표 모두 명시.
- **특징**: EINV2 기반, 가중치 공유 + attention, log-mel + intensity vector + 증강.
- **링크**: https://arxiv.org/abs/2507.17941

### ⑥ USTC/NERC-SLIP — *DCASE 2024 1위, joint modeling 설계 가이드*
- **제목**: *An Experimental Study on Joint Modeling for Sound Event Localization and Detection with Source Distance Estimation*
- **요지**: SELD+거리추정에서 **joint vs separate modeling** 3방식 실험 비교. DCASE 2024 Task 3 **1위** 계보.
- **가치**: "거리·방향을 같이 풀 때 모델 구성" 설계 가이드.
- **링크**: https://arxiv.org/abs/2501.10755

---

## 3. 점수 비교 (같은 챌린지 내에서만 유효)

**DCASE 2025 Task 3 (stereo, 거리+방향 동시)**
```
Surrey AV (③)        F = 48.0%          ← AV 트랙 최고
NTU audio-only (①)   F20°/1 = 45.32%    ← audio-only 최고 + 코드공개
BiMambaAC (②)        F20°/1 ≈ 39.6%     (76M, 리뷰글 존재)
ToS (④)              DOA error = 16.7°  (방향 정확도 강점)
```

**DCASE 2024 Task 3 (FOA, 참고)**
```
EINV2 (⑤)   F=40.2% / DOA=17.7° / RDE=0.32
USTC (⑥)    챌린지 1위 (joint modeling 분석)
```

---

## 4. 공통 데이터셋 · 코드 · 챌린지 링크

| 항목 | 링크 |
|---|---|
| DCASE 2025 Task 3 (Stereo SELD in Regular Video) 공식 | https://dcase.community/challenge2025/task-stereo-sound-event-localization-and-detection-in-regular-video-content |
| DCASE 2025 SELD baseline (official) | https://github.com/partha2409/DCASE2025_seld_baseline |
| NTU SNTL 코드 (① ) | https://github.com/itsjunwei/NTU_SNTL_Task3 |
| PSELDnet (② 사전학습 기반) | https://github.com/Jinbo-Hu/PSELDNets |
| STARSS23 (실측 SELD 데이터, stereo 파생 원본) | https://zenodo.org/records/7880637 |
| BiMambaAC 리뷰글 (제3자) | https://www.themoonlight.io/en/review/stereo-sound-event-localization-and-detection-based-on-pseldnet-pretraining-and-bimamba-sequence-modeling |

---

## 5. 한계 / 솔직한 메모

- **제3자 분석글(블로그·칼럼)**: 주제가 매우 최신(2025~2026)·전문적이라 리뷰가 거의 없음. 확인된 것은 **② BiMambaAC의 themoonlight.io 영문 리뷰** 하나뿐. 나머지는 arXiv 원문 / DCASE 공식 페이지 중심.
- **모델 파라미터 수**: 명시 확인된 것은 **② BiMambaAC = 76M**, **① NTU ≈ 4M(baseline급)** 뿐. ③④⑤⑥은 기술 리포트라 수치 누락.
- **2024 vs 2025 비교 불가**: 포맷(FOA 4ch ↔ stereo 2ch)·지표가 달라 표를 가로질러 비교 금지.

---

## 6. 추천 요약

- **재현/실험 시작점** → **① NTU SNTL (2507.00874)**: 코드+데이터 공개, 경량, distance normalization 범용.
- **방법론 최신성 + 리뷰글 동반** → **② BiMambaAC (2506.13455)**: Mamba 도입, 유일한 제3자 리뷰.
- **최고 성능(AV)** → **③ Surrey (2509.06598)**.
- **거리+방향 동시 추정 설계 참고** → **④ ToS (2601.17611)** + **⑥ USTC joint-modeling (2501.10755)**.
