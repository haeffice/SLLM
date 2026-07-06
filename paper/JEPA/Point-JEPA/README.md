# Point-JEPA — JEPA for 3D Point-Cloud SSL (WACV 2025, arXiv:2404.16432)

Self-contained PyTorch 2.8 reimplementation of **Point-JEPA**: the
deployable point-cloud encoder (FPS+KNN tokenizer → greedy *sequencer* →
12-layer ViT) and a full JEPA self-supervised pre-training pipeline
(student + EMA teacher + narrow predictor + proximity-block masking).

- **Paper:** *Point-JEPA: A Joint Embedding Predictive Architecture for
  Self-Supervised Learning on Point Cloud*, Saito & Poovvancheri,
  WACV 2025 — https://arxiv.org/abs/2404.16432
- **Official code (referenced heavily for the architecture):**
  https://github.com/Ayumu-J-S/Point-JEPA  (MIT)
- **Lineage:** I-JEPA (arXiv:2301.08243) → V-JEPA 2 (`../VJEPA2`) /
  LLM-JEPA (`../LLM-JEPA`) → Point-JEPA (JEPA for unordered 3D point sets).
  Tokenizer follows Point-MAE (arXiv:2203.06604).

## 이 논문을 선정한 이유

`paper/`에는 오디오(WavJEPA/SpatialWavJEPA/BAT), 비디오(V-JEPA 2),
언어(LLM-JEPA) JEPA가 구현되어 있어, 마지막 주요 modality인 **3D
포인트클라우드** JEPA를 추가해 JEPA 계열을 사실상 전 영역으로 완성하고자
Point-JEPA를 선정했습니다. 구체적 근거:

1. **최신성·발표처:** WACV 2025 정식 게재 논문으로 TODO의 "최신(25년도~)"
   조건을 충족합니다.
2. **성능 (SOTA):** ModelNet40 linear-SVM **93.7%** (당시 모든 SSL 모델
   중 최고), fine-tuning **93.8/94.1% (OA/Voting)**, ScanObjectNN
   **86.6% OA**, 4개 few-shot 프로토콜 전부 SOTA.
3. **핵심 기여의 명확성:** 비정렬 포인트 토큰에 JEPA를 적용하기 위한
   **greedy sequencer**(인접 인덱스가 공간적으로 근접하도록 토큰을
   정렬 → context/target block 선택이 단순 연속 구간 슬라이스가 됨)는
   재현 가치가 크고 코드로 명확히 표현됩니다.
4. **공개 코드:** MIT 라이선스 공식 구현(`Ayumu-J-S/Point-JEPA`)이
   있어 토크나이저/시퀀서/하이퍼파라미터를 충실히 재현했습니다.
5. **modality 커버리지:** 본 repo의 JEPA 라인업(오디오·비디오·언어)에
   3D를 더해 멀티모달 SSL 관점에서 완결성을 갖춥니다.

## 방법 (구현 기준, 공식 config)

```
1024 pts → FPS C=64 centers → KNN k=32 groups (center-relative)
         → mini-PointNet patch embed (D=384)
         → greedy sequencer  (start: 최소 좌표합 center, 이후 최근접
            미방문 center 반복) — 토큰을 근접순으로 재배열
         → + center 좌표의 MLP positional embedding
         → 12-layer ViT (D=384, heads=6)
```

JEPA 학습 (`PointJEPA_Trainer.py`):

| 항목 | 값 |
|---|---|
| target blocks | 4개, ratio∈(0.15, 0.20), 시퀀스의 **연속 구간** |
| context | ratio∈(0.40, 0.75) 연속 구간 − target 겹침 제거 |
| predictor | depth 6, dim **192** (narrow), mask token + target center pos-emb |
| target | EMA teacher(전체 토큰) 출력, LayerNorm 정규화 |
| loss | smooth-L1 (β=2), 4개 target block 평균 |
| EMA | decay 0.995 → 1.0 (선형 anneal) |
| optim | AdamW, lr 1e-3, warmup → cosine |

## 구성 파일

| 파일 | 역할 |
|---|---|
| `PointJEPA.py` | 추론 인코더: FPS/KNN/mini-PointNet 토크나이저 + greedy sequencer + 12L ViT + `from_checkpoint` |
| `PointJEPA_Trainer.py` | JEPA 학습: student + EMA teacher + narrow predictor + block masker + smooth-L1 |
| `train_pointjepa.py` | `transformers.Trainer` 기반 SSL 사전학습 (Dataset/DataLoader/콜백) |
| `config.yaml` | 학습 단일 설정 파일 |
| `run_train_PointJEPA.sh` | torchrun 런처 (기존 checkpoint 존재 시 시작 거부) |
| `make_synthetic_manifest.py` | CPU 스모크용 합성 포인트클라우드/매니페스트 생성 |
| `requirements.txt` | 의존성 (torch==2.8.0, CPU; FPS/KNN은 pure-torch) |

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt
```

FPS / KNN / greedy-sequencer는 순수 torch로 구현되어 `torch_cluster`/
`pointnet2_ops` 같은 빌드 의존성이 필요 없습니다.

## 데이터셋

포인트클라우드를 `.npy`/`.npz` (배열 `(P, 3)` 또는 `(P, ≥3)`, XYZ만 사용)로
저장하고 매니페스트로 가리킵니다:

```json
{"data": [{"points_id": "02691156/xxxx.npy"}, ...]}
```

`points_path = data.points_root / points_id`. 단위 구로 정규화 후
`num_points` 샘플링(train 랜덤 / eval 앞부분).

논문에서 사용한 공개 데이터셋:

- **ShapeNet** (사전학습, 41,952 instances):
  https://shapenet.org  (또는 Point-BERT 전처리본
  https://github.com/lulutang0608/Point-BERT — `ShapeNet55` .npy)
- **ModelNet40** (downstream 분류, 1024 pts):
  https://modelnet.cs.princeton.edu  (전처리본
  https://github.com/ma-xu/pointMLP-pytorch)
- **ScanObjectNN** (실세계 분류, 2048 pts):
  https://hkust-vgd.github.io/scanobjectnn/

다운로드 후 위 JSON 스키마의 `train.json`/`valid.json`을 만들고
`config.yaml`의 `data` 섹션을 가리키면 됩니다. 공식 저장소
`Ayumu-J-S/Point-JEPA`에 데이터 전처리 스크립트가 있습니다.

## 학습

```bash
bash run_train_PointJEPA.sh config.yaml
NPROC_PER_NODE=4 bash run_train_PointJEPA.sh config.yaml   # 멀티-GPU
```

모든 인자는 `config.yaml` 한 파일에 정리되어 있고, 셸 스크립트는
`train.output_dir`에 checkpoint가 있으면 학습을 시작하지 않습니다. HF
checkpoint와 함께 step 번호가 들어간 student-only `.pt`
(`pointjepa_student_step{N}.pt`)가 저장되어
`PointJEPA.from_checkpoint(...)`로 바로 로드됩니다.

학습 로그(요청 사항 충족): 모듈별 학습 가능 파라미터 수; 첫 배치 첫
샘플의 **포인트클라우드 경로**(모델 feed 전); `logging_steps`마다 step /
train_loss / (valid_loss) / lr / ema_decay; `save_steps`마다 step 번호
포함 checkpoint; `init_from` 0개 로드 시 즉시 중단.

## CPU 스모크 테스트

```bash
python make_synthetic_manifest.py --out /tmp/pj_smoke
bash run_train_PointJEPA.sh /tmp/pj_smoke/config.yaml
```

스모크 config는 토큰/인코더를 작게 축소해 CPU에서 수 step 만에 끝납니다
(train_loss 감소 확인됨).

## 추론 (frozen feature)

```python
from PointJEPA import PointJEPA
m = PointJEPA.from_checkpoint("pointjepa_student_step30000.pt")  # eval+no-grad
feats = m(points)                       # (B, 64, 384) per-token
desc  = m.get_shape_representation(points)   # (B, 768) SVM-eval descriptor
```
