# EchoScan — 음향 에코로부터 공간 맵(floorplan) 생성

다채널 **room impulse response(RIR)** = 작은 마이크 어레이가 받은 음향 에코로부터
방의 **공간 맵**을 직접 추론하는 모델. 2-D **floorplan** segmentation 이미지와
1-D **height** 프로파일을 예측하고, 둘의 외적으로 3-D 방 부피를 복원한다. 곡선 벽
이나 비(非)볼록 형태도 벽 개수를 가정하지 않고 처리한다.

> Inmo Yeon, Iljoo Jeong, Seungchul Lee, Jung-Woo Choi,
> **"EchoScan: Scanning Complex Room Geometries via Acoustic Echoes"**,
> *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2024.
> arXiv: [2310.11728](https://arxiv.org/abs/2310.11728) ·
> DOI: [10.1109/TASLP.2024.3485516](https://dl.acm.org/doi/10.1109/TASLP.2024.3485516)

**공개 코드:** 저자의 공식 GitHub 저장소를 (작성 시점 기준) 찾지 못했다. 따라서 본
구현은 논문 본문·그림의 서술에 충실하게 from-scratch 작성했으며, 명시되지 않은 일부
세부(디코더 채널 스케줄, "projective skip connection")는 단순화하고 아래
[주의사항](#참고--주의사항)에 명시했다. 공식 코드가 공개되면 README에 경로를 추가하고
그에 맞춰 재정렬할 것.

## 이 논문을 선정한 이유

- **주제 정합성 (audio → space map):** "오디오에 기반하여 주어진 공간의 space map을
  생성"이라는 요구에 **가장 직접적으로** 부합한다. 대부분의 공간 음향 연구가 음원
  위치추정(DoA)·거리추정·잔향파라미터에 머무는 반면, EchoScan은 출력 자체가 방의
  **floorplan 이미지 + height 맵**, 즉 공간 맵이다.
- **임의 형태 일반화:** 기존 기법(예: 1차 반사 기반 기하 추정)은 shoebox/볼록 방을
  가정하지만, EchoScan은 segmentation으로 정식화하여 **L·T자, 곡선 벽 등 복잡 형태**를
  벽 개수 사전지식 없이 추론한다.
- **저·고차 반사 동시 활용 (핵심 기여):** RIR의 저차 반사(기하 단서)와 고차 반사(공간
  전체 정보)의 복잡한 관계를 **Multi-Aggregation(평균 + GeM 풀링)** 모듈로 함께
  포착하는 것이 성능 향상의 핵심이다.
- **현실적 장치 구성:** 5 cm 원형 6-mic 어레이 + 중앙 스피커 = 상용 음성비서 스피커
  형태와 호환. 실험 재현·실제 적용성이 높다.
- **검증/영향력:** IEEE/ACM TASLP(오디오 분야 top journal) 게재. RGI-Net 등 동일
  저자군의 후속·관련 연구로 이어지는, 이 niche에서 사실상의 대표 연구다.
- **데이터 획득 가능성:** 데이터셋이 **시뮬레이션(Pyroomacoustics)** 으로 생성되므로
  gated 다운로드 없이 누구나 학습 데이터를 확보할 수 있다(아래 참고).

## 아키텍처

```
RIR  X ∈ R^{M×N}            M=6 omni mic (5 cm 원형 어레이), N=1024 @ 8 kHz, 직접음 제거
  └─ Encoder (1-D ResNet)   stem Conv1d(k=9, /2) + 6× 잔차 conv block
        → F ∈ R^{C_L×D_L}   C_L=1024 채널, D_L=16 시간스텝
  └─ Multi-Aggregation      시간축 AvgPool(ρ=1) ‖ GeM(ρ=3),
        → m ∈ R^{512}          각각 Linear(1024→256)+L2-norm 후 concat
  ├─ Floorplan decoder      m→(2,16,16) reshape, k× UpConv(×2)  (16·2^k=b)
  │     → Y^LW ∈ R^{b×b}      b=1024 (±10.24 m, 2 cm/px), sigmoid
  └─ Height decoder         단일 Linear
        → y^H ∈ R^{h}          h=512 (±5.12 m, 2 cm/px), sigmoid
3-D 복원  Y^3D = Y^LW ⊗ y^H
```

- **좌표계:** 항상 장치(어레이 중심)를 원점으로 하는 로컬 좌표. floorplan은 위에서 내려
  본 평면도, height는 바닥/천장의 상대 높이 프로파일.
- **손실 (Eq. 3):** `L = MSE(LW) + α·Dice(LW) + β·MSE_PIT(H)`, α=0.3, β=1.0.
  Dice는 경계 학습을 돕고, height는 상하 반전 모호성 때문에 **Permutation-Invariant
  Training**(원본/뒤집은 타깃 중 더 낮은 손실 선택)으로 학습.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `EchoScan.py` | 추론 모델(Encoder/MA/두 디코더) + 손실/IoU + `from_checkpoint`/`predict`/`reconstruct_3d`. |
| `make_echoscan_dataset.py` | **Pyroomacoustics 기반 데이터셋 생성기**(RIR 시뮬레이션 + floorplan/height 라벨 래스터화). |
| `train_echoscan.py` | HF `PreTrainedModel` 래퍼 + `Trainer` 학습 파이프라인 + 콜백. |
| `eval_echoscan.py` | 학습과 분리된 load/inference + floorplan/height IoU + 시각화. |
| `config.yaml` | 모든 학습/모델/데이터 인자(단일 진실 공급원). |
| `run_train_EchoScan.sh` | 실행 스크립트(yaml만 인자, 기존 체크포인트 존재 시 시작 거부). |
| `requirements.txt` | CPU 의존성, torch==2.8.0. |

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt
```

## 데이터셋 준비 (다운로드 / 획득 방법)

EchoScan의 학습 데이터는 **단일 다운로드 파일이 아니라 시뮬레이션으로 생성**한다.
`make_echoscan_dataset.py`가 논문의 파이프라인(랜덤 방 폴리곤 → 장치 배치 →
Pyroomacoustics ray-tracing으로 다채널 RIR 시뮬레이션 → floorplan/height 라벨
래스터화)을 그대로 재현하므로, 아래 한 줄이면 학습 데이터가 **로컬에 확보**된다.

```bash
# Basic Room 세트 (논문: 학습 1.2 M / 테스트 6 k RIR — 규모는 --n 으로 조절)
python make_echoscan_dataset.py --out data/train --n 200000 --seed 0
python make_echoscan_dataset.py --out data/test  --n 6000   --seed 1
```

각 샘플은 `{idx}.npz`(`rir` (M,N) float32, `floor_packed` = floorplan 비트팩,
`height` (h,) uint8)로 저장되고 `manifest.json`에 메타와 함께 인덱싱된다.

**실측 레이아웃 기반(Manhattan-Atlanta) 학습을 원할 경우** — 논문은 실제 건물 평면을
방 폴리곤으로 사용한다. 폴리곤 소스는 공개 데이터에서 얻는다:
- **Matterport3D**(Manhattan, 직각 벽): <https://niessner.github.io/Matterport/>
  (데이터 사용 동의서 제출 후 다운로드)
- **AtlantaNet / PanoContext·Stanford 2D-3D-S**(Atlanta, 곡선 벽 허용 주석):
  <https://github.com/crij-as/AtlantaNet>
얻은 평면 폴리곤을 `make_polygon`이 반환하는 형식(CCW, metres)으로 변환해
`--out` 파이프라인에 주입하면 동일하게 RIR을 시뮬레이션할 수 있다(폴리곤 로더는
사용 데이터 라이선스상 본 저장소에 포함하지 않음).

### CPU smoke 테스트 (작은 맵·짧은 RIR, ISM only)

```bash
python make_echoscan_dataset.py --out /tmp/echoscan_smoke/train --n 12 \
    --floorplan-size 128 --height-size 64 --rir-length 256 --no-ray-tracing
python make_echoscan_dataset.py --out /tmp/echoscan_smoke/test  --n 6 --seed 1 \
    --floorplan-size 128 --height-size 64 --rir-length 256 --no-ray-tracing
# config.yaml의 floorplan_size/height_size/rir_length를 위 값과 동일하게 맞춘 뒤:
NPROC_PER_NODE=1 ./run_train_EchoScan.sh config.yaml
```
> 작은 맵(±1.28 m)은 방보다 좁아 라벨이 거의 전부 interior가 되므로 **배관 검증용**
> 이다. 의미 있는 학습은 논문 해상도(b=1024, h=512, N=1024)에서 수행할 것.

## 학습

```bash
# config.yaml의 train_manifest/output_dir 등을 채운 뒤:
NPROC_PER_NODE=8 ./run_train_EchoScan.sh config.yaml
```

- 옵티마이저: AdamW + linear-warmup + cosine decay. `config.yaml`이 유일한 인자.
- 로그: 모듈별 학습 가능 파라미터 수, 첫 배치 첫 샘플의 **RIR 경로**(feed 이전),
  `logging_steps`마다 step/train_loss/valid_loss/lr, `save_steps`마다
  `echoscan_step{step}.pt`(이식 가능 체크포인트) 저장.
- `run_train_EchoScan.sh`는 `output_dir`에 체크포인트가 이미 있으면 시작을 거부한다.
- 모델 입력 해상도(`data.floorplan_size`/`height_size`/`rir_length`)는 데이터셋
  생성 시 값(manifest `meta`)과 **반드시 일치**해야 한다.

## 평가

```bash
python eval_echoscan.py --config config.yaml \
    --checkpoint /path/to/ckpts/echoscan/echoscan_final.pt \
    --manifest data/test/manifest.json \
    --save-viz viz_out --max-viz 8
```

- `EchoScan.from_checkpoint`가 파라미터를 **freeze + eval()** 하고, 0개 매칭 시
  로드를 거부한다(무작위 가중치로 추론하는 사고 방지). `predict`도 `eval()/no_grad`
  재확인.
- floorplan IoU, height IoU(상하 반전 보정) 출력. `--save-viz`로 GT/예측 평면도와
  height 프로파일 비교 PNG 저장(matplotlib 필요).

## 추론 예시

```python
import torch
from EchoScan import EchoScan
m = EchoScan.from_checkpoint("echoscan_final.pt")     # freeze + eval
rir = torch.randn(1, 6, 1024)                          # (B, M, N)
out = m.predict(rir)                                   # floorplan (B,b,b), height (B,h)
vol = EchoScan.reconstruct_3d(out["floorplan"][0], out["height"][0])  # (b,b,h) bool
```

## 참고 / 주의사항

- **직접음 제거:** 마이크가 음원에서 ~5 cm이므로 직접음은 RIR 맨 앞 몇 샘플에 위치.
  생성기는 에너지 onset에서 N=1024 샘플 윈도우를 잡고 앞쪽 직접음 구간을 0으로 만든다
  (논문의 "direct sound omitted"에 대한 실용적 근사).
- **디코더 단순화:** floorplan 디코더의 채널 스케줄은 `256→128→…→16`로 자체 설정했고,
  논문의 "projective skip connection"은 본 구현에서 생략했다(전역 MA 벡터만 사용).
  공식 코드 공개 시 정렬 예정.
- **해상도-비용:** 논문 해상도(1024² floorplan)는 CPU에서 추론은 가능하나(≈0.1 s/샘플)
  대규모 학습·`reconstruct_3d`(1024²×512 voxel)는 GPU/메모리가 필요하다. smoke 설정은
  맵/RIR을 축소한다.
- **height PIT:** 라벨에는 상하 방향이 정해져 있으나, 모델의 상하 대칭성을 고려해
  손실·평가 모두 반전 보정(PIT)을 적용한다.
