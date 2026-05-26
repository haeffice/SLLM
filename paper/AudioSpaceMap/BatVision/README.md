# BatVision — 양이(binaural) 에코로부터 공간 깊이맵(depth map) 생성

스피커가 쏜 짧은 chirp의 **양이 에코**(두 마이크가 받은 반향)만으로 정면 시야의
**깊이맵(depth map)** 을 추론하는 모델. 박쥐의 반향정위(echolocation)에서 영감을 받아
"두 귀로 듣고 앞 공간의 3-D 배치를 본다". 출력 깊이맵은 벽·복도·문 개구부·대략적인
가구 윤곽을 방위각·고도·거리상에서 해상하는 **조밀한 공간 맵**이다. EchoScan이
*시뮬레이션 다채널 RIR → 위에서 본 floorplan*을 다룬다면, BatVision은 *실측 양이 녹음
→ 정면 깊이 이미지*를 다루는 상보적 접근이다.

> Jesper H. Brunetto, Sascha Hornauer, Stella X. Yu, Fabien Moutarde,
> **"The Audio-Visual BatVision Dataset for Research on Sight and Sound"**, IROS 2023.
> arXiv: [2303.07257](https://arxiv.org/abs/2303.07257)
>
> Jesper Haahr Christensen, Sascha Hornauer, Stella X. Yu,
> **"BatVision: Learning to See 3D Spatial Layout with Two Ears"**, ICRA 2020.
> arXiv: [1912.07011](https://arxiv.org/abs/1912.07011)

**공개 코드 (구현 시 크게 참고):**
- 데이터셋 + baseline(audio-only U-Net): <https://github.com/AmandineBtto/Batvision-Dataset>
  (`UNetSoundOnly/`의 dataloader·model·train 로직을 본 구현이 따른다)
- 원본 BatVision 코드: <https://github.com/SaschaHornauer/Batvision>
- U-Net generator 백본 출처(pix2pix): <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>

## 이 논문을 선정한 이유

- **주제 정합성 (audio → space map):** chirp 에코만으로 정면 공간의 **깊이맵**을
  직접 출력한다. 깊이맵은 벽·문·복도·가구를 방위·고도·거리로 표현하는 조밀한 공간
  맵이므로 "오디오에 기반한 space map 생성" 요구에 직접 부합한다.
- **EchoScan과의 상보성:** EchoScan은 **시뮬레이션** 다채널 RIR로 **top-down floorplan**
  을 만든다. BatVision은 **실측** 양이 녹음으로 **정면 depth map**을 만든다. 데이터
  취득 방식(시뮬 vs 실측), 출력 형식(평면도 vs 깊이), 백본(1-D ResNet+MA vs 2-D U-Net)
  이 모두 달라, 같은 niche를 서로 다른 축에서 보완한다.
- **공개 데이터셋 + 코드 (재현성):** 대다수 음향-공간 연구가 비공개 시뮬레이션 데이터에
  머무는 반면, BatVision은 **대규모 실세계 audio-visual 데이터셋을 공개**하고(아래
  다운로드 링크), baseline 학습/평가 코드까지 GitHub에 공개한다. 본 과제의 "다운로드
  가능한 학습 데이터 명시" 요건을 가장 확실히 만족한다.
- **영향력/계보:** ICRA 2020 원본 + IROS 2023 데이터셋 논문으로 이어지는, "audio→depth"
  반향정위 분야의 대표 reference. 후속 연구(Beyond Image to Depth 등)가 동일한 깊이
  오차지표(δ<1.25, RMSE, abs_rel)를 공유한다.
- **실용 장치:** 블루투스 스피커(JBL Flip4) + 양이 마이크로 구성되는 저가 장치라
  로봇·내비게이션에 적용성이 높다.

## 아키텍처

```
binaural waveform  X ∈ R^{2×T}        2 ch, 44.1 kHz (V1 ~72.5 ms / V2 ~0.45 s)
  └─ 채널별 magnitude spectrogram      STFT n_fft=512, win=64, hop=16, |·|^1
        → S ∈ R^{2×F×Ts}               256×256로 resize
  └─ U-Net generator (pix2pix)         unet_256 = 8 down / 8 up + skip connection
        → D ∈ R^{1×256×256}            sigmoid → 정규화 깊이 [0,1]
metres            depth_m = D · max_depth   (V1: 12 m, V2: 30 m)
```

- **입력 전처리:** 파형 → STFT 크기 스펙트로그램(dB 변환 없음, 원 논문과 동일) →
  채널 2개를 쌓아 `(2,256,256)`로 resize. (torchaudio 대신 `torch.stft`, torchvision
  대신 `F.interpolate`를 써서 의존성을 줄임 — 결과는 동일.)
- **백본:** pix2pix `UnetGenerator`(reference baseline이 그대로 차용). 입력 채널 2,
  출력 채널 1, BatchNorm, `depth_norm`이면 출력단 Sigmoid.
- **손실:** **유효 픽셀(깊이 > 0)에 대한 L1**. 깊이 0은 센서 hole/클리핑이므로 제외
  (reference의 `criterion(pred[gt!=0], gt[gt!=0])`와 동일).
- **깊이 정규화:** 깊이를 mm→m 변환 후 `max_depth`로 클리핑하고 MinMax로 `[0,1]`
  스케일. 평가 시 `max_depth`를 곱해 metres로 되돌려 오차지표를 계산.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `BatVision.py` | 추론 모델(U-Net generator) + 스펙트로그램 front end + 손실/깊이지표 + `from_checkpoint`/`predict`. |
| `make_batvision_dataset.py` | **합성 smoke 데이터셋 생성기**(실 데이터 대체 아님). 실 BatVision **V1** 디스크 레이아웃으로 작은 데이터셋을 만들어 CPU 파이프라인 점검. |
| `train_batvision.py` | HF `PreTrainedModel` 래퍼 + `Trainer` 학습 + V1/V2 Dataset + 콜백. |
| `eval_batvision.py` | 학습과 분리된 load/inference + 깊이 오차지표(RMSE/abs_rel/δ) + 시각화. |
| `config.yaml` | 모든 학습/모델/데이터 인자(단일 진실 공급원). |
| `run_train_BatVision.sh` | 실행 스크립트(yaml만 인자, 기존 체크포인트 존재 시 시작 거부). |
| `requirements.txt` | CPU 의존성, torch==2.8.0. |

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt
# V2(.wav) 학습 시에만 soundfile 필요 (V1/smoke는 불필요)
```

## 데이터셋 준비 (다운로드 / 획득 방법)

BatVision은 **실세계 공개 데이터셋**이다 (CC-BY-SA-4.0). 두 버전:

| 버전 | 환경 | 규모 | 오디오 | 마이크 | 깊이 클립 |
|---|---|---|---|---|---|
| **V1** | UC Berkeley 오피스 | ~52,220 instance | ~72.5 ms, 44.1 kHz, 양이 `.npy` 파형 | MAONO AU-410 ×2 (실리콘 귀) | 12 m |
| **V2** | Mines Paris 캠퍼스(다양한 재질/형상, 일부 실외) | ~3,120 instance | ~0.45 s, 44.1 kHz, stereo `.wav` | 3Dio Free Space | 30 m |

공통 하드웨어: 스피커 JBL Flip4, chirp 20 Hz→20 kHz / 3 ms.

**다운로드 링크 (둘 중 하나):**
- 주소: <https://cloud.minesparis.psl.eu/index.php/s/qurl3oySgTmT85M>
- 미러(DOI, Recherche Data Gouv): <https://doi.org/10.57745/HYLZNL>
  (`https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/HYLZNL`)

**디스크 레이아웃 (config가 기대하는 형식):**
- **V1** — 하나의 루트에 `train.csv`/`val.csv`/`test.csv`. CSV 컬럼:
  `depth path`, `audio path left`, `audio path right` (모두 루트 기준 상대경로).
  깊이는 mm 단위 `.npy`, 오디오는 좌/우 `.npy` 파형.
- **V2** — 루트 아래 **location 폴더마다** `train.csv`/`val.csv`/`test.csv`. CSV 컬럼:
  `depth path`+`depth file name`, `audio path`+`audio file name` (stereo `.wav`).

`config.yaml`에서 `data.variant`(`v1`|`v2`), `data.dataset_dir`(루트), `data.max_depth`
(12/30)만 맞추면 그대로 학습된다.

### CPU smoke 테스트 (다운로드 없이 배관 검증)

실 데이터는 수 GB이므로, 동일한 **V1 레이아웃**의 합성 데이터로 전체 파이프라인을
점검할 수 있다. 합성 신호는 약식 물리(평균 깊이 → 왕복 지연 `2·d/c`로 에코 onset)를
따르므로 학습이 동작은 하지만 **벤치마크가 아니다**.

```bash
# 하나의 루트에 train/val/test CSV + 공유 audio/·depth/ 생성
python make_batvision_dataset.py --out data/smoke --splits train:8,val:4,test:4 --seed 0

# config.yaml에서 dataset_dir=data/smoke, generator=unet_128, images_size=128 으로
# 줄여 빠르게 검증 (unet_256/256은 GPU 권장):
NPROC_PER_NODE=1 ./run_train_BatVision.sh config.yaml
```

## 학습

```bash
# config.yaml의 data.dataset_dir / train.output_dir 등을 채운 뒤:
NPROC_PER_NODE=8 ./run_train_BatVision.sh config.yaml
```

- 옵티마이저: AdamW(+ 선택적 warmup·cosine). `config.yaml`이 유일한 인자.
- 논문 설정: batch 256, lr 1e-3(V1)/2e-3(V2), L1 손실, 깊이 MinMax 정규화.
- 로그: 모듈별 학습 가능 파라미터 수(encoder Conv / decoder ConvT / norm / total),
  첫 배치 첫 샘플의 **오디오 경로**(feed 이전), `logging_steps`마다
  step/train_loss/valid_loss/lr, `save_steps`마다 `batvision_step{step}.pt` 저장.
- `run_train_BatVision.sh`는 `output_dir`에 체크포인트가 이미 있으면 시작을 거부한다.
- `model.generator`(`unet_256`/`unet_128`)와 `data.images_size`(256/128)는 **반드시
  일치**해야 한다(불일치 시 학습 시작 전 에러).

## 평가

```bash
python eval_batvision.py --config config.yaml \
    --checkpoint /path/to/ckpts/batvision/batvision_final.pt \
    --csv test.csv --save-viz viz_out --max-viz 8
```

- `BatVision.from_checkpoint`가 파라미터를 **freeze + eval()** 하고, 0개 매칭 시 로드를
  거부한다(무작위 가중치 추론 사고 방지). `predict`도 `eval()/no_grad` 재확인.
- metres 기준 **RMSE·MAE·abs_rel·log10·δ<1.25/1.25²/1.25³** 출력. `--save-viz`로
  입력 스펙트로그램 / GT 깊이 / 예측 깊이 비교 PNG 저장(matplotlib 필요).

## 추론 예시

```python
import torch
from BatVision import BatVision, binaural_spectrogram, resize_2d
m = BatVision.from_checkpoint("batvision_final.pt")   # freeze + eval, max_depth 포함
wav = torch.randn(2, 3197)                            # (2, T) 양이 파형 (~72.5 ms)
spec = resize_2d(binaural_spectrogram(wav), m.image_size)[None]   # (1, 2, S, S)
out = m.predict(spec)
depth_m = out["depth_m"][0, 0]                        # (S, S) metres
```

## 참고 / 주의사항

- **스펙트로그램 등가성:** 본 구현은 `torch.stft`(Hann, center, reflect)로 reference의
  `torchaudio.transforms.Spectrogram(n_fft=512, win=64, hop=16, power=1.0)`와 동일한
  크기 스펙트로그램을 만든다. dB 변환은 reference와 마찬가지로 적용하지 않는다.
- **U-Net baseline 한정:** 모델 입력은 스펙트로그램 `(2,S,S)`이다. `audio_format:
  waveform`은 본 U-Net과 형상이 맞지 않아 학습 시작 전 거부된다.
- **V2 cut:** V2는 reference처럼 파형을 `2·max_depth/c·sr` 샘플로 잘라 STFT한다.
- **합성 데이터는 벤치마크 아님:** `make_batvision_dataset.py`는 파이프라인 점검용이다.
  의미 있는 수치는 실 BatVision V1/V2에서 학습할 것.
- **해상도-비용:** 논문 해상도(256² depth, unet_256, batch 256)는 GPU가 필요하다.
  CPU에서는 `unet_128`/`images_size:128`/작은 batch로 동작 확인만 권장.
