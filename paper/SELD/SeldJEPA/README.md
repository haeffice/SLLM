# SeldJEPA — 실시간 SELD를 위한 Causal-Conformer LeJEPA 인코더 + Multi-ACCDOA 분류기

> 2-stage 구성. **Stage-1**: 인과(causal) Conformer 오디오 인코더를 **LeJEPA/SIGReg**
> 자기지도학습으로 사전학습. **Stage-2**: 동결(frozen)된 인코더 위에 **Multi-ACCDOA**
> (N=3 track) 헤드를 **ADPIT** 손실로 학습하여 Sound Event Localization & Detection 수행.
>
> 참고 논문:
> - R. Balestriero & Y. LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning
>   Without the Heuristics", arXiv:2511.08544 (2025).
> - K. Shimada et al., "Multi-ACCDOA: Localizing and Detecting Overlapping Sounds...",
>   ICASSP 2022, arXiv:2110.07124. ("ACCDOA", ICASSP 2021, arXiv:2010.15306.)
> - A. Gulati et al., "Conformer: Convolution-augmented Transformer for Speech
>   Recognition", Interspeech 2020.
>
> 공개 코드: 본 디렉터리는 위 논문들의 아이디어를 결합한 **from-scratch 구현**이다.
> SIGReg 구현과 multi-crop/HF 학습 골격은 형제 디렉터리 `../../JEPA/LeJEPA/`,
> 멀티채널 공유 마스킹은 `../../JEPA/SpatialWavJEPA/`, 지도학습+평가 골격은
> `../../SpatialAudioLLM/PhaseCoder/` 의 관례를 따른다.

## 이 논문을 선정한 이유

실시간 SELD는 (1) 사람이 지연을 인지하지 못하는 짧은 청크 단위의 **인과 스트리밍**
인코딩과, (2) 동시 발생 음원을 다루는 **다중 트랙(Multi-ACCDOA)** 디코딩이 동시에
필요하다. LeJEPA는 EMA teacher·stop-gradient·스케줄러 없이 **단일 하이퍼파라미터**로
안정적으로 학습되는 최신 JEPA 계열 자기지도학습이라, 레이블이 비싼 SELD에서 인코더를
비지도로 키운 뒤 작은 레이블로 헤드만 붙이는 2-stage 전략에 적합하다.

## 아키텍처

```
2ch wav ─ features.py ─► feat (C_feat=4, T_f, M=64)   [log-mel_L, log-mel_R, sinIPD, cosIPD]
  Stage-1 (SeldConformer.py)
    feat ─ CausalConvSubsampling(time x2 -> 50 fps, freq x4) ─► (T_enc, D=512)
         ─ + sinusoidal PE ─ ConformerBlock x16 (Macaron FFN / chunk-causal MHSA / causal depthwise-conv, LayerNorm)
         ─ causal pool(last) ─► z in R^512
    LeJEPA:  predictor(z_view) -> global z ;  SIGReg({z}) -> N(0, I)
             L = L_pred + sigreg_coeff * SIGReg     (no EMA / stop-grad / scheduler)
  Stage-2 (SeldMultiACCDOA.py)
    frozen encoder ─► (T_enc, D) ─ temporal avg-pool x5 ─► 10 Hz ─ Linear(D -> N*C*2)
                   ─► pred (T_label, N=3, C, 2)   ADPIT loss (13 duplicated-permutation layouts)
    decode: ||(x,y)|| > 0.5 = active, azimuth = atan2(y,x), 같은 클래스 트랙 30° 이내 통합
```

- **인코더 크기**: `encoder_dim=512, num_layers=16` → 약 **98M (~0.1B)** 파라미터.
- **인과성(실시간)**: depthwise conv 좌측 패딩만, self-attention은 청크 단위 causal mask
  (`chunk_frames=5` = 100 ms), conv 모듈은 BatchNorm 대신 LayerNorm, pooling은 마지막
  스트리밍 상태. → 알고리즘 지연이 **1 청크(100 ms)** 로 bound.
- **청크 길이 100 ms**: 1 청크 = 1 라벨 프레임(10 Hz)이라 프레임율 정렬이 단순.
- **특징(2채널)**: log-mel×2(+암묵적 ILD) + sin/cos-IPD → 4채널. (FOA 관련 주의는 하단 참고.)

## 참고 구현 (Reference)

- LeJEPA/SIGReg: `../../JEPA/LeJEPA/LeJEPA.py` 의 `SIGReg`(Epps-Pulley + Gauss-Hermite)
  클래스를 그대로 사용. 멀티크롭 predict-the-global-views 레시피 동일.
- 멀티채널 공유 마스킹: `../../JEPA/SpatialWavJEPA/SpatialWavJEPA_Trainer.py` 의
  cross-channel `repeat` 마스킹과 동일한 "모든 채널 동일 마스크" 원칙(`features.spec_augment`).
- 지도학습 + 평가 골격: `../../SpatialAudioLLM/PhaseCoder/` (`from_checkpoint`/`predict`,
  param-count/first-sample 콜백, eval 출력 블록).

## 파일 구성

| 파일 | 설명 |
|---|---|
| `features.py` | 2ch→4채널 feature(log-mel×2 + sin/cos-IPD), 채널 일관 SpecAugment, L/R-swap |
| `SeldConformer.py` | Stage-1 인코더(인과 Conformer) + SIGReg + LeJEPA `compute_loss` + `from_checkpoint` |
| `train_seld_jepa.py` | Stage-1 HF 래퍼 + Trainer + 콜백 + 멀티크롭 Dataset + `main()` |
| `SeldMultiACCDOA.py` | Stage-2 모델(동결 인코더 + Multi-ACCDOA 헤드) + ADPIT + decode |
| `train_seld_accdoa.py` | Stage-2 HF 래퍼 + Trainer + 콜백 + 라벨 Dataset + `main()` |
| `eval_seld_accdoa.py` | Stage-2 평가(동결 로드, ER/F/LE/LR/ε_SELD) |
| `seld_metrics.py` | azimuth-only SELD 메트릭(Hungarian 매칭) |
| `make_seld_dataset.py` | 합성 2ch 데이터 생성 + manifest + smoke config |
| `config_pretrain.yaml` / `config_seld.yaml` | 단일 설정 파일(각 stage) |
| `run_train_SeldJEPA.sh` / `run_train_SeldACCDOA.sh` / `run_eval_SeldACCDOA.sh` | 런처 |
| `requirements.txt` | 고정 의존성(torch==2.8.0 기준) |

## 설치

```bash
python -m venv .venv && source .venv/bin/activate     # 또는 리포 공용 paper/.venv 사용
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt
```

## 데이터셋 준비

manifest 형식(두 stage 공용):

```json
{ "meta": {"fs":24000, "n_channels":2, "n_classes":12, "n_tracks":3, "label_hop_ms":100},
  "data": [ {"audio_id":"wav/clip_0000.wav", "label_id":"label/clip_0000.npy"}, ... ] }
```

- `audio_id`: 2채널 wav (Stage-1은 이 항목만 사용).
- `label_id`: `(T_label, C, 3, 2)` numpy. 프레임·클래스별 최대 3개 음원의 azimuth 단위벡터
  `(x,y)=(cos φ, sin φ)`, 활성 음원을 앞 슬롯부터 채우고 나머지는 0.

합성 데이터 생성:

```bash
python make_seld_dataset.py --out data --n 512 --n-classes 12 --clip-seconds 3.0
```

### CPU smoke 테스트

작은 합성 데이터 + 축소 설정으로 전체 파이프라인을 1분 내 검증한다.

```bash
python make_seld_dataset.py --out /tmp/seld_smoke --n 16 --seed 0
bash run_train_SeldJEPA.sh   /tmp/seld_smoke/config_pretrain.yaml
bash run_train_SeldACCDOA.sh /tmp/seld_smoke/config_seld.yaml
bash run_eval_SeldACCDOA.sh  /tmp/seld_smoke/config_seld.yaml /tmp/seld_smoke/ckpts/accdoa/seld_accdoa_final.pt
```

## 학습

```bash
# Stage-1 (자기지도, 인코더 사전학습)
bash run_train_SeldJEPA.sh config_pretrain.yaml          # NPROC_PER_NODE=N 으로 multi-GPU
# Stage-2 (지도, Multi-ACCDOA 헤드) — config_seld.yaml 의 model.encoder_ckpt 가 Stage-1 출력을 가리켜야 함
bash run_train_SeldACCDOA.sh config_seld.yaml
```

- 런처는 `train.output_dir` 에 체크포인트(`checkpoint-*/` 또는 `*.pt`)가 이미 있으면 시작을 거부한다.
- 로그: 모듈별 학습 파라미터 수, 첫 배치 첫 샘플의 **오디오 경로**·feature shape(Stage-2는 라벨 shape),
  `logging_steps` 마다 step/loss/lr(+pred/reg 또는 adpit), `save_steps` 마다 step 번호가 들어간 `.pt`.

## 평가

```bash
bash run_eval_SeldACCDOA.sh config_seld.yaml ckpts/seld_accdoa/seld_accdoa_final.pt
```

`ER_20 / F_20 / LE_CD / LR_CD / ε_SELD` 를 출력한다(낮을수록/높을수록 좋은 방향은 출력에 표기).
평가 경로는 `from_checkpoint` 로 `eval()` + 전 파라미터 `requires_grad=False` 를 강제하고
재확인 assert 를 둔다.

## 추론 예시 (스트리밍)

`SeldMultiACCDOA.predict(feat)` 는 `(B, T_label, N=3, C, 2)` 를 반환한다. 실시간 사용 시
100 ms(=feature 10 프레임) 단위로 feature 를 만들어 인코더에 흘려보내면, 인과 마스킹/좌측
패딩 덕분에 각 청크는 과거만 참조한다(청크 1개 = 라벨 프레임 1개). `decode` 로 활성/azimuth 를
얻고 같은 클래스 트랙을 30° 이내에서 통합한다.

## 참고 / 주의사항

- **FOA(2채널 한계)**: 요청의 "IPD 및 FOA" 중 **참 FOA/인텐시티 벡터는 4채널(W,X,Y,Z)이
  필요하며 2채널에서는 수학적으로 복원 불가**(omni-W 기준·고도 Z 부재). 따라서 본 구현은 2채널을
  바이노럴로 보고 **log-mel×2(암묵적 ILD) + sin/cos-IPD** 로 공간 단서를 구성하며, DOA 라벨은
  **방위각(azimuth)만**(x,y) 사용한다. 고도/전후 모호성은 본질적으로 관측 불가하다.
  (사용자 선택에 따른 결정 사항.)
- **프레임율**: STFT 100 fps → 인코더 50 fps(프론트엔드 ×2) → 라벨 10 Hz(`pool_factor=5`).
  `pool_factor = label_hop_ms / (stft_hop_ms × frontend_downsample) = 100 / (10 × 2) = 5`.
  STFT center 패딩으로 인해 `T_enc` 가 1프레임 더 나올 수 있어, 손실/평가에서 pred·target 을
  공통 최소 길이로 잘라 정렬한다.
- **위치 인코딩**: 길이 일반화를 위해 절대 sinusoidal PE 를 사용한다(상대 위치 인코딩은 향후
  개선 여지로 둔다).
- **SIGReg 분산학습**: 본 구현은 형제 `LeJEPA` 와 동일하게 per-device 통계를 사용한다(논문의
  CF 평균 all-reduce 는 대규모 multi-GPU 시 추가 개선 항목).
- **합성 데이터 단순화**: `make_seld_dataset.py` 는 free-field 2-mic(ITD 분수 지연 + 약한 ILD)
  모델이다. 실제 룸/HRTF 효과는 포함하지 않으므로 메트릭 절댓값보다 파이프라인 검증용으로 본다.
