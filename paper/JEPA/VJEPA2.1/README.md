# V-JEPA 2.1 — Unlocking Dense Features in Video SSL (arXiv:2603.14482)

V-JEPA 2의 학습 레시피를 개선해 **고품질·시간 일관성 있는 dense
per-token feature**를 얻는 V-JEPA 2.1의 PyTorch 2.8 재구현. 배포되는
인코더 구조는 V-JEPA 2와 **동일**하고, 바뀌는 것은 SSL **학습 목적함수**
뿐이므로 본 디렉터리는 형제 디렉터리 `../VJEPA2`의 인코더·학습 기반을
그대로 import 하고 그 위에 2.1 레시피만 얹는다.

- **Paper:** *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised
  Learning*, Mur-Labadia et al., Meta, 2026-03 — https://arxiv.org/abs/2603.14482
- **Lineage:** I-JEPA (2301.08243) → V-JEPA (2404.08471) → V-JEPA 2
  (2506.09985, `../VJEPA2`) → **V-JEPA 2.1**.

## 이 논문을 선정한 이유

JEPA 최신 트렌드 리서치 결과, 2026년 JEPA 계열의 핵심 축 중 하나가
**"dense feature"** 였다. V-JEPA 2는 전역(global) 비디오 이해에는
강했지만, 손실이 *masked 토큰에만* 적용되어 visible context 토큰이 국소
공간 정보를 잃고 전역 aggregator로 퇴화하는 한계가 있었다. V-JEPA 2.1은
이를 정면으로 해결하는 가장 최신(2026-03) 후속작이며, 이미 구현된
`../VJEPA2`를 최소 변경으로 확장해 검증할 수 있어 선정했다.

## 무엇이 바뀌나 — 2.1의 네 가지 재료

| # | 재료 | 본 구현 위치 |
|---|---|---|
| (i) | **Dense Predictive Loss** — masked 토큰뿐 아니라 **모든 토큰**(context∪masked)에 대해 EMA-teacher feature를 예측. context는 `context_loss_weight`로 약하게. | `VJEPA21_Trainer.py` `forward` |
| (ii) | **Deep Self-Supervision** — 최종 레이어뿐 아니라 **여러 중간 인코더 깊이**의 teacher feature를 추가로 예측(깊이별 경량 linear head). | `VJEPA21_Trainer.py` `deep_heads` / `_teacher_final_and_depths` |
| (iii) | **Multi-modal tokenizer** — 이미지를 1-프레임 클립으로 받아 비디오와 통합 학습(단일 프레임 `.npy`를 `num_frames`로 타일링). | `train_vjepa21.py` 데이터셋(`../VJEPA2`의 `VJEPA2VideoDataset` 재사용) |
| (iv) | **Scaling** — variant/차원/스텝을 config로 노출. | `config.yaml` |

(i)·(ii)가 SSL 본질의 변경이라 핵심으로 구현하고, (iii)·(iv)는
데이터/설정 레벨로 처리한다.

## 구성 파일

| 파일 | 역할 |
|---|---|
| `VJEPA21.py` | 추론 인코더(`VJEPA2` 상속) + `get_dense_features` + `from_checkpoint` |
| `VJEPA21_Trainer.py` | `VJEPA21Trainer(VJEPA2Trainer)` — dense loss + deep self-supervision |
| `train_vjepa21.py` | `transformers.Trainer` 기반 SSL 학습 (Dataset/콜백은 `../VJEPA2` 재사용) |
| `eval_vjepa21.py` | frozen 인코더 attentive-probe top-1 + dense feature export |
| `config.yaml` | 학습/평가 단일 설정 (`recipe:` 섹션이 2.1 재료) |
| `run_train_VJEPA21.sh` | torchrun 런처 (기존 checkpoint 존재 시 시작 거부) |
| `run_eval_VJEPA21.sh` | 평가 런처 |
| `make_synthetic_manifest.py` | CPU 스모크용 합성 데이터/매니페스트 (이미지 1-프레임 포함) |
| `requirements.txt` | 의존성 (torch==2.8.0, CPU) |

> **의존성:** 본 디렉터리는 `../VJEPA2`의 `VJEPA2.py`, `VJEPA2_Trainer.py`,
> `train_vjepa2.py`를 import 한다. 두 디렉터리는 같은 부모 아래 유지할 것.

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt
```

## 데이터셋

라벨이 필요 없는 순수 SSL 비디오 사전학습. 매니페스트 형식은 V-JEPA 2와
동일하며, 이미지도 1-프레임 `.npy`로 섞을 수 있다(multi-modal):

```json
{"data": [{"video_id": "rel/path/clip0001.mp4"}, ...]}
```

`video_path = data.video_root / video_id`. `.npy`((T,H,W,3) uint8 /
(3,T,H,W) float, T=1이면 이미지)와 `decord` 디코딩 영상 모두 지원.
권장 공개 데이터셋(Kinetics-400/700, SSv2, Epic-Kitchens-100, HowTo100M)은
`../VJEPA2/README.md` 참조.

평가(`eval:` 섹션)는 라벨이 필요하다:
`{"data":[{"video_id":"...","label":3}, ...]}`.

## 학습

```bash
bash run_train_VJEPA21.sh config.yaml
NPROC_PER_NODE=4 bash run_train_VJEPA21.sh config.yaml   # 멀티-GPU
```

학습 로그: 모듈별 학습 파라미터 수(+ deep-supervision 깊이) · 첫 배치
첫 샘플의 **비디오 경로** · `logging_steps`마다 step/train_loss/(valid_loss)/
lr/ema_decay/**main_loss**/**deep_loss** · `save_steps`마다 step 번호가
들어간 `vjepa21_student_step{N}.pt` 저장(=`VJEPA21.from_checkpoint`로 로드).
`init_from` 워밍스타트 시 0개 로드면 즉시 중단.

## CPU 스모크 테스트

```bash
python make_synthetic_manifest.py --out /tmp/vjepa21_smoke
bash run_train_VJEPA21.sh /tmp/vjepa21_smoke/config.yaml
# (선택) dense feature export / attentive probe
bash run_eval_VJEPA21.sh /tmp/vjepa21_smoke/config.yaml \
     /tmp/vjepa21_smoke/ckpts/vjepa21_student_step2.pt --mode dense
```

## 평가

```bash
# attentive probe top-1
bash run_eval_VJEPA21.sh config.yaml /path/to/ckpts/vjepa21/vjepa21_student_step2000.pt
# dense per-token feature export (B, N, D)
bash run_eval_VJEPA21.sh config.yaml /path/.../vjepa21_student_step2000.pt --mode dense
```

`from_checkpoint`이 `model.eval()` + `requires_grad=False`를 적용하며, 단
한 개의 텐서도 로드되지 않으면 즉시 예외를 던진다. 학생 인코더가 V-JEPA 2와
동일하므로 2.1 체크포인트와 V-JEPA 2 체크포인트는 상호 로드 가능하다.
