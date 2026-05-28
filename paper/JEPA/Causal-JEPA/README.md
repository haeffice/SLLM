# Causal-JEPA — World Models through Object-Level Latent Interventions (arXiv:2602.11389)

물체(object) 단위로 masked joint-embedding prediction을 수행하는
**Causal-JEPA**의 PyTorch 2.8 자가완결(self-contained) 재구현. 각 프레임을
소수의 object-centric **slot**으로 인코딩하고, JEPA predictor가 **가려진
물체(whole object)** 를 *나머지 물체들로부터* 추론하게 한다. 물체 하나를
통째로 가리는 것은 그 물체의 **현재 상태를 제거하되 정체성(identity)은
보존**하는 *잠재 개입(latent intervention)* 이며, 이는 predictor가 patch
단위 지름길 대신 **물체 간 상호작용**을 추론하도록 강제한다(논문의 인과적
귀납 편향).

- **Paper:** *Causal-JEPA: Learning World Models through Object-Level Latent
  Interventions*, 2026 — https://arxiv.org/abs/2602.11389
- **Lineage:** I-JEPA (2301.08243) → V-JEPA (2404.08471) →
  object-centric JEPA (**Causal-JEPA**).

## 이 논문을 선정한 이유

JEPA 최신 트렌드의 또 다른 축은 **world model / planning** 이며, 그 중에서도
"무엇이 무엇과 상호작용하는가"를 다루는 **object-centric + causal** 방향이
부상하고 있다. Causal-JEPA는 slot 단위 마스킹을 *latent intervention* 으로
정식화해 상호작용 추론을 필수로 만든 점이 핵심 신선도이며, latent 토큰을
patch 대비 ~1% 수준으로 줄여 MPC planning을 8× 이상 가속한다.

## 핵심 구성요소 (자가완결)

`paper/JEPA`의 모든 디렉터리가 외부 가중치 다운로드 없이 CPU에서 smoke
가능한 self-contained 구현이므로, 논문이 사용한 frozen DINOv2/VideoSAUR
slot 인코더 대신 **학습형 Slot Attention**(Locatello 2020) 오토인코더로
대체한다. spatial-broadcast decoder의 재구성 손실이 slot을 object-like하게
만들어 처음부터(from scratch) 학습 가능하다. `freeze_encoder` +
`init_from`으로 논문의 frozen-`g` 프로토콜도 재현할 수 있다.

| 구성 | 설명 |
|---|---|
| **Slot 인코더** | CNN(/8) + soft-position-embed → Slot Attention → N개 slot (dim 128) |
| **Broadcast decoder** | slot→RGBA broadcast + alpha 합성 재구성 (object 학습 신호) |
| **Slot predictor** | slot 시퀀스(T·N 토큰) 위의 masked ViT (6L, 16heads, mlp 2048 — 논문) |
| **Object-level masking** | history 프레임마다 0~`max_masked_slots`개 물체를 통째로 마스킹 |
| **Identity anchor** | 가려진 slot = `proj(slot@t0) + mask_token` (정체성 보존, 상태 제거) |
| **Loss** | masked-history MSE + future MSE (`L_history`+`L_future`) + recon aux |

## 구성 파일

| 파일 | 역할 |
|---|---|
| `CausalJEPA.py` | 모델+트레이너(슬롯 인코더/디코더/predictor/마스킹/손실) + `from_checkpoint` + `encode_slots`/`rollout`/`goal_distance` |
| `train_causal_jepa.py` | `transformers.Trainer` 기반 SSL 학습 (Dataset/콜백) |
| `eval_causal_jepa.py` | latent rollout — future slot MSE + MPC goal distance |
| `config.yaml` | 학습/평가 단일 설정 |
| `run_train_CausalJEPA.sh` | torchrun 런처 (기존 checkpoint 존재 시 시작 거부) |
| `run_eval_CausalJEPA.sh` | 평가 런처 |
| `make_synthetic_manifest.py` | CPU 스모크용 합성 moving-shapes 클립 생성 |
| `requirements.txt` | 의존성 (torch==2.8.0, CPU) |

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt
```

## 데이터셋

라벨이 필요 없는 비디오 SSL. 매니페스트:

```json
{"data": [{"video_id": "rel/path/clip0001.npy"}, ...]}
```

`video_path = data.video_root / video_id`. `.npy`((T,H,W,3) uint8/float 또는
(3,T,H,W) float) 클립을 프레임 단위로 인코딩한다. 논문 데이터셋:

- **CLEVRER** (다물체 상호작용/VQA): http://clevrer.csail.mit.edu/
- **Push-T** (로봇 평면 조작/planning): diffusion-policy 벤치마크.

`num_frames = history_len + future` 윈도로 잘려 학습된다.

## 학습

```bash
bash run_train_CausalJEPA.sh config.yaml
NPROC_PER_NODE=4 bash run_train_CausalJEPA.sh config.yaml   # 멀티-GPU
```

학습 로그: 모듈별 학습 파라미터 수(encoder/slot_attn/decoder/predictor) ·
첫 배치 첫 샘플의 **비디오 경로**(feed 전) · `logging_steps`마다 step/
train_loss/**pred_loss**/**recon_loss**/**history_loss**/**future_loss**/lr ·
`save_steps`마다 step 번호가 들어간 `causal_jepa_step{N}.pt` 저장
(=`CausalJEPA.from_checkpoint`로 로드). `init_from` 0개 로드면 즉시 중단.

## CPU 스모크 테스트

```bash
python make_synthetic_manifest.py --out /tmp/cjepa_smoke
bash run_train_CausalJEPA.sh /tmp/cjepa_smoke/config.yaml
bash run_eval_CausalJEPA.sh /tmp/cjepa_smoke/config.yaml \
     /tmp/cjepa_smoke/ckpts/causal_jepa_step2.pt
```

## 평가 (latent rollout / planning)

```bash
bash run_eval_CausalJEPA.sh config.yaml /path/to/ckpts/causal_jepa/causal_jepa_step2000.pt
```

`history_len` 프레임을 컨텍스트로 predictor를 rollout하여 (1) **future
slot-prediction MSE**, (2) 마지막 프레임에 대한 **MPC latent goal distance**
`‖Ŝ_T − S_goal‖²` 를 출력한다. `from_checkpoint`이 `model.eval()` +
`requires_grad=False`를 적용하며, 한 개의 텐서도 로드되지 않으면 즉시 예외.
