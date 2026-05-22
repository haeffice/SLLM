# LLM-JEPA — Joint-Embedding Predictive Objective for LLMs (arXiv:2509.14252)

Self-contained PyTorch 2.8 / `transformers` reimplementation of
**LLM-JEPA**: a fine-tuning (and pre-training) objective that augments the
standard next-token cross-entropy with a JEPA term computed entirely in
embedding space over two *views* of an example (e.g. a natural-language
`Text` and its `Code`/SQL/regex).

- **Paper:** *LLM-JEPA: Large Language Models Meet Joint Embedding
  Predictive Architectures*, Huang, LeCun et al., Sept 2025 —
  https://arxiv.org/abs/2509.14252
- **Official code (referenced heavily for the loss/predictor design):**
  https://github.com/rbalestr-lab/llm-jepa  (Apache-2.0)
  - alt mirror: https://github.com/hytopoulos/LLM-JEPA
- **Lineage:** I-JEPA (arXiv:2301.08243) → V-JEPA / V-JEPA 2
  (`../VJEPA2`) → LLM-JEPA (JEPA applied to language models).

## 이 논문을 선정한 이유

`paper/`에는 오디오(WavJEPA/SpatialWavJEPA/BAT)와 비디오(V-JEPA 2) JEPA가
구현되어 있어, **언어(LLM) 도메인의 JEPA**를 추가해 JEPA 계열을 modality
전반으로 완성하고자 LLM-JEPA를 선정했습니다. 구체적 근거:

1. **최신성 (2025):** 2025년 9월 공개된 최신 논문으로 TODO의 "최신(25년도~)"
   조건을 충족합니다. (저자에 Yann LeCun 포함)
2. **성능 향상:** 표준 LLM 학습 목적함수 대비 NL-RX, GSM8K, Spider,
   RottenTomatoes 등 여러 데이터셋·여러 모델군(Llama3, OpenELM, Gemma2,
   Olmo)에서 일관되게 유의미한 성능 향상을 보고하며, 오버피팅에도
   강건합니다 (LoRA/full-FT 모두에서).
3. **관심도:** 공식 구현(`rbalestr-lab/llm-jepa`, Apache-2.0)이 공개되어
   있고 HuggingFace Papers에 등재되는 등 JEPA의 LLM 확장으로 빠르게
   주목받고 있습니다.
4. **SLLM 적합성:** 본 repository는 Speech/Multimodal **LLM** 프로젝트로,
   LLM 학습 목적함수를 직접 개선하는 LLM-JEPA가 가장 직접적으로
   부합합니다. 추가 가중치 없이(예측자가 backbone과 weight-tied) 적용
   가능해 기존 LLM 파이프라인에 그대로 얹을 수 있습니다.
5. **공개 코드 존재:** 손실/예측자 설계를 공식 구현 기준으로 충실히
   재현했습니다(아래 참고).

## 방법 (구현 기준)

| 기호 | 정의 |
|---|---|
| `Enc(s)` | backbone의 **마지막 레이어**, **마지막 비-pad 토큰** hidden state |
| `Pred(Enc(Text))` | 동일 backbone(가중치 공유)에 `Text` + k개의 학습형 `[predictor_i]` 토큰을 붙여 forward → 마지막 predictor 토큰의 hidden state |
| `L_JEPA` | `1 − cos(Pred(Enc(Text)), Enc(Code))` (기본; `l2`/`mse`/`infonce` 선택 가능, target은 stop-grad) |
| `L` | `gamma · L_LM + lambda · L_JEPA` |

기본값(공식 구현과 동일): `gamma=1.0`, `lambda(lbd)=0.1`, `k(num_predictors)=1`,
cosine objective, LoRA/full-FT 선택 가능.

예측자는 backbone과 **weight-tied**이며 추가 파라미터는 임베딩 테이블에
추가되는 k개의 predictor-token 행뿐입니다.

## 구성 파일

| 파일 | 역할 |
|---|---|
| `LLMJEPA.py` | 모델 클래스: AutoModelForCausalLM backbone + tied predictor + JEPA/LM 손실 + LoRA + `embed()` 추론 헬퍼 + offline tiny 모드 |
| `train_llmjepa.py` | `transformers.Trainer` 기반 fine-tuning, view-pair Dataset/DataLoader, 콜백 |
| `config.yaml` | 학습 단일 설정 파일 |
| `run_train_LLMJEPA.sh` | torchrun 런처 (기존 checkpoint 존재 시 시작 거부) |
| `make_synthetic_manifest.py` | CPU 스모크용 합성 view-pair + tiny config 생성 |
| `requirements.txt` | 의존성 (torch==2.8.0 기준, CPU) |

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt
```

## 데이터셋

LLM-JEPA는 (Text, Code) **뷰 쌍**이 필요합니다. JSONL, 한 줄당 한 예제:

```json
{"text": "List the names of all singers.", "code": "SELECT name FROM singer;"}
```

또는 chat 형식:

```json
{"messages": [{"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}]}
```

논문에서 사용한 공개 데이터셋(다운로드 경로):

- **Spider** (NL → SQL): https://yale-lily.github.io/spider
  (HF: `xlangai/spider`) — `text`=question, `code`=query.
- **GSM8K** (수학 문제 → 풀이): https://huggingface.co/datasets/openai/gsm8k
  — `text`=question, `code`=answer.
- **NL-RX-SYNTH** (자연어 → 정규식):
  https://github.com/nicholaslocascio/deep-regex
  (HF: `nl-rx`) — `text`=설명, `code`=regex.
- **Rotten Tomatoes** (리뷰 → 감성):
  https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
  — `text`=review, `code`=label 문자열.

데이터셋을 받아 위 JSONL 스키마로 변환 후 `config.yaml`의
`data.train_file` / `data.valid_file`을 가리키면 됩니다. 공식 구현
`rbalestr-lab/llm-jepa`의 `datasets/`에 변환된 JSONL 예시가 있습니다.

## 학습

```bash
# config.yaml 의 model.model_name 에 임의의 HF causal-LM repo id 지정
bash run_train_LLMJEPA.sh config.yaml
# 멀티-GPU:
NPROC_PER_NODE=4 bash run_train_LLMJEPA.sh config.yaml
```

모든 인자는 `config.yaml` 한 파일에 정리되어 있고, 셸 스크립트는
`train.output_dir`에 checkpoint가 이미 있으면 학습을 시작하지 않습니다.
HF checkpoint와 함께 step 번호가 들어간 배포용 backbone 디렉터리
(`backbone_step{N}/`, LoRA면 adapter)가 저장되어
`AutoModelForCausalLM`/`peft`로 바로 로드됩니다.

학습 로그(요청 사항 충족):
- 학습 가능 파라미터 수 (학습 시작 시)
- 첫 배치 첫 샘플의 **토크나이즈 전 prompt** (Text/Code 뷰, 모델 feed 전)
- `logging_steps`마다 step / train_loss / lm_loss / jepa_loss /
  (valid_loss) / lr
- `save_steps`마다 step 번호가 들어간 checkpoint 저장
- `init_from` 워밍스타트 시 0개 로드면 즉시 중단

## CPU 스모크 테스트 (오프라인)

```bash
python make_synthetic_manifest.py --out /tmp/llmjepa_smoke
bash run_train_LLMJEPA.sh /tmp/llmjepa_smoke/config.yaml
```

스모크 config는 `model_name="__tiny__"` → 네트워크/다운로드 없이
초소형 random `LlamaForCausalLM` + byte 토크나이저를 만들어 CPU에서 수
step 만에 끝납니다 (lm_loss / jepa_loss 모두 감소 확인됨).

## 추론 (frozen embedding)

```python
from LLMJEPA import LLMJEPA, LLMJEPAConfig
m = LLMJEPA(LLMJEPAConfig(model_name="meta-llama/Llama-3.2-1B"))
emb = m.embed(["a sentence", "another"])   # model.eval() + requires_grad=False 적용됨
```
