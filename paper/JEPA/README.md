# JEPA 계열

JEPA(Joint-Embedding Predictive Architecture) 및 공간/멀티모달 오디오 인식 계열
논문 구현 모음. 각 디렉터리는 자체 README·requirements·학습/평가 코드를 가진다.

| 디렉터리 | 논문 / 내용 |
|---|---|
| `VJEPA2` | V-JEPA 2 video encoder + JEPA trainer (self-contained). |
| `VJEPA2.1` | V-JEPA 2.1 dense-feature recipe — Dense Predictive Loss + Deep Self-Supervision on top of `VJEPA2` (arXiv:2603.14482, 2026). |
| `CAPI` | Cluster-and-predict latent patches: masked-latent MIM + Sinkhorn clustering (2025, Meta FAIR). |
| `Causal-JEPA` | Object-centric JEPA — Slot-Attention slots + object-level masking as latent intervention (arXiv:2602.11389, 2026). |
| `LLM-JEPA` | LLM-JEPA fine-tuning objective (arXiv:2509.14252). |
| `Point-JEPA` | Point-JEPA 3D point-cloud SSL (WACV 2025). |
| `EB-JEPA` | Energy-based JEPA image SSL (2026, Meta FAIR). |
| `LeJEPA` | Provable JEPA + SIGReg image SSL, heuristics-free (2025, LeCun & Balestriero). |
| `LeWorldModel` | End-to-end JEPA world model from pixels (2026, SIGReg). |
| `WavJEPA` | WavJEPA audio SSL 추론 + JEPA trainer (+10 s variant). |
| `SpatialWavJEPA` | WavJEPA-Nat 공간(binaural) 오디오 SSL 인코더 + JEPA trainer. |
| `BAT` | Spatial-AST + Q-Former + Llama-2 공간 음향 추론(SpatialSoundQA) 평가. |
| `NeuralTokenizer` | JEPA-as-a-Neural-Tokenizer — 2-stage 가역 음성 토크나이저: JEPA+DAAM 인코더 → FSQ + mixed-radix + HiFi-GAN (arXiv:2512.07168, 2025). |

> 의존성: `SpatialWavJEPA`는 형제 디렉터리 `WavJEPA`의 모듈(`WavJEPA.py`,
> `WavJEPA_Trainer.py`)을, `VJEPA2.1`은 `VJEPA2`의 모듈(`VJEPA2.py`,
> `VJEPA2_Trainer.py`, `train_vjepa2.py`)을 import 하므로 각각 같은 부모 아래
> 유지해야 한다.
> `Point-JEPA`/`LLM-JEPA`/`LeWorldModel` README의 lineage 링크(`../VJEPA2` 등)도
> 형제 관계를 전제로 한다.
