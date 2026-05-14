# Attribution — vendored from SLAM-LLM

This directory bundles a minimal slice of [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM)
necessary to run BAT (Bandwidth-extended Audio Transformer / Binaural Audio
Transformer — Spatial-AST + Q-Former + Llama-2-7b) inference without depending
on the upstream package.

## License

SLAM-LLM is licensed under the **Apache License, Version 2.0**. A copy of the
license is available at https://www.apache.org/licenses/LICENSE-2.0 and in the
upstream repository (`LICENSE`).

All code in this directory derived from SLAM-LLM remains under Apache-2.0; the
rest of this project's own licensing is unaffected.

## Source provenance

| Local file              | Upstream source(s) (path under `X-LANCE/SLAM-LLM@main`)                                                                                                          | What was kept                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `spatial_ast.py`        | `src/slam_llm/models/SpatialAST/SpatialAST.py`                                                                                                                  | Verbatim copy (only the relative import resolves to our local `vision_transformer`).             |
| `vision_transformer.py` | `src/slam_llm/models/SpatialAST/vision_transformer.py`                                                                                                          | Verbatim copy of the full upstream module so the `BinauralEncoder` state-dict keys line up 1:1.  |
| `projector.py`    | `src/slam_llm/models/projector.py`                                                                                                                              | `EncoderProjectorQFormer` only. `EncoderProjectorConcat` / `EncoderProjectorCov1d` dropped.    |
| `preprocess.py`   | `examples/seld_spatialsoundqa/dataset/spatial_audio_dataset.py` + `src/slam_llm/datasets/base_dataset.py`                                                       | `format_prompt`, `normalize_audio`, `padding` helper; inference-time mono→stereo / 32 kHz resampling / 10 s padding added. Reverb convolution and second-source mixing dropped (training only). |
| `model.py`        | Composes the above; orchestration logic derived from `src/slam_llm/models/slam_model.py::slam_model.forward/generate` and `examples/seld_spatialsoundqa/model/slam_model_seld.py::model_factory`. | Reimplemented as `BAT(AudioLLM)` with `load()` / `infer()`. Hydra/omegaconf/fairseq/deepspeed dependencies removed. |

Constants (`encoder_dim=768`, `llm_dim=4096`, `qformer_layers=8`, `query_len=64`,
LoRA `r=8 / alpha=32 / dropout=0.05 / target=[q_proj, v_proj]`) come from
`examples/seld_spatialsoundqa/seld_config.py`.

## What is NOT vendored

- Multi-stage training pipeline (`finetune_seld.py`, dataloaders, FSDP/DDP, deepspeed).
- Alternative encoders (Whisper, BEATs, EAT, CLAP, WavLM, HuBERT, MusicFM, Emotion2vec, AV-HuBERT).
- Alternative projectors (`EncoderProjectorConcat`, `EncoderProjectorCov1d`).
- `fairseq`, `hydra-core`, `omegaconf`, `gradio` integrations (none needed for inference).

## Known risks / open items

- **Projector ckpt prefix mapping** in `model.py::_split_projector_lora` is
  inferred from the SLAM-LLM save format (`encoder_projector.*`, `llm.*` …);
  `model.py::load` logs the top-level keys on first run so unexpected layouts
  are visible immediately.
- The encoder runs in fp32 (`@torch.no_grad`); the LLM is loaded in fp16 and we
  cast projector output to match. If a future BAT checkpoint requires bf16 the
  cast in `infer` would need a tweak.
