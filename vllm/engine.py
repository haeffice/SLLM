"""AsyncLLMEngine builder.

``vllm`` is imported lazily inside ``build_engine`` so that the rest of the
backend (config, preflight, FastAPI app import for tests) works on the
AVX2-only dev box where the CPU wheel may not import.

KV cache size / OMP thread binding / tcmalloc preload are NOT engine args —
they are read by the vLLM C++/OpenMP runtime at process init from
``VLLM_CPU_KVCACHE_SPACE`` / ``VLLM_CPU_OMP_THREADS_BIND`` / ``LD_PRELOAD``,
which ``run.sh`` exports before starting uvicorn.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from config import Config

log = logging.getLogger("be.engine")


@dataclass
class EngineHandle:
    engine: object          # vllm.engine.async_llm_engine.AsyncLLMEngine
    model_len: int
    tp_size: int
    dtype: str


def _model_len_from_config(model_path: str, override: int) -> int:
    if override:
        return override
    try:
        with open(os.path.join(model_path, "config.json")) as fh:
            c = json.load(fh)
        return int(c.get("max_position_embeddings") or 4096)
    except (OSError, ValueError):
        return 4096


def build_engine(cfg: Config, rec_tp_size: int) -> EngineHandle:
    """Blocking — call inside ``loop.run_in_executor``."""
    # lazy import: only here do we touch the (AVX512) CPU wheel
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs

    if cfg.block_size % 32 != 0:
        raise ValueError(f"block_size must be a multiple of 32, got {cfg.block_size}")

    tp_size = cfg.tp_size or rec_tp_size
    model_len = _model_len_from_config(cfg.model_path, cfg.max_model_len)

    args = AsyncEngineArgs(
        model=cfg.model_path,
        dtype="bfloat16" if cfg.dtype in ("", "auto") else cfg.dtype,
        tensor_parallel_size=tp_size,
        max_num_seqs=cfg.max_num_seqs,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        block_size=cfg.block_size,
        max_model_len=model_len,
        quantization=cfg.quantization or None,
        device="cpu",
        trust_remote_code=cfg.trust_remote_code,
        enforce_eager=cfg.enforce_eager,
        disable_log_requests=True,  # we keep our own access log
    )
    log.info(
        "building AsyncLLMEngine — model=%s tp=%d dtype=%s max_model_len=%d",
        cfg.model_path, tp_size, args.dtype, model_len,
    )
    engine = AsyncLLMEngine.from_engine_args(args)
    return EngineHandle(engine, model_len, tp_size, args.dtype)
