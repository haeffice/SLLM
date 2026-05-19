"""Environment-driven configuration for the CPU vLLM backend.

All knobs are read once at import time from environment variables with
sensible defaults, following the ``${VAR:-default}`` convention used by
``localization/be``. Process-level vLLM env vars (``VLLM_CPU_*``,
``LD_PRELOAD``) MUST be exported before the Python process starts — the
C++/OpenMP runtime reads them at init — so they are only surfaced here
read-only for logging; ``run.sh`` is responsible for setting them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger("be.config")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("invalid int for %s=%r, using default %d", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class Config:
    # --- server binding (same defaults as localization/be) ---------------
    host: str = field(default_factory=lambda: _env("BE_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("BE_PORT", 9001))

    # --- model / engine --------------------------------------------------
    model_path: str = field(default_factory=lambda: _env("VLLM_BE_MODEL_PATH"))
    dtype: str = field(default_factory=lambda: _env("VLLM_BE_DTYPE", "bfloat16"))
    # empty -> auto (number of NUMA nodes, decided by preflight)
    tp_size: int = field(default_factory=lambda: _env_int("VLLM_BE_TP_SIZE", 0))
    max_num_seqs: int = field(default_factory=lambda: _env_int("VLLM_BE_MAX_NUM_SEQS", 256))
    max_num_batched_tokens: int = field(
        default_factory=lambda: _env_int("VLLM_BE_MAX_NUM_BATCHED_TOKENS", 4096)
    )
    block_size: int = field(default_factory=lambda: _env_int("VLLM_BE_BLOCK_SIZE", 128))
    # empty -> taken from the model's config.json
    max_model_len: int = field(default_factory=lambda: _env_int("VLLM_BE_MAX_MODEL_LEN", 0))
    quantization: str = field(default_factory=lambda: _env("VLLM_BE_QUANTIZATION"))
    trust_remote_code: bool = field(
        default_factory=lambda: _env_bool("VLLM_BE_TRUST_REMOTE_CODE", False)
    )
    enforce_eager: bool = field(
        default_factory=lambda: _env_bool("VLLM_BE_ENFORCE_EAGER", True)
    )

    # --- preflight -------------------------------------------------------
    # warn -> log and continue; enforce -> abort startup on FAIL; off -> skip
    preflight_mode: str = field(
        default_factory=lambda: _env("VLLM_BE_PREFLIGHT_MODE", "warn").lower()
    )

    # --- process-level vLLM env (set by run.sh; read-only here) ----------
    kvcache_space: str = field(default_factory=lambda: _env("VLLM_CPU_KVCACHE_SPACE"))
    omp_threads_bind: str = field(default_factory=lambda: _env("VLLM_CPU_OMP_THREADS_BIND"))
    reserved_cpu: str = field(default_factory=lambda: _env("VLLM_CPU_NUM_OF_RESERVED_CPU"))
    ld_preload: str = field(default_factory=lambda: _env("LD_PRELOAD"))

    def __post_init__(self) -> None:
        if self.block_size % 32 != 0:
            raise ValueError(
                f"VLLM_BE_BLOCK_SIZE must be a multiple of 32, got {self.block_size}"
            )

    def log_summary(self) -> None:
        log.info(
            "config — model=%s dtype=%s tp=%s max_num_seqs=%d "
            "max_num_batched_tokens=%d block_size=%d max_model_len=%s "
            "quant=%s enforce_eager=%s preflight=%s",
            self.model_path or "(unset)",
            self.dtype,
            self.tp_size or "auto",
            self.max_num_seqs,
            self.max_num_batched_tokens,
            self.block_size,
            self.max_model_len or "(from config.json)",
            self.quantization or "(none)",
            self.enforce_eager,
            self.preflight_mode,
        )
        log.info(
            "process env — VLLM_CPU_KVCACHE_SPACE=%s VLLM_CPU_OMP_THREADS_BIND=%s "
            "VLLM_CPU_NUM_OF_RESERVED_CPU=%s LD_PRELOAD=%s",
            self.kvcache_space or "(unset)",
            self.omp_threads_bind or "(unset)",
            self.reserved_cpu or "(unset)",
            self.ld_preload or "(unset)",
        )


cfg = Config()
