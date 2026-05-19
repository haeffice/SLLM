"""CPU vLLM FastAPI backend.

The vLLM AsyncLLMEngine is built ONCE in the lifespan startup (background
task), kept resident, and reused for every request via continuous batching.
Structure follows ``localization/be/main.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, Request

from config import cfg
from preflight import FAIL, run_preflight
from routers import generate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("be")


class ModelStatus(str, Enum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg.log_summary()
    app.state.engine = None
    app.state.model_status = ModelStatus.LOADING
    app.state.model_error = None
    app.state.preflight = None
    app.state._load_task = None

    # --- preflight (synchronous, pure-python, fast) ---------------------
    report = None
    if cfg.preflight_mode != "off":
        report = run_preflight(
            cfg.model_path,
            dtype=cfg.dtype,
            quantization=cfg.quantization,
            tp_size=cfg.tp_size,
            max_num_seqs=cfg.max_num_seqs,
            max_model_len=cfg.max_model_len,
        )
        app.state.preflight = report
        for c in report.checks:
            lv = log.error if c.status == FAIL else (
                log.warning if c.status == "WARN" else log.info)
            lv("preflight [%s] %s — %s", c.status, c.name, c.detail)
        log.info("preflight OVERALL: %s", report.overall)
        if cfg.preflight_mode == "enforce" and report.overall == FAIL:
            raise RuntimeError(
                "preflight FAILED and VLLM_BE_PREFLIGHT_MODE=enforce — "
                "aborting startup"
            )

    rec_tp = report.recommendations.tp_size if report else (cfg.tp_size or 1)

    # --- background engine load ----------------------------------------
    async def _load():
        from engine import build_engine

        t0 = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            handle = await loop.run_in_executor(None, build_engine, cfg, rec_tp)
            app.state.engine = handle
            app.state.model_status = ModelStatus.READY
            log.info("engine ready (%.1fs)", time.monotonic() - t0)
        except Exception as e:  # noqa: BLE001
            app.state.model_status = ModelStatus.FAILED
            app.state.model_error = str(e)
            log.exception("engine load failed after %.1fs: %s",
                          time.monotonic() - t0, e)

    app.state._load_task = asyncio.create_task(_load())

    try:
        yield
    finally:
        log.info("shutdown — releasing engine")
        task = app.state._load_task
        if task is not None and not task.done():
            task.cancel()
        handle = app.state.engine
        if handle is not None:
            shutdown = getattr(handle.engine, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:  # noqa: BLE001
                    pass
        app.state.engine = None


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.monotonic()
    response = await call_next(request)
    dur_ms = (time.monotonic() - t0) * 1000
    log.info(
        "%s %s → %d (%.1fms, client=%s)",
        request.method,
        request.url.path,
        response.status_code,
        dur_ms,
        request.client.host if request.client else "-",
    )
    return response


@app.get("/health")
def health(request: Request):
    state = request.app.state
    rep = state.preflight
    handle = state.engine
    return {
        "status": "ok",
        "model_status": state.model_status.value,
        "error": state.model_error,
        "preflight": ({"overall": rep.overall} if rep else None),
        "engine": (
            {"tp_size": handle.tp_size, "dtype": handle.dtype,
             "max_model_len": handle.model_len}
            if handle is not None else None
        ),
    }


@app.get("/preflight")
def preflight(request: Request):
    """Re-run the spec check live against the running host."""
    rep = run_preflight(
        cfg.model_path,
        dtype=cfg.dtype,
        quantization=cfg.quantization,
        tp_size=cfg.tp_size,
        max_num_seqs=cfg.max_num_seqs,
        max_model_len=cfg.max_model_len,
    )
    return dataclasses.asdict(rep)


app.include_router(generate.router)
