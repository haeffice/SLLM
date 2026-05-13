import asyncio
import logging
import time
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, Request

from llm import default_model_id, enabled_model_ids, load_one
from routers import inference, localize

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
    ids = enabled_model_ids()
    app.state.models = {}
    app.state.model_status = {mid: ModelStatus.LOADING for mid in ids}
    app.state.model_errors = {mid: None for mid in ids}
    app.state.default_model_id = default_model_id()

    log.info(
        "startup — enabled models: %s, default: %s",
        ids,
        app.state.default_model_id,
    )

    async def _load_all():
        loop = asyncio.get_running_loop()
        for mid in ids:
            t0 = time.monotonic()
            try:
                model = await loop.run_in_executor(None, load_one, mid)
                app.state.models[mid] = model
                app.state.model_status[mid] = ModelStatus.READY
                log.info("model %s ready (%.1fs)", mid, time.monotonic() - t0)
            except Exception as e:
                app.state.model_status[mid] = ModelStatus.FAILED
                app.state.model_errors[mid] = str(e)
                log.exception(
                    "model %s load failed after %.1fs: %s",
                    mid,
                    time.monotonic() - t0,
                    e,
                )

    app.state._load_task = asyncio.create_task(_load_all())

    try:
        yield
    finally:
        log.info("shutdown — releasing models")
        app.state.models.clear()
        for mid in ids:
            app.state.model_status[mid] = ModelStatus.LOADING
        if not app.state._load_task.done():
            app.state._load_task.cancel()


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
    return {
        "status": "ok",
        "default_model": request.app.state.default_model_id,
        "models": {
            mid: {
                "status": s.value,
                "error": request.app.state.model_errors.get(mid),
                "device": _device_for_loaded(request.app.state.models.get(mid)),
            }
            for mid, s in request.app.state.model_status.items()
        },
    }


def _device_for_loaded(model) -> str | None:
    if model is None:
        return None
    dev = getattr(model, "device", None)
    return str(dev) if dev is not None else None


app.include_router(localize.router)
app.include_router(inference.router)
