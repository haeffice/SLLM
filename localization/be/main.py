import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm import load_model
from routers import inference, localize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("be")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading LLM model...")
    app.state.llm_model = load_model()
    log.info("LLM model ready")
    try:
        yield
    finally:
        app.state.llm_model = None
        log.info("LLM model unloaded")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": getattr(app.state, "llm_model", None) is not None,
    }


app.include_router(localize.router)
app.include_router(inference.router)
