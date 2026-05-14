import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("be.inference")
router = APIRouter()

# Multiple questions are dispatched as one batch through model.infer so the
# audio encoder runs once and the LLM generates all answers in parallel.
# Each entry probes a different task that BAT was trained on (detection /
# direction / distance / joint), so the responses give a well-rounded view
# of the same audio.
DEFAULT_QUESTIONS: list[str] = [
    "What sound events do you hear in this audio?",
    "From which direction is the sound source coming?",
    "How far away is the sound source?",
    "List all sound events along with their direction and distance.",
]


@router.post("/inference")
async def inference_endpoint(
    request: Request,
    model: str | None = Query(default=None, description="model id; defaults to server default"),
    question: list[str] | None = Query(
        default=None,
        description="repeat the parameter for multiple questions (e.g. ?question=A&question=B); "
                    "if omitted, the server uses DEFAULT_QUESTIONS",
    ),
):
    state = request.app.state
    model_id = model or state.default_model_id

    if model_id not in state.model_status:
        log.info("inference rejected — unknown model id %r", model_id)
        raise HTTPException(
            status_code=400,
            detail=f"unknown model {model_id!r}; available: {list(state.model_status)}",
        )

    status_value = state.model_status[model_id].value
    if status_value == "loading":
        log.info("inference rejected — %s still loading", model_id)
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "30"},
            content={
                "detail": f"model {model_id!r} is still loading, please retry shortly",
                "model_id": model_id,
                "model_status": "loading",
            },
        )
    if status_value == "failed":
        err = state.model_errors.get(model_id)
        log.warning("inference rejected — %s failed: %s", model_id, err)
        return JSONResponse(
            status_code=503,
            content={
                "detail": f"model {model_id!r} failed to load at startup",
                "model_id": model_id,
                "model_status": "failed",
                "error": err,
            },
        )

    instance = state.models.get(model_id)
    if instance is None:
        log.warning("inference rejected — %s is None despite status=%s", model_id, status_value)
        raise HTTPException(status_code=503, detail=f"model {model_id!r} not available")

    wav_bytes = await request.body()
    if not wav_bytes:
        log.info("inference rejected — empty body")
        raise HTTPException(status_code=400, detail="empty audio body")

    questions = question if question else DEFAULT_QUESTIONS
    log.info(
        "inference start: model=%s, bytes=%d, batch=%d, questions=%s",
        model_id,
        len(wav_bytes),
        len(questions),
        questions,
    )

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, instance.infer, wav_bytes, questions)
    except ValueError as e:
        log.warning("inference bad input on %s: %s", model_id, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("inference failed on %s", model_id)
        raise HTTPException(status_code=500, detail=f"inference error: {e}")

    if isinstance(result, dict) and "model_id" not in result:
        result["model_id"] = model_id

    if isinstance(result, dict):
        responses_list = result.get("responses")
        if isinstance(responses_list, list):
            log.info(
                "inference done: model=%s, batch=%d, responses=%s",
                model_id,
                len(responses_list),
                responses_list,
            )
        else:
            log.info(
                "inference done: model=%s, response=%s",
                model_id,
                result.get("response", ""),
            )
    return result
