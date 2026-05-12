from fastapi import APIRouter, HTTPException, Request

from llm import infer

router = APIRouter()


@router.post("/inference")
async def inference_endpoint(request: Request):
    model = getattr(request.app.state, "llm_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    return infer(model, payload)
