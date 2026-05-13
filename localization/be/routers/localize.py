import io
import logging

import torchaudio
from fastapi import APIRouter, HTTPException, Request

from processor import process_stereo

log = logging.getLogger("be.localize")
router = APIRouter()


@router.post("/localize")
async def localize_endpoint(request: Request):
    data = await request.body()
    if not data:
        log.info("localize rejected — empty body")
        raise HTTPException(status_code=400, detail="empty request body")
    try:
        waveform, sample_rate = torchaudio.load_with_torchcodec(io.BytesIO(data))
    except Exception as e:
        log.warning("localize decode failed: %s", e)
        raise HTTPException(status_code=400, detail=f"failed to decode audio: {e}")

    log.info(
        "localize: shape=%s, sr=%d Hz, bytes=%d",
        tuple(waveform.shape),
        sample_rate,
        len(data),
    )
    result = process_stereo(waveform, sample_rate)
    log.info("localize result: %s", result)
    return result
