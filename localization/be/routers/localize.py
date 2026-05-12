import io

import torchaudio
from fastapi import APIRouter, HTTPException, Request

from processor import process_stereo

router = APIRouter()


@router.post("/localize")
async def localize_endpoint(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty request body")
    try:
        waveform, sample_rate = torchaudio.load_with_torchcodec(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to decode audio: {e}")
    return process_stereo(waveform, sample_rate)
