import io

import torchaudio
from fastapi import FastAPI, HTTPException, Request

from processor import process_stereo

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/localize")
async def localize(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty request body")
    try:
        waveform, sample_rate = torchaudio.load_with_torchcodec(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to decode audio: {e}")
    return process_stereo(waveform, sample_rate)
