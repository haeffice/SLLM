import io
from pathlib import Path

import torchaudio
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from inference import infer

FE_DIR = Path(__file__).resolve().parent.parent / "fe"

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer")
async def infer_endpoint(request: Request):
    wav_bytes = await request.body()
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="empty request body")
    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(wav_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to decode audio: {e}")
    return infer(waveform, sample_rate)


app.mount("/", StaticFiles(directory=str(FE_DIR), html=True), name="fe")
