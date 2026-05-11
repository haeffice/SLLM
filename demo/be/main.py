import io

import torchaudio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from inference import infer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


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
