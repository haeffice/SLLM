import os

import httpx
from fastapi import FastAPI, HTTPException, Request, Response

BE_URL = os.environ.get("BE_URL", "http://localhost:9001").rstrip("/")
TIMEOUT = float(os.environ.get("RELAY_TIMEOUT", "30.0"))

app = FastAPI()
client = httpx.AsyncClient(timeout=TIMEOUT)


@app.get("/health")
async def health():
    try:
        r = await client.get(f"{BE_URL}/health")
        return {"relay": "ok", "be_url": BE_URL, "be": r.json()}
    except Exception as e:
        return {"relay": "ok", "be_url": BE_URL, "be_error": str(e)}


@app.post("/localize")
async def localize(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty request body")
    content_type = request.headers.get("Content-Type", "audio/wav")
    try:
        r = await client.post(
            f"{BE_URL}/localize",
            content=data,
            headers={"Content-Type": content_type},
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"relay → BE error: {e}")
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("Content-Type", "application/json"),
    )
