import asyncio
import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

log = logging.getLogger("be.ws")
router = APIRouter()


def _src_tgt_for(direction: str) -> tuple[str, str] | None:
    """Map a client direction token ('en2ko'|'ko2en') to (src, tgt)."""
    if direction == "en2ko":
        return "en", "ko"
    if direction == "ko2en":
        return "ko", "en"
    return None


def _reset_decode(session: dict) -> None:
    """Drop accumulated audio/decoding context, keeping the routing config.

    Called whenever the live stream restarts or its config changes (mic off,
    direction/task switch) so a new utterance decodes from a clean slate.
    """
    src, tgt, task = session.get("src"), session.get("tgt"), session.get("task")
    session.clear()
    session.update(src=src, tgt=tgt, task=task)


@router.websocket("/ws/translate")
async def ws_translate(
    websocket: WebSocket,
    model: str | None = Query(default=None, description="model id; defaults to server default"),
    src: str = Query(default="en", description="source language code"),
    tgt: str = Query(default="ko", description="target language code"),
    task: str = Query(default="translate", description="'translate' or 'transcribe'"),
):
    """Live translation stream.

    Client → server, two frame kinds on the same socket:
        - binary frames: raw little-endian PCM16, mono, 16 kHz (the audio).
        - text frames: JSON control messages that mutate the session in place
          (no reconnect):
            ``{"type": "directionchange", "direction": "en2ko"|"ko2en"}``
            ``{"type": "taskchange", "task": "translate"|"transcribe"}``
            ``{"type": "micoff"}``  (reset the decode buffer)
    Server → client: JSON ``{"confirmed": str, "prediction": str}`` (or
        ``{"error": str}`` on a guard failure, after which the socket closes).
    """
    state = websocket.app.state
    model_id = model or state.default_model_id
    await websocket.accept()

    status = state.model_status.get(model_id)
    if status is None:
        await websocket.send_json({"error": f"unknown model {model_id!r}"})
        await websocket.close()
        return
    if status.value != "ready":
        await websocket.send_json(
            {"error": f"model {model_id!r} not ready", "model_status": status.value}
        )
        await websocket.close()
        return
    instance = state.models.get(model_id)
    if instance is None:
        await websocket.send_json({"error": f"model {model_id!r} unavailable"})
        await websocket.close()
        return

    log.info("ws connected: model=%s src=%s tgt=%s task=%s", model_id, src, tgt, task)
    session: dict = {"src": src, "tgt": tgt, "task": task}
    loop = asyncio.get_running_loop()
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                log.info("ws disconnected: model=%s", model_id)
                break

            data = message.get("bytes")
            if data is not None:
                # Audio frame: feed one PCM chunk and emit any update.
                try:
                    result = await loop.run_in_executor(
                        None,
                        instance.stream_step,
                        data,
                        session["src"],
                        session["tgt"],
                        session["task"],
                        session,
                    )
                except Exception:
                    log.exception("ws: stream_step failed on %s", model_id)
                    await websocket.send_json({"error": "translate error"})
                    continue
                if result:
                    await websocket.send_json(result)
                continue

            text = message.get("text")
            if text is None:
                continue
            # Control frame: mutate the live session in place.
            try:
                ctrl = json.loads(text)
            except Exception:
                log.warning("ws: dropping non-JSON control frame")
                continue
            ctype = ctrl.get("type")
            if ctype == "directionchange":
                mapped = _src_tgt_for(ctrl.get("direction"))
                if mapped:
                    session["src"], session["tgt"] = mapped
                    _reset_decode(session)
                    log.info("ws: direction → %s/%s", *mapped)
            elif ctype == "taskchange":
                new_task = ctrl.get("task")
                if new_task in ("translate", "transcribe"):
                    session["task"] = new_task
                    _reset_decode(session)
                    log.info("ws: task → %s", new_task)
            elif ctype == "micoff":
                _reset_decode(session)
    except WebSocketDisconnect:
        log.info("ws disconnected: model=%s", model_id)
    except Exception:
        log.exception("ws: connection error on %s", model_id)
        try:
            await websocket.close()
        except Exception:
            pass
