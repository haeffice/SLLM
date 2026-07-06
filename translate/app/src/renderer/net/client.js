// Server communication: a WebSocket carries the live audio stream and the
// translation results; /health stays a plain HTTP GET for model tags.
window.SLLM = window.SLLM || {};

SLLM.net = (() => {
  async function fetchHealth(base) {
    const res = await fetch(`${base}/health`, { method: "GET" });
    const json = await res.json().catch(() => ({}));
    return { status: res.status, ok: res.ok, json };
  }

  // Pull the architecture tags for the server's default model out of /health.
  function tagsFromHealth(json) {
    if (!json || !json.models) return [];
    const def = json.default_model;
    const entry = (def && json.models[def]) || Object.values(json.models)[0];
    return entry && Array.isArray(entry.tags) ? entry.tags : [];
  }

  function wsUrl(base, { src, tgt, task, model }) {
    const wsBase = base.replace(/^http/i, "ws"); // https→wss, http→ws
    const q = new URLSearchParams({ src, tgt });
    if (task) q.set("task", task);
    if (model) q.set("model", model);
    return `${wsBase}/ws?${q.toString()}`;
  }

  // The socket carries two kinds of client→server frames:
  //   - binary frames: raw little-endian PCM16, mono, 16 kHz (the audio stream).
  //   - text frames: JSON control messages that mutate the live session
  //     in place (direction/task) without tearing down the connection.
  // Server→client frames are always JSON: {confirmed, prediction} or {error}.
  function connectWs(base, params, handlers) {
    const socket = new WebSocket(wsUrl(base, params));
    socket.binaryType = "arraybuffer";
    socket.onopen = () => handlers.onOpen && handlers.onOpen();
    socket.onmessage = (e) => {
      let obj;
      try {
        obj = JSON.parse(e.data);
      } catch (_) {
        return;
      }
      handlers.onMessage && handlers.onMessage(obj);
    };
    socket.onerror = () => handlers.onError && handlers.onError();
    socket.onclose = () => handlers.onClose && handlers.onClose();

    const sendControl = (obj) => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(obj));
      }
    };

    return {
      isOpen: () => socket.readyState === WebSocket.OPEN,
      // Raw binary PCM16 — one WebSocket binary frame per 20 ms audio chunk.
      sendAudio: (bytes) => {
        if (socket.readyState === WebSocket.OPEN) socket.send(bytes);
      },
      // Flip translation direction mid-session ('en2ko' | 'ko2en').
      setDirection: (direction) => sendControl({ type: "directionchange", direction }),
      // Switch between translation and verbatim transcription.
      setTask: (task) => sendControl({ type: "taskchange", task }),
      // Push AIOptions (hyperparameters) to the server as a flat payload.
      setOptions: (options) => sendControl({ type: "optionschanged", ...options }),
      // Signal the mic was turned off so the server resets its decode buffer.
      micOff: () => sendControl({ type: "micoff" }),
      close: () => {
        try {
          socket.close();
        } catch (_) {}
      },
    };
  }

  return { fetchHealth, tagsFromHealth, connectWs };
})();
