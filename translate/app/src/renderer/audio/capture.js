// Real-time audio capture for WebSocket streaming.
// Captures from the mic or Windows system-audio (loopback) at 16 kHz mono,
// converts to PCM16 via an AudioWorklet, and emits SEND_BYTES (20 ms) chunks
// of raw PCM bytes through an onChunk(Uint8Array) callback. Small 20 ms frames
// keep latency low; the server buffers and decides when to emit a result.
window.SLLM = window.SLLM || {};

SLLM.AudioCapture = (() => {
  const TARGET_SAMPLE_RATE = 16000;
  const CHUNK_MS = 80;
  // PCM16 mono: 2 bytes/sample. 16000 * 2 * 0.02 = 640 bytes per 20 ms.
  const SEND_BYTES = Math.round((TARGET_SAMPLE_RATE * 2 * CHUNK_MS) / 1000);

  // The AudioWorklet processor source. Loaded via a Blob URL rather than a
  // file path because module fetches from a file:// origin are blocked by
  // Chromium; a same-origin blob: URL sidesteps that (CSP allows blob:).
  const WORKLET_SRC = `
    class PCMProcessor extends AudioWorkletProcessor {
      process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (ch && ch.length) {
          const pcm = new Int16Array(ch.length);
          for (let i = 0; i < ch.length; i++) {
            const s = Math.max(-1, Math.min(1, ch[i]));
            pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          this.port.postMessage(
            { buffer: pcm.buffer, byteLength: pcm.buffer.byteLength },
            [pcm.buffer]
          );
        }
        return true;
      }
    }
    registerProcessor("pcm-processor", PCMProcessor);
  `;

  function create(onChunk) {
    let audioCtx = null;
    let mediaStream = null;
    let sourceNode = null;
    let workletNode = null;
    let muteGain = null;
    let chunks = [];
    let bytes = 0;

    function onWorkletMessage(e) {
      chunks.push(new Uint8Array(e.data.buffer));
      bytes += e.data.byteLength;
      while (bytes >= SEND_BYTES) {
        const merged = new Uint8Array(bytes);
        let off = 0;
        for (const c of chunks) {
          merged.set(c, off);
          off += c.length;
        }
        const send = merged.slice(0, SEND_BYTES);
        const rest = merged.slice(SEND_BYTES);
        chunks = rest.length > 0 ? [rest] : [];
        bytes = rest.length;
        onChunk(send);
      }
    }

    async function getStream(source) {
      if (source === "system") {
        const stream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: true,
        });
        // We only consume audio; drop the mandatory video track immediately.
        stream.getVideoTracks().forEach((t) => t.stop());
        return stream;
      }
      return navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: TARGET_SAMPLE_RATE,
        },
      });
    }

    async function start(source) {
      mediaStream = await getStream(source);

      const Ctx = window.AudioContext || window.webkitAudioContext;
      audioCtx = new Ctx({ sampleRate: TARGET_SAMPLE_RATE });
      if (audioCtx.state === "suspended") await audioCtx.resume();

      const blobUrl = URL.createObjectURL(
        new Blob([WORKLET_SRC], { type: "application/javascript" })
      );
      try {
        await audioCtx.audioWorklet.addModule(blobUrl);
      } finally {
        URL.revokeObjectURL(blobUrl);
      }

      sourceNode = audioCtx.createMediaStreamSource(mediaStream);
      // Force a single (down-mixed) input channel so stereo sources (system
      // loopback) collapse to mono before reaching the processor.
      workletNode = new AudioWorkletNode(audioCtx, "pcm-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        channelCount: 1,
        channelCountMode: "explicit",
        channelInterpretation: "speakers",
      });
      workletNode.port.onmessage = onWorkletMessage;

      // Route through a muted gain to the destination so the graph is pulled
      // (process() runs) without the captured audio being audible.
      muteGain = audioCtx.createGain();
      muteGain.gain.value = 0;
      sourceNode.connect(workletNode);
      workletNode.connect(muteGain);
      muteGain.connect(audioCtx.destination);
    }

    async function stop() {
      try {
        if (workletNode) {
          workletNode.port.onmessage = null;
          workletNode.disconnect();
        }
        if (muteGain) muteGain.disconnect();
        if (sourceNode) sourceNode.disconnect();
        if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
        if (audioCtx) await audioCtx.close();
      } catch (_) {}
      audioCtx = null;
      mediaStream = null;
      sourceNode = null;
      workletNode = null;
      muteGain = null;
      chunks = [];
      bytes = 0;
    }

    return { start, stop };
  }

  return { create };
})();
