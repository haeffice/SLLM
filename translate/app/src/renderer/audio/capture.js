// Real-time audio capture, ported from demo/fe/app.js.
// Captures from the mic or Windows system-audio (loopback), resamples to
// 16 kHz mono, and flushes 720 ms WAV chunks via an onChunk callback.
window.SLLM = window.SLLM || {};

SLLM.AudioCapture = (() => {
  const TARGET_SAMPLE_RATE = 16000;
  const CHUNK_MS = 720;
  const CHUNK_SAMPLES_TARGET = Math.round((TARGET_SAMPLE_RATE * CHUNK_MS) / 1000); // 11520

  // --- ported verbatim from demo/fe/app.js -------------------------------
  function resampleLinear(input, inputRate, outputRate) {
    if (inputRate === outputRate) return input;
    const ratio = inputRate / outputRate;
    const outLen = Math.floor(input.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i * ratio;
      const i0 = Math.floor(srcIdx);
      const i1 = Math.min(i0 + 1, input.length - 1);
      const frac = srcIdx - i0;
      out[i] = input[i0] * (1 - frac) + input[i1] * frac;
    }
    return out;
  }

  function encodeWAV(samples, sampleRate) {
    const byteLen = samples.length * 2;
    const buffer = new ArrayBuffer(44 + byteLen);
    const view = new DataView(buffer);
    const writeStr = (o, s) => {
      for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i));
    };
    writeStr(0, "RIFF");
    view.setUint32(4, 36 + byteLen, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, byteLen, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      off += 2;
    }
    return new Blob([buffer], { type: "audio/wav" });
  }
  // -----------------------------------------------------------------------

  function create(onChunk) {
    let audioCtx = null;
    let mediaStream = null;
    let sourceNode = null;
    let processorNode = null;
    let muteGain = null;
    let frames = [];
    let frameSamples = 0;

    function flushReadyChunks() {
      const nativeRate = audioCtx.sampleRate;
      const thresholdNative = Math.round((nativeRate * CHUNK_MS) / 1000);

      while (frameSamples >= thresholdNative) {
        const merged = new Float32Array(frameSamples);
        let off = 0;
        for (const f of frames) {
          merged.set(f, off);
          off += f.length;
        }
        const chunkNative = merged.slice(0, thresholdNative);
        const tail = merged.slice(thresholdNative);
        frames = tail.length > 0 ? [tail] : [];
        frameSamples = tail.length;

        // System-audio mix is typically 44.1/48 kHz, so always resample from
        // the actual context rate — never assume it is already 16 kHz.
        const chunk16k = resampleLinear(chunkNative, nativeRate, TARGET_SAMPLE_RATE);
        const fixed =
          chunk16k.length === CHUNK_SAMPLES_TARGET
            ? chunk16k
            : chunk16k.subarray(0, Math.min(chunk16k.length, CHUNK_SAMPLES_TARGET));

        onChunk(encodeWAV(fixed, TARGET_SAMPLE_RATE));
      }
    }

    async function getStream(source, sampleRate) {
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
          sampleRate: sampleRate || TARGET_SAMPLE_RATE,
        },
      });
    }

    async function start(source, sampleRate) {
      mediaStream = await getStream(source, sampleRate);

      const Ctx = window.AudioContext || window.webkitAudioContext;
      audioCtx = new Ctx({ sampleRate: sampleRate || TARGET_SAMPLE_RATE });
      if (audioCtx.state === "suspended") await audioCtx.resume();

      sourceNode = audioCtx.createMediaStreamSource(mediaStream);
      processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
      processorNode.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        frames.push(new Float32Array(input));
        frameSamples += input.length;
        flushReadyChunks();
      };

      muteGain = audioCtx.createGain();
      muteGain.gain.value = 0;
      sourceNode.connect(processorNode);
      processorNode.connect(muteGain);
      muteGain.connect(audioCtx.destination);
    }

    async function stop() {
      try {
        if (processorNode) processorNode.disconnect();
        if (muteGain) muteGain.disconnect();
        if (sourceNode) sourceNode.disconnect();
        if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
        if (audioCtx) await audioCtx.close();
      } catch (_) {}
      audioCtx = null;
      mediaStream = null;
      sourceNode = null;
      processorNode = null;
      muteGain = null;
      frames = [];
      frameSamples = 0;
    }

    return { start, stop };
  }

  return { create };
})();
