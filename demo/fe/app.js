const TARGET_SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 10;
const CHUNK_SAMPLES_TARGET = TARGET_SAMPLE_RATE * CHUNK_SECONDS;
const BE_URL = "/infer";

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

let audioCtx = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let muteGain = null;
let frames = [];
let frameSamples = 0;

function setStatus(s) {
  statusEl.textContent = s;
}

function appendLog(text, isError = false) {
  const li = document.createElement("li");
  li.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
  if (isError) li.classList.add("error");
  logEl.prepend(li);
}

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

async function sendChunk(samples) {
  const blob = encodeWAV(samples, TARGET_SAMPLE_RATE);
  try {
    const res = await fetch(BE_URL, {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: blob,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    appendLog(JSON.stringify(json));
  } catch (e) {
    appendLog(`전송 실패: ${e.message}`, true);
  }
}

function flushReadyChunks() {
  const nativeRate = audioCtx.sampleRate;
  const thresholdNative = Math.round(nativeRate * CHUNK_SECONDS);

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

    const chunk16k = resampleLinear(chunkNative, nativeRate, TARGET_SAMPLE_RATE);
    const fixed = chunk16k.length === CHUNK_SAMPLES_TARGET
      ? chunk16k
      : chunk16k.subarray(0, Math.min(chunk16k.length, CHUNK_SAMPLES_TARGET));

    sendChunk(fixed);
  }
}

async function start() {
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus("마이크 권한 요청 중...");

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: TARGET_SAMPLE_RATE,
      },
    });
  } catch (e) {
    appendLog(`마이크 접근 실패: ${e.message}`, true);
    setStatus("오류");
    startBtn.disabled = false;
    stopBtn.disabled = true;
    return;
  }

  const Ctx = window.AudioContext || window.webkitAudioContext;
  audioCtx = new Ctx({ sampleRate: TARGET_SAMPLE_RATE });
  if (audioCtx.state === "suspended") await audioCtx.resume();

  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  const bufSize = 4096;
  processorNode = audioCtx.createScriptProcessor(bufSize, 1, 1);

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

  setStatus(`녹음 중... (ctx=${audioCtx.sampleRate}Hz, 전송=${TARGET_SAMPLE_RATE}Hz, 청크=${CHUNK_SECONDS}s)`);
  appendLog(`녹음 시작 (AudioContext sampleRate=${audioCtx.sampleRate})`);
}

async function stop() {
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("정지 중...");

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

  setStatus("정지됨");
  appendLog("녹음 종료");
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
